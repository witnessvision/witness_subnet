"""Validator-owned W8A8 FP8 kernels for the pinned PyTorch/Triton runtime.

The Transformers 4.57 compressed-tensors path otherwise decompresses and runs
fake quantization. Decode uses one fused vector kernel; prefill uses CUDA FP8
matrix multiplication. No miner Python, calibration data or network access.
"""
import torch
import triton
import triton.language as tl


def _seed_extensions():
    """CPU launchers are image-owned; writable /tmp stays noexec in the sandbox."""
    import os
    from pathlib import Path
    source = Path(__file__).with_name('native_triton')
    if not source.is_dir():
        return  # Ordinary non-sandboxed GPU setup can compile its trusted code.
    cache = Path(os.environ.get('TRITON_CACHE_DIR', '/tmp/triton'))
    for library in source.glob('*/*.so'):
        if library.is_symlink():
            raise RuntimeError('CUDA untrusted_triton_extension')
        target = cache / library.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.parent.is_symlink():
            raise RuntimeError('CUDA untrusted_triton_cache')
        if target.is_symlink():
            if target.resolve() != library.resolve():
                raise RuntimeError('CUDA untrusted_triton_cache')
        elif target.exists():
            raise RuntimeError('CUDA untrusted_triton_cache')
        else:
            target.symlink_to(library)

@triton.jit
def _one(X,Q,S,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    x=tl.load(X+i,i<N,0).to(tl.float32)
    s=tl.maximum(tl.max(tl.abs(x)),1.e-12)/448.
    q=tl.minimum(tl.maximum(x/s,-448.),448.)
    tl.store(Q+i,q,i<N)
    tl.store(S,s)

@triton.jit
def _many(X,Q,S,N,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,0).to(tl.float32)
    s=tl.load(S)
    tl.store(Q+i,tl.minimum(tl.maximum(x/s,-448.),448.),i<N)

def quantize(x):
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    if x.shape[0]==1 and x.numel()<=16384:
        scale=torch.empty(1,device=x.device,dtype=torch.float32)
        _one[(1,)](x,q,scale,x.numel(),triton.next_power_of_2(x.numel()))
    else:
        scale=x.abs().amax().float().clamp_min(1e-12).reshape(1)/448.
        _many[(triton.cdiv(x.numel(),1024),)](x,q,scale,x.numel(),1024)
    return q,scale

@triton.jit
def _gemv(X,W,S,Y,K:tl.constexpr,N:tl.constexpr,BX:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    ix=tl.arange(0,BX)
    whole=tl.load(X+ix,ix<K,0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(whole)),1.e-12)/448.
    rows=tl.program_id(0)*R+tl.arange(0,R)
    kk=tl.arange(0,B)
    acc=tl.full((R,B),0.,tl.float32)
    for chunk in range(tl.cdiv(K,B)):
        k=chunk*B+kk
        x=tl.load(X+k,k<K,0).to(tl.float32)
        q=tl.minimum(tl.maximum(x/scale,-448.),448.).to(tl.float8e4nv).to(tl.float32)
        w=tl.load(W+rows[:,None]*K+k[None,:],(rows[:,None]<N)&(k[None,:]<K),0.).to(tl.float32)
        acc+=w*q[None,:]
    value=tl.sum(acc,1)*scale*tl.load(S)
    tl.store(Y+rows,value,rows<N)

def _gemv_result(x,weight,scale):
    n,k=weight.shape
    out=torch.empty((1,n),device=x.device,dtype=x.dtype)
    _gemv[(triton.cdiv(n,8),)](x,weight,scale,out,k,n,triton.next_power_of_2(k),8,1024,num_warps=4)
    return out


class NativeFP8Linear(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.register_buffer('weight', module.weight.detach())
        self.register_buffer('scale', module.weight_scale.detach().float().reshape(1))
        self.register_buffer('bias', module.bias.detach() if module.bias is not None else None)
        self.out_features, self.in_features = self.weight.shape
        if not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError('invalid_fp8_weight_scale')

    def forward(self, x):
        shape = x.shape
        a = x.reshape(-1, self.in_features).contiguous()
        if a.shape[0] == 1 and self.bias is None and self.in_features <= 16384:
            y = _gemv_result(a, self.weight, self.scale)
        else:
            q, scale = quantize(a)
            y = torch._scaled_mm(q, self.weight.t(), scale_a=scale, scale_b=self.scale,
                                 bias=self.bias, out_dtype=x.dtype, use_fast_accum=True)
        return y.reshape(*shape[:-1], self.out_features)


def install_native_fp8(model):
    """Replace only supported, already-loaded compressed-tensors linears.

    BF16 and other quantization schemes retain their original implementation.
    Kernel compilation belongs to the bounded model-load phase, before clips.
    """
    from compressed_tensors.linear.compressed_linear import CompressedLinear
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, CompressedLinear):
            continue
        scheme = module.quantization_scheme
        weights, inputs = scheme.weights, scheme.input_activations
        if (weights is None or inputs is None or weights.type != 'float' or
                weights.num_bits != 8 or weights.strategy != 'tensor' or
                not weights.symmetric or inputs.type != 'float' or inputs.num_bits != 8 or
                inputs.strategy != 'tensor' or not inputs.symmetric or not inputs.dynamic or
                scheme.output_activations is not None or
                module.weight.dtype != torch.float8_e4m3fn or not module.weight.is_contiguous() or
                module.weight_scale.numel() != 1 or any(n % 16 for n in module.weight.shape)):
            continue
        selected.append((name, module))
    if selected and torch.cuda.get_device_capability() < (8, 9):
        raise RuntimeError('CUDA native_fp8_requires_sm89_or_newer')
    if selected:
        _seed_extensions()
    warm = {}
    for name, module in selected:
        parent, _, leaf = name.rpartition('.')
        replacement = NativeFP8Linear(module)
        model.get_submodule(parent)._modules[leaf] = replacement
        warm.setdefault((replacement.in_features, replacement.out_features, replacement.bias is None), replacement)
    with torch.inference_mode():
        for module in warm.values():
            for rows in (1, 2):
                module(torch.zeros(rows, module.in_features, device=module.weight.device, dtype=torch.bfloat16))
        if warm:
            torch.cuda.synchronize()
    return model


def prepare_extensions(destination):
    """Provisioning only: compile trusted launchers without any miner or media."""
    import os
    import shutil
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='witness-triton-build-') as scratch:
        os.environ['TRITON_CACHE_DIR'] = scratch
        with torch.inference_mode():
            weight = torch.ones(16, 16, device='cuda', dtype=torch.float8_e4m3fn)
            layer = NativeFP8Linear(SimpleNamespace(weight=weight,
                    weight_scale=torch.ones(1, device='cuda'), bias=None))
            for rows in (1, 2):
                x = torch.ones(rows, 16, device='cuda', dtype=torch.bfloat16)
                quantize(x)
                layer(x)
            torch.cuda.synchronize()
        for library in Path(scratch).glob('*/*.so'):
            target = destination / library.relative_to(scratch)
            target.parent.mkdir(mode=0o755)
            shutil.copyfile(library, target)
            target.chmod(0o444)
    if len(list(destination.glob('*/*.so'))) < 3:
        raise RuntimeError('incomplete_triton_extension_bundle')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build trusted Triton CPU launchers for the pinned GPU runtime')
    parser.add_argument('--prepare-extensions', required=True)
    prepare_extensions(parser.parse_args().prepare_extensions)
