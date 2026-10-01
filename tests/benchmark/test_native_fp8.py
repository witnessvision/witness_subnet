"""GPU qualification of represented FP8 arithmetic, independent of model quality."""
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA GPU required')


@pytest.mark.parametrize('rows,inside,outside',[(1,2560,9728),(1,9728,2560),(128,2560,2560)])
def test_native_fp8_matches_quantized_dense_reference(rows,inside,outside):
    from witness.benchmark.pod_fp8 import NativeFP8Linear
    torch.manual_seed(13)
    with torch.inference_mode():
        w=torch.randn(outside,inside,device='cuda',dtype=torch.bfloat16)*.01
        sw=w.float().abs().amax().reshape(1)/448
        qw=(w.float()/sw).to(torch.float8_e4m3fn)
        model=NativeFP8Linear(SimpleNamespace(weight=qw,weight_scale=sw,bias=None))
        x=torch.randn(rows,inside,device='cuda',dtype=torch.bfloat16)
        sx=x.float().abs().amax().reshape(1)/448
        qx=(x.float()/sx).clamp(-448,448).to(torch.float8_e4m3fn)
        ref=torch.nn.functional.linear(qx.float()*sx,qw.float()*sw)
        actual=model(x).float()
        error=(actual-ref).square().mean().sqrt()/ref.square().mean().sqrt()
        assert torch.isfinite(actual).all() and error<.01
        assert torch.count_nonzero(model(torch.zeros_like(x)))==0
