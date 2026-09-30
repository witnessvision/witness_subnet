"""Keep PyAV frame reformatters from creating a thread pool for every frame.

Torchvision retains decoded frames while converting them. With FFmpeg's
threaded swscale each retained reformatter owns a pool, exhausting bounded
workers even on ordinary clips. One decoder CPU preserves the pixels and
avoids thousands of threads; model inference keeps its own CPU/GPU settings.
This directory is the executor's fixed PYTHONPATH, never a miner model path.
"""
import ctypes
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec('av')
if spec and spec.origin:
    for path in sorted((Path(spec.origin).parent.parent / 'av.libs').glob('libavutil*.so*')):
        library = ctypes.CDLL(str(path))
        library.av_cpu_force_count.argtypes = [ctypes.c_int]
        library.av_cpu_force_count.restype = None
        library.av_cpu_force_count(1)
