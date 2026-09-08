"""Small ctypes binding for the locally installed eSpeak NG library."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import threading
from dataclasses import dataclass

import numpy as np


class TTSUnavailable(RuntimeError):
    """Raised when a real local speech synthesizer cannot be loaded."""


@dataclass(frozen=True)
class SpeechClip:
    samples: np.ndarray
    sample_rate: int
    engine: str
    voice: str


_LOCK = threading.Lock()


def synthesize(text: str, voice: str = "en-us", rate: int = 155) -> SpeechClip:
    """Return mono int16 speech synthesized by the system libespeak-ng.

    eSpeak's retrieval callback is process-global, so calls are serialized.
    """

    library = ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak")
    if not library:
        raise TTSUnavailable(
            "libespeak-ng is required: install the distro espeak-ng runtime library"
        )

    with _LOCK:
        lib = ctypes.CDLL(library)
        callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_short),
            ctypes.c_int,
            ctypes.c_void_p,
        )
        chunks: list[np.ndarray] = []

        @callback_type
        def callback(wav: ctypes.POINTER(ctypes.c_short), count: int, _events: int) -> int:
            if count > 0 and bool(wav):
                chunks.append(np.ctypeslib.as_array(wav, shape=(count,)).copy())
            return 0

        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.espeak_Initialize.restype = ctypes.c_int
        lib.espeak_SetSynthCallback.argtypes = [callback_type]
        lib.espeak_SetSynthCallback.restype = None
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_SetVoiceByName.restype = ctypes.c_int
        lib.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.espeak_SetParameter.restype = ctypes.c_int
        lib.espeak_Synth.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_void_p,
        ]
        lib.espeak_Synth.restype = ctypes.c_int
        lib.espeak_Synchronize.argtypes = []
        lib.espeak_Synchronize.restype = ctypes.c_int
        lib.espeak_Terminate.argtypes = []
        lib.espeak_Terminate.restype = ctypes.c_int

        # AUDIO_OUTPUT_RETRIEVAL=1, espeakRATE=1, espeakCHARS_UTF8=1.
        sample_rate = int(lib.espeak_Initialize(1, 0, None, 0))
        if sample_rate <= 0:
            raise TTSUnavailable("libespeak-ng initialization failed")
        try:
            lib.espeak_SetSynthCallback(callback)
            if lib.espeak_SetVoiceByName(voice.encode("utf-8")) != 0:
                raise TTSUnavailable(f"eSpeak voice {voice!r} is unavailable")
            lib.espeak_SetParameter(1, rate, 0)
            payload = text.encode("utf-8") + b"\0"
            uid = ctypes.c_uint(0)
            result = lib.espeak_Synth(
                payload,
                len(payload),
                0,
                1,
                0,
                1,
                ctypes.byref(uid),
                None,
            )
            if result != 0 or lib.espeak_Synchronize() != 0:
                raise TTSUnavailable(f"eSpeak synthesis failed with code {result}")
        finally:
            lib.espeak_Terminate()

        if not chunks:
            raise TTSUnavailable("eSpeak returned no speech samples")
        # eSpeak's callback can vary its terminal silence by a few samples across
        # repeated initializations. Canonicalize the delivered clip to the next
        # 100 ms boundary; the padded PCM is the clip that gets mixed and timed.
        samples = np.concatenate(chunks).astype(np.int16, copy=False)
        quantum = sample_rate // 10
        target = math.ceil(len(samples) / quantum) * quantum
        if target > len(samples):
            samples = np.pad(samples, (0, target - len(samples)))
        return SpeechClip(samples=samples, sample_rate=sample_rate, engine="espeak-ng", voice=voice)
