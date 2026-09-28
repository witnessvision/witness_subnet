"""GPU-side audio evidence for labeling pool clips. Validator code only.

Speech: faster-whisper large-v3 with word timestamps, grouped into sentences.
Sounds: AST AudioSet tags per 4-second window, with music/speech flags. The
output has the shape ``annotate.label_clip`` expects for reference annotation.

Usage: python pod_audio.py JOB.json OUT.jsonl
JOB = {"clips": [{"file", "path", "duration"}]}
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Pinned revisions: every validator labels with exactly the same audio models.
WHISPER = ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478")
AST = ("MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902")
ASR_MODEL = f"faster-whisper-large-v3-words@{WHISPER[1][:12]}"
SOUND_MODEL = f"ast-audioset-10-10-0.4593@{AST[1][:12]}"
WINDOW_S = 4.


def pcm(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def speech(whisper, samples: np.ndarray, duration: float) -> list[dict]:
    segments, _ = whisper.transcribe(samples, word_timestamps=True, vad_filter=True)
    rows = []
    for segment in segments:
        sentence: list = []
        for word in segment.words or []:
            sentence.append(word)
            # Sentence-level segments keep speech intervals tight.
            if word.word.strip().endswith((".", "?", "!", "。", "？", "！")) or word is segment.words[-1]:
                text = "".join(w.word for w in sentence).strip()
                if text:
                    rows.append({"start": round(max(0., sentence[0].start), 2),
                                 "end": round(min(sentence[-1].end, duration), 2),
                                 "text": text, "no_speech_prob": round(segment.no_speech_prob, 3)})
                sentence = []
    return [{"index": index, **row} for index, row in enumerate(sorted(rows, key=lambda row: row["start"]))]


def sounds(extractor, tagger, samples: np.ndarray, duration: float) -> list[dict]:
    labels = tagger.config.id2label
    spans = [(i * WINDOW_S, min((i + 1) * WINDOW_S, duration)) for i in range(math.ceil(duration / WINDOW_S - 1e-9))]
    chunks = []
    for start, end in spans:
        chunk = samples[int(start * 16000):int(end * 16000)]
        chunks.append(np.pad(chunk, (0, max(0, 16000 - len(chunk)))))
    inputs = extractor(chunks, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.sigmoid(tagger(input_values=inputs["input_values"].cuda().half()).logits.float()).cpu()
    rows = []
    for index, ((start, end), p) in enumerate(zip(spans, probs)):
        top = [(labels[int(i)], float(p[i])) for i in p.argsort(descending=True)[:6] if p[i] >= .2]
        names = [label.lower() for label, _ in top]

        def flag(words):
            return "yes" if any(word in name for name in names for word in words) else "no"
        rows.append({"index": index, "start": start, "end": end,
                     "sounds": ", ".join(f"{label} ({value:.2f})" for label, value in top) or "none detected",
                     "music": flag(("music", "song", "singing")), "speech": flag(("speech", "conversation", "narration"))})
    return rows


def main():
    job, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    try:
        from faster_whisper import WhisperModel
        from huggingface_hub import snapshot_download
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        whisper = WhisperModel(snapshot_download(WHISPER[0], revision=WHISPER[1]), device="cuda", compute_type="float16")
        ast = snapshot_download(AST[0], revision=AST[1], allow_patterns=["*.json", "*.safetensors"])
        extractor = ASTFeatureExtractor.from_pretrained(ast)
        tagger = ASTForAudioClassification.from_pretrained(ast, use_safetensors=True).cuda().eval().half()
    except Exception as error:
        print(f"audio_models_unavailable: {error}"[:300], file=sys.stderr)
        raise SystemExit(3)
    done = {json.loads(line)["file"] for line in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as stream:
        for clip in job["clips"]:
            if clip["file"] in done:
                continue
            samples = pcm(clip["path"])
            stream.write(json.dumps({"file": clip["file"], "speech_segments": speech(whisper, samples, clip["duration"]),
                                     "sound_windows": sounds(extractor, tagger, samples, clip["duration"]),
                                     "asr_model": ASR_MODEL, "sound_model": SOUND_MODEL}) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
