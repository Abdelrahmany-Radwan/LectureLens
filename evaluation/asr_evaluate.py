import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from jiwer import wer
from transformers import pipeline

ROOT = Path(__file__).parent
MODEL = "openai/whisper-tiny.en"
DATASET = "hf-internal-testing/librispeech_asr_dummy"
CONFIG = "clean"
SPLIT = "validation"
SAMPLE_COUNT = 8


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sample_wer(reference: str, hypothesis: str) -> float:
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return wer(reference, hypothesis)


dataset = load_dataset(DATASET, CONFIG, split=SPLIT)
dataset = dataset.select(range(min(SAMPLE_COUNT, len(dataset))))

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL,
    device=-1,
)

references = []
hypotheses = []
samples = []
total_audio_seconds = 0.0
total_inference_seconds = 0.0

for row in dataset:
    audio = row["audio"]
    array = audio["array"]
    sampling_rate = audio["sampling_rate"]
    duration = len(array) / sampling_rate

    start = time.perf_counter()
    result = asr({"array": array, "sampling_rate": sampling_rate})
    elapsed = time.perf_counter() - start

    reference = normalize(row["text"])
    hypothesis = normalize(result["text"])

    references.append(reference)
    hypotheses.append(hypothesis)
    total_audio_seconds += duration
    total_inference_seconds += elapsed

    samples.append(
        {
            "id": row.get("id"),
            "duration_seconds": round(duration, 3),
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(sample_wer(reference, hypothesis), 4),
            "inference_seconds": round(elapsed, 3),
        }
    )

overall_wer = wer(references, hypotheses)
rtf = (
    total_inference_seconds / total_audio_seconds
    if total_audio_seconds
    else 0.0
)

results = {
    "model": MODEL,
    "dataset": DATASET,
    "config": CONFIG,
    "split": SPLIT,
    "sample_count": len(samples),
    "domain": "read English speech from LibriSpeech; this is a reproducible ASR regression benchmark, not a classroom-lecture benchmark",
    "metric": "normalized word error rate",
    "normalization": "lowercase; punctuation removed; whitespace collapsed",
    "overall_wer": round(overall_wer, 4),
    "total_audio_seconds": round(total_audio_seconds, 3),
    "total_inference_seconds": round(total_inference_seconds, 3),
    "real_time_factor": round(rtf, 4),
    "samples": samples,
}

(ROOT / "asr_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
