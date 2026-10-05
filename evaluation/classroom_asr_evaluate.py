import json
import re
import time
from pathlib import Path

import gdown
import librosa
from jiwer import wer
from transformers import pipeline

ROOT = Path(__file__).parent
CACHE = ROOT / ".cotacs"
CACHE.mkdir(exist_ok=True)

MODEL = "openai/whisper-tiny.en"
AUDIO_ID = "1JvS5dNmiQ5y3L5LmF9mDAMF64lKzNgdd"
TEXTGRID_ID = "1PFMD2Nx8JpzE9hFJt2X9avAvNXUyHYCW"
AUDIO_PATH = CACHE / "ATA1.wav"
TEXTGRID_PATH = CACHE / "ATA1.TextGrid"
SEGMENT_SECONDS = 120


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def download_if_needed(file_id: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    result = gdown.download(id=file_id, output=str(path), quiet=False)
    if not result or not path.exists():
        raise RuntimeError(f"Failed to download {path.name}")


def parse_textgrid_tiers(text: str):
    item_pattern = re.compile(
        r'item \[\d+\]:\s*(.*?)(?=\n\s*item \[\d+\]:|\Z)',
        re.S,
    )
    interval_pattern = re.compile(
        r'intervals \[\d+\]:\s*'
        r'xmin\s*=\s*([0-9.]+)\s*'
        r'xmax\s*=\s*([0-9.]+)\s*'
        r'text\s*=\s*"(.*?)"',
        re.S,
    )

    tiers = []
    for block in item_pattern.findall(text):
        name_match = re.search(r'name\s*=\s*"(.*?)"', block)
        class_match = re.search(r'class\s*=\s*"(.*?)"', block)
        if not name_match or not class_match or "IntervalTier" not in class_match.group(1):
            continue

        intervals = []
        for start, end, label in interval_pattern.findall(block):
            label = label.replace('""', '"').strip()
            if label:
                intervals.append((float(start), float(end), label))

        tiers.append({"name": name_match.group(1), "intervals": intervals})
    return tiers


def choose_reference_tier(tiers, end_seconds: float):
    hints = ("word", "orth", "transcript", "transcription")

    def score(tier):
        labels = [
            label
            for start, end, label in tier["intervals"]
            if start < end_seconds and end > 0
        ]
        if not labels:
            return -1

        joined = " ".join(labels)
        alpha = re.findall(r"[A-Za-z']+", joined)
        token_count = len(alpha)
        avg_words_per_label = token_count / max(len(labels), 1)
        hint_bonus = 10000 if any(h in tier["name"].lower() for h in hints) else 0
        single_symbol_penalty = sum(
            1 for label in labels if len(re.sub(r"\W", "", label)) <= 1
        )
        shape_bonus = 100 if 0.8 <= avg_words_per_label <= 5 else 0
        return hint_bonus + token_count + shape_bonus - single_symbol_penalty

    ranked = sorted(tiers, key=score, reverse=True)
    if not ranked or score(ranked[0]) < 0:
        raise RuntimeError("No usable orthographic tier found in TextGrid")
    return ranked[0]


def tier_reference(tier, end_seconds: float) -> str:
    labels = []
    for start, end, label in tier["intervals"]:
        if start >= end_seconds:
            break
        if end <= 0:
            continue
        cleaned = re.sub(r"<[^>]+>|\[[^\]]+\]", " ", label)
        cleaned = normalize(cleaned)
        if cleaned:
            labels.append(cleaned)
    return normalize(" ".join(labels))


download_if_needed(AUDIO_ID, AUDIO_PATH)
download_if_needed(TEXTGRID_ID, TEXTGRID_PATH)

textgrid_text = TEXTGRID_PATH.read_text(errors="ignore")
tiers = parse_textgrid_tiers(textgrid_text)
reference_tier = choose_reference_tier(tiers, SEGMENT_SECONDS)
reference = tier_reference(reference_tier, SEGMENT_SECONDS)

if len(reference.split()) < 20:
    raise RuntimeError(
        f"Reference extraction produced too little speech from tier {reference_tier['name']!r}"
    )

audio, sample_rate = librosa.load(
    AUDIO_PATH,
    sr=16000,
    mono=True,
    duration=SEGMENT_SECONDS,
)

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL,
    device=-1,
)

started = time.perf_counter()
output = asr(
    {"array": audio, "sampling_rate": sample_rate},
    chunk_length_s=30,
    stride_length_s=5,
)
elapsed = time.perf_counter() - started
hypothesis = normalize(output["text"])

metric = wer(reference, hypothesis)
duration = len(audio) / sample_rate
rtf = elapsed / duration if duration else 0.0

results = {
    "model": MODEL,
    "corpus": "CoTACS — Corpus of Teaching Assistant Classroom Speech",
    "speaker": "ATA1",
    "domain": "real university classroom speech",
    "corpus_license": "CC BY-NC 4.0",
    "reference_tier": reference_tier["name"],
    "segment_seconds": round(duration, 3),
    "reference_words": len(reference.split()),
    "hypothesis_words": len(hypothesis.split()),
    "normalized_wer": round(metric, 4),
    "inference_seconds": round(elapsed, 3),
    "real_time_factor": round(rtf, 4),
    "scope_note": (
        "Single-speaker classroom regression sample; not a population-level classroom ASR claim."
    ),
}

(ROOT / "classroom_asr_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
