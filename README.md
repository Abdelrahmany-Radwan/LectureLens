# LectureLens

> **Hear it. Find it. Study it.**

LectureLens is a local-first lecture companion that turns timestamped class transcripts into searchable evidence. It is designed around one product principle: **answers should point back to where the instructor actually said it.**

**Product direction:** record/import lecture audio → build a timestamp-aware transcript index → retrieve relevant source passages with semantic + lexical ranking → jump back to the exact source time.

## Why it exists

Lecture transcription, summaries, and searchable notes are useful, but many products put the most valuable features behind recurring subscriptions. LectureLens explores how much of that workflow can be built with browser APIs and compact local ML while keeping the source evidence visible.

## Current milestone

- in-browser audio recording with `MediaRecorder`
- audio import and playback
- local browser speech-to-text with `Xenova/whisper-tiny.en`
- timestamped Whisper transcript output
- live waveform feedback
- timestamp-aware transcript parsing
- MiniLM sentence embeddings in the browser with Transformers.js
- semantic + lexical evidence ranking
- clickable source timestamps
- structured lecture outline
- quick-review study cards
- local transcript persistence
- lexical fallback if the semantic model is unavailable
- responsive, reduced-motion-aware UI

Browser speech-to-text is now implemented with **Whisper Tiny English running locally through Transformers.js**. Recorded or imported audio is decoded in-browser, transcribed into timestamped segments, and then indexed for semantic search.

## System

```text
microphone / audio file
        │
        ▼
browser recording + playback
        │
        ▼
timestamped transcript
        │
        ▼
transcript segmentation
        │
        ├──────────────┐
        ▼              ▼
MiniLM embeddings   lexical overlap
        │              │
        └──────┬───────┘
               ▼
       hybrid evidence ranker
               │
               ▼
 question → source passage → timestamp
```

The public interface runs `Xenova/all-MiniLM-L6-v2` with Transformers.js. Search remains usable with lexical scoring if model loading fails.

## Evaluation

LectureLens includes a reproducible retrieval benchmark under `evaluation/` designed to test whether a question retrieves the correct lecture passage, including hard negatives that discuss related concepts without actually answering the question.

The current 32-pair regression benchmark uses a development split for threshold selection and a held-out test split for final classification metrics.

| Method | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| MiniLM semantic | 0.714 | 1.000 | 0.833 | 0.833 |
| **Hybrid semantic + lexical** | **0.833** | **1.000** | **0.909** | **0.917** |

Median semantic scoring latency on the GitHub Actions CPU runner was **6.647 ms per query/passage pair**.

These results describe the current regression benchmark only; they are not claims about population-level lecture search quality. Ranking metrics are intentionally not promoted yet because the current benchmark grouping is too sparse for a meaningful Recall@K claim.

## Engineering decisions

- **Source evidence over generated certainty.** Results point to transcript text and timestamps.
- **Local-first architecture.** The first release does not require a LectureLens backend for indexing or search.
- **Separable pipeline.** Capture, transcription, segmentation, retrieval, and study views are independent layers.
- **Graceful degradation.** Lexical search remains available when the embedding model cannot load.
- **Measured retrieval.** Evaluation is separate from UI validation and deployment.
- **Privacy-aware defaults.** Browser recording and transcript persistence avoid unnecessary centralized storage.

## Roadmap

1. transcript chunking tuned for lecture structure
2. searchable lecture chapters
3. Word Error Rate evaluation using labeled lecture audio
4. IndexedDB lecture library
5. exportable study packs
6. Web Worker / WebGPU performance work

## License

MIT
