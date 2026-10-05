# Architecture

LectureLens is structured as a set of separable stages so each layer can be tested and replaced independently.

## 1. Capture

The browser uses `MediaRecorder` for microphone capture and supports imported audio files. Playback is handled locally through the browser audio element.

The current milestone records audio but does not yet run speech recognition on the recording. Transcription is treated as its own subsystem rather than being coupled to retrieval.

## 2. Transcript representation

A transcript is represented as timestamped chunks with `time`, `seconds`, and `text` fields. If pasted text has no timestamps, LectureLens falls back to sentence segmentation with synthetic offsets so the retrieval UI remains testable.

## 3. Semantic representation

The browser loads `Xenova/all-MiniLM-L6-v2` through Transformers.js. Each transcript chunk and user question is converted into a normalized sentence embedding.

## 4. Hybrid retrieval

For each transcript chunk:

- semantic score = cosine similarity between question and chunk embeddings
- lexical score = normalized token overlap
- hybrid score = `0.86 * semantic + 0.14 * lexical`

The weighting is an initial product configuration. The evaluation harness is the source of truth for future tuning.

## 5. Evidence interaction

Results expose transcript passage, relevance score, and source timestamp. Selecting a result seeks the audio player and highlights the corresponding timeline segment.

## 6. Failure behavior

If the embedding model cannot load, search falls back to lexical ranking rather than blocking the workflow.

## 7. Privacy model

The first release does not require a LectureLens backend for recording, transcript indexing, or semantic retrieval. Audio remains attached to a local browser object URL for the session, the most recent transcript may be stored in localStorage, and embeddings remain in memory.

## 8. Planned speech layer

The next milestone will add browser/local speech recognition. Evaluation will use Word Error Rate (WER) against manually transcribed lecture samples.


## v1 browser architecture

LectureLens now separates interface work from ML inference.

```text
main UI thread
  ├─ MediaRecorder / Web Audio
  ├─ IndexedDB lecture library
  ├─ transcript / timeline rendering
  ├─ lexical fallback
  └─ message bridge
          │
          ▼
      ml-worker.js
        ├─ Whisper Tiny English
        └─ MiniLM feature extraction
```

### Persistent lecture records

Each saved lecture is stored in IndexedDB as a record containing:

- stable lecture ID
- title and course label
- timestamped transcript
- optional audio Blob
- update timestamp

The UI reconstructs object URLs only when a lecture is opened, avoiding attempts to persist temporary Blob URLs.

### Responsiveness

Whisper transcription and MiniLM embedding calls run inside a module Web Worker. Audio preprocessing remains in browser audio APIs, then the 16 kHz mono Float32 buffer is transferred to the worker rather than copied.

### Browser product testing

Playwright tests exercise the product at the browser level: sample loading, evidence retrieval, IndexedDB persistence across reloads, lecture reopening, and workspace reset behavior. A deterministic query flag disables external model loading only inside end-to-end tests; production behavior still uses the ML worker.
