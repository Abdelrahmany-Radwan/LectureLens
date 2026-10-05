# Changelog

## 1.0.0 — 2026-10-05

### Product
- in-browser microphone recording with MediaRecorder
- audio import, playback, and waveform feedback
- local Whisper Tiny English speech-to-text
- timestamp-aware transcript parsing and source-linked playback
- semantic + lexical lecture search
- persistent multi-lecture IndexedDB library
- saved transcript and optional audio Blob reopening across reloads
- lightweight lecture chapter grouping and study cards
- bright student-first responsive interface

### ML / evaluation
- MiniLM semantic embeddings with lexical fallback
- grouped retrieval benchmark with held-out query groups
- real-audio LibriSpeech WER benchmark
- classroom-domain CoTACS WER benchmark: **30.3% WER** on ATA1 first 120 seconds
- latency and real-time-factor reporting
- explicit scope notes to avoid population-level accuracy claims

### Engineering
- Whisper and MiniLM inference moved into a module Web Worker
- transferable 16 kHz audio buffers for background transcription
- Playwright end-to-end browser tests
- CI validation, retrieval evaluation, ASR evaluation, and GitHub Pages deployment
- Dependabot, issue templates, PR template, and protected main branch
- architecture, security, contribution, and evaluation documentation

## 0.1.0

- local microphone recording
- audio import and playback
- waveform feedback
- timestamp-aware transcript parsing
- MiniLM semantic search
- lexical fallback
- clickable source timestamps
- lecture outline and study cards
- retrieval benchmark
- CI, evaluation, and Pages workflows
