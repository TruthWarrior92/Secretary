# Secretary — Future Ideas & Backlog

Track ideas and features to consider after the initial release. Add new items at the bottom under "Other ideas."

---

## From plan (§6 – other functions)

- [x] **Playback sync** — Click segment row ▶ to play that segment in-app; "Open in player" still opens file in system player.
- [x] **Search in transcript** — Text search across segments; highlight and scroll to match.
- [x] **Re-run only diarization** — Keep Whisper result; re-run diarization and re-assign speakers without re-transcribing.
- [x] **Import/export JSON project** — Save and load project (audio path + segments + speaker labels) to continue editing later.
- [x] **Batch** — Queue multiple files; run pipeline on each; list results and open any in the editor.
- [x] **Optional summarization** — Local Ollama REST API summarization with configurable model/endpoint.
- [x] **Speaker colors** — Assign a color per speaker in the table for quick visual scanning.
- [x] **Statistics** — Total time per speaker, segment count per speaker (simple stats panel).

---

## From plan (§9 – additional ideas)

- [x] **Split segment** — Split one segment into two (set time boundary; optionally re-assign speaker).
- [x] **Transcription mode** — Offer "Fast" (Whisper then diarize) vs "Accurate" (diarize then Whisper per segment).
- [x] **Word-level timestamps** — Whisper word-level output enabled for finer segment boundaries.
- [x] **First-run setup** — On first launch, prompt for HF token and model terms; save to config; link to HF token and model pages.
- [x] **Theme** — Light/dark toggle (persist in config).
- [x] **Minimum segment length** — Option to merge very short segments (configurable threshold).
- [x] **Logging** — Pipeline steps and errors to `~/.secretary/logs/` with 7-day rotation.
- [x] **Tests** — Unit tests for models, diarize_utils, and export (18 tests).
- [x] **Optional VAD** — Pre-filter silence so Whisper focuses on speech-only regions (toggle in UI).

---

## UX / polish

- [x] **Undo/redo** — For speaker and text edits (Ctrl+Z / Ctrl+Y, 50-deep stack).
- [x] **Diarization all SPEAKER_00** — Fixed: extract Annotation from DiarizeOutput wrapper; handle argmax() None; hint min_speakers/max_speakers.

---

## Other ideas

_(Add new items here as they come up.)_
