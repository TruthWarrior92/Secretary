# Secretary — Future Ideas & Backlog

Track ideas and features to consider after the initial release. Add new items at the bottom under "Other ideas."

---

## From plan (§6 – other functions)

- [ ] **Playback sync** — Click segment row to jump to that time (embedded or system player).
- [ ] **Search in transcript** — Text search across segments; highlight and scroll to match.
- [ ] **Re-run only diarization** — Keep Whisper result; re-run diarization and re-assign speakers without re-transcribing.
- [ ] **Import/export JSON project** — Save and load project (audio path + segments + speaker labels) to continue editing later.
- [ ] **Batch** — Queue multiple files; run pipeline on each; list results and open any in the editor.
- [ ] **Optional summarization** — Send transcript (or per-speaker) to local or API summarizer; keep scope small.
- [ ] **Speaker colors** — Assign a color per speaker in the table for quick visual scanning.
- [ ] **Statistics** — Total time per speaker, segment count per speaker (simple stats panel).

---

## From plan (§9 – additional ideas)

- [ ] **Split segment** — Split one segment into two (set time boundary; optionally re-assign speaker).
- [ ] **Transcription mode** — Offer "Fast" (Whisper then diarize) vs "Accurate" (diarize then Whisper per segment).
- [ ] **Word-level timestamps** — Optional Whisper word-level output for finer SRT or "click word to jump."
- [ ] **First-run setup** — On first launch, prompt for HF token and model terms; save to config; link to HF token and model pages.
- [ ] **Theme** — Light/dark toggle (persist in config).
- [ ] **Minimum segment length** — Option to merge or hide very short segments (e.g. &lt; 0.5 s).
- [ ] **Logging** — Pipeline steps and errors to e.g. `~/.secretary/logs/` for debugging.
- [ ] **Tests** — Unit tests for `diarize_utils` and `export`; optional integration test with short sample WAV.
- [ ] **Optional VAD** — Pre-filter silence so diarization/Whisper focus on speech-only regions (advanced option).

---

## UX / polish

- [ ] **Undo/redo** — For speaker and text edits (v1 assumes no undo).

---

## Other ideas

_(Add new items here as they come up.)_
