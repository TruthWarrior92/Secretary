# Secretary

Python GUI for **transcription + speaker diarization**: upload an audio file, run [Whisper](https://github.com/openai/whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio) to get a transcript with "who said what," then edit speaker labels, correct or merge speakers, and export.

## Features

### Transcription & Diarization
- **Upload** WAV, MP3, FLAC, M4A, OGG
- **Transcribe & diarize** in one step (Whisper + pyannote) with word-level timestamps
- **Fast mode** — Whisper full-file then diarize and assign speakers (default)
- **Accurate mode** — Diarize first, then run Whisper per speaker segment for cleaner per-turn text
- **Re-diarize** — Re-run only diarization without re-transcribing (keeps Whisper results)
- **VAD pre-filter** — Optional Voice Activity Detection to trim silence before Whisper (faster, fewer hallucinations)
- **Min segment length** — Automatically merge segments shorter than a configurable threshold

### Editing
- **Edit** per-segment speaker (dropdown) and text inline
- **Label speakers** (e.g. `SPEAKER_00` → "Alice")
- **Merge speakers** (merge one ID into another)
- **Split segments** — Split a segment at any time point (scissors button per row)
- **Undo / Redo** — 50-deep state stack with Ctrl+Z / Ctrl+Y

### Playback
- **In-app playback** — Click ▶ on any row to hear that segment instantly (sounddevice + librosa fallback for M4A)
- **Open in player** — Open audio file in system default player

### Visual
- **Speaker colors** — Color-coded strip on each row (12-color palette, updates live on speaker change)
- **Search** — Find text in transcript with highlight, scroll-to-match, and Next/Prev cycling
- **Statistics panel** — Per-speaker duration, segment count, and percentage breakdown
- **Dark / Light theme** — Toggle persisted across sessions

### Export
- **TXT** — `0.00 - 2.50 [Alice] text` (one line per segment)
- **SRT / VTT** — Subtitles with speaker in the text
- **JSON** — Full project (segments + speaker labels + audio path) for save/load

### Advanced
- **Batch processing** — Queue multiple audio files, process sequentially with per-file progress bars
- **Summarization** — Generate meeting summaries via local Ollama LLM (configurable model and endpoint)
- **Logging** — Pipeline steps and errors logged to `~/.secretary/logs/` with 7-day rotation
- **First-run setup** — Guided HF token prompt on first launch with links to token creation and model conditions

## Requirements

- **Python 3.10+**
- **Hugging Face token** (free): required for pyannote speaker-diarization. You must accept the model terms on the model page.
- **Ollama** (optional): only needed for transcript summarization. Install from [ollama.com](https://ollama.com) and pull a model (e.g. `ollama pull llama3`).

## Installation

1. Clone the repo and create a virtual environment:

   ```bash
   cd Secretary
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have PyTorch yet, install it (e.g. with CUDA if you have a GPU):

   ```bash
   pip install torch torchaudio
   ```

3. **Hugging Face token**

   - Create a **read** token at [https://hf.co/settings/tokens](https://hf.co/settings/tokens)
   - Accept the user conditions for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - Put the token in one of these (first found wins):
     - **Recommended:** Copy `config.json.example` to `config.json` and set `"HUGGINGFACE_TOKEN": "hf_xxxx"`
     - Or use `.env` with `HUGGINGFACE_TOKEN=your_token`
     - Or create `~/.secretary/config.json` with `{"HUGGINGFACE_TOKEN": "hf_xxxx"}`
   - On first launch, the app will prompt for the token if not found.

## Usage

```bash
python run.py
```

1. **Select audio** — choose a WAV/MP3/FLAC/M4A file.
2. Set **Whisper model** (tiny → large), **Language** (blank = auto), **Mode** (Fast/Accurate), and optionally enable **VAD pre-filter** or set **Min seg** threshold.
3. Click **Transcribe & diarize** — wait for the pipeline to finish.
4. Edit the **segment table**: change speaker per row, edit text, split segments, or play audio.
5. **Label speaker** / **Merge speakers** to clean up diarization results.
6. **Search** the transcript, view **Stats**, or **Summarize** via Ollama.
7. **Export** TXT, SRT, VTT, or JSON. Use **Load project** to reopen a saved JSON.
8. Use **Batch** to queue and process multiple files at once.

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

18 unit tests covering models, diarize utilities, and all export formats.

## License

Use and modify as you like. Whisper and pyannote have their own licenses.
