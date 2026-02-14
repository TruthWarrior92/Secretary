# Secretary

Python GUI for **transcription + speaker diarization**: upload an audio file, run [Whisper](https://github.com/openai/whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio) to get a transcript with “who said what,” then edit speaker labels, correct or merge speakers, and export.

## Features

- **Upload** WAV, MP3, FLAC, M4A (or OGG)
- **Transcribe & diarize** in one step (Whisper + pyannote)
- **Edit** per-segment speaker (dropdown) and text
- **Label speakers** (e.g. `SPEAKER_00` → “Alice”)
- **Merge speakers** (merge one ID into another)
- **Export** TXT, SRT, VTT, JSON (project save)
- **Load project** from JSON to continue editing
- **Open at time** (open default player for the file; seek manually)

## Requirements

- **Python 3.10+**
- **Hugging Face token** (free): required for pyannote speaker-diarization. You must accept the model terms on the model page.

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

   If you don’t have PyTorch yet, install it (e.g. with CUDA if you have a GPU):

   ```bash
   pip install torch torchaudio
   ```

3. **Hugging Face token**

   - Create a token at [https://hf.co/settings/tokens](https://hf.co/settings/tokens)
   - Accept the user conditions for the diarization model: [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - Either:
     - Copy `.env.example` to `.env` and set `HUGGINGFACE_TOKEN=your_token`, or
     - Create `~/.secretary/config` (or `%USERPROFILE%\.secretary\config`) with a line: `HUGGINGFACE_TOKEN=your_token`

## Usage

From the project root:

```bash
python run.py
```

1. **Select audio** — choose a WAV/MP3/FLAC/M4A file.
2. Optionally set **Whisper model** (tiny → large) and **Language** (blank = auto).
3. Click **Transcribe & diarize** — wait for Whisper and diarization to finish.
4. Edit the **segment table**: change speaker per row (dropdown) or edit text.
5. **Label speaker**: pick a speaker ID, enter a display name, click “Apply label” (used in exports).
6. **Merge speakers**: choose “Merge A → B” to turn all A into B.
7. **Export** TXT, SRT, VTT, or JSON. Use **Load project…** to open a saved JSON and continue editing.

## Export formats

- **TXT** — One line per segment: `start end SPEAKER_00 text` (compatible with [pyannote-whisper](https://github.com/yinruiqing/pyannote-whisper) style).
- **SRT / VTT** — Subtitles with speaker in the text.
- **JSON** — Full project (segments + speaker labels + audio path) for later load.

## Optional / future

See [FUTURE_IDEAS.md](FUTURE_IDEAS.md) for planned features (playback sync to timestamp, search, re-run diarization only, batch, theme, etc.).

## License

Use and modify as you like. Whisper and pyannote have their own licenses.
