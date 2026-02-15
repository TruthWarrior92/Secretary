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

   - Create a **read** token (or fine-grained with read access) at [https://hf.co/settings/tokens](https://hf.co/settings/tokens)
   - Accept the user conditions for the diarization pipeline: [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (log in, then accept to share contact info; no extra sub-models to accept)
   - Put the token in one of these (first found wins):
     - **Recommended:** Copy `config.json.example` to `config.json` in the project folder and set `"HUGGINGFACE_TOKEN": "hf_xxxx"`. This avoids terminal environment variable issues.
     - Or use `.env` with `HUGGINGFACE_TOKEN=your_token`
     - Or create `~/.secretary/config.json` (or `%USERPROFILE%\.secretary\config.json`) with `{"HUGGINGFACE_TOKEN": "hf_xxxx"}`
   - The app uses **speaker-diarization-community-1**: expects mono 16 kHz audio (we resample and pass a waveform dict). Pipelines run on CPU by default; pyannote can use GPU if you call `pipeline.to(torch.device("cuda"))` (future enhancement).

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
