"""Secretary GUI: upload audio, transcribe + diarize, edit segments, export."""
import copy
import os
import sys
import threading
import traceback
import warnings
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import customtkinter as ctk

# Add project root for config
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import DEFAULT_WHISPER_MODEL, get_hf_token, get_config, save_config
from secretary.diarize_utils import diarize_text, merge_short_segments
from secretary.export import to_json, to_srt, to_txt, to_vtt, load_json
from secretary.models import (
    SPEAKER_COLORS,
    Segment,
    get_display_speaker,
    snapshot_segments,
    split_segment,
)
from secretary.logging_setup import setup_logging, get_log_dir
from secretary.pipeline import (
    load_diarization_pipeline,
    load_whisper,
    log_pipeline_error,
    run_diarization_only,
    run_pipeline,
    run_pipeline_accurate,
)

logger = setup_logging()


# Supported audio extensions
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def norm_path_for_windows(path: str | Path) -> str:
    """Normalize path so Windows (UNC, startfile, isfile) works."""
    p = Path(path) if not isinstance(path, Path) else path
    s = str(p)
    if os.name == "nt":
        if s.startswith("//") and not s.startswith("\\\\"):
            s = "\\\\" + s[2:].replace("/", "\\")
        else:
            s = os.path.normpath(s)
    return s


def get_audio_duration(path: str) -> float:
    """Return duration in seconds; 0 if unreadable."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                import librosa
                y, sr = librosa.load(path, sr=None, duration=0)
            return len(y) / sr if sr else 0
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# First-run setup dialog
# ---------------------------------------------------------------------------

class _FirstRunDialog(ctk.CTkToplevel):
    """Prompt for HF token on first launch."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Secretary — First-run Setup")
        self.geometry("520x260")
        self.resizable(False, False)
        self.grab_set()
        self.token_value: str | None = None

        ctk.CTkLabel(self, text="Hugging Face token is required for speaker diarization.",
                      wraplength=480, anchor="w").pack(padx=20, pady=(20, 5), anchor="w")
        ctk.CTkLabel(self, text="1. Create a read token at https://hf.co/settings/tokens",
                      wraplength=480, anchor="w").pack(padx=30, anchor="w")
        ctk.CTkLabel(self, text="2. Accept conditions at https://huggingface.co/pyannote/speaker-diarization-community-1",
                      wraplength=480, anchor="w").pack(padx=30, anchor="w")
        ctk.CTkLabel(self, text="3. Paste your token below and click Save.",
                      wraplength=480, anchor="w").pack(padx=30, pady=(0, 10), anchor="w")

        self._entry = ctk.CTkEntry(self, width=460, placeholder_text="hf_...")
        self._entry.pack(padx=20, pady=5)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=15)
        ctk.CTkButton(btn_frame, text="Save", width=100, command=self._save).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Skip", width=100, fg_color="gray", command=self.destroy).pack(side="left", padx=5)

    def _save(self):
        token = self._entry.get().strip()
        if token:
            save_config({"HUGGINGFACE_TOKEN": token})
            self.token_value = token
        self.destroy()


# ---------------------------------------------------------------------------
# Statistics dialog
# ---------------------------------------------------------------------------

class _StatsDialog(ctk.CTkToplevel):
    def __init__(self, parent, segments: list[Segment]):
        super().__init__(parent)
        self.title("Transcript Statistics")
        self.geometry("400x320")
        self.resizable(False, False)
        self.grab_set()

        speakers: dict[str, dict] = {}
        total_dur = 0.0
        for seg in segments:
            d = seg.duration()
            total_dur += d
            if seg.speaker_id not in speakers:
                speakers[seg.speaker_id] = {"duration": 0.0, "count": 0}
            speakers[seg.speaker_id]["duration"] += d
            speakers[seg.speaker_id]["count"] += 1

        ctk.CTkLabel(self, text=f"Total duration: {total_dur:.2f}s  |  Segments: {len(segments)}",
                      anchor="w").pack(padx=15, pady=(15, 10), anchor="w")

        frame = ctk.CTkScrollableFrame(self, width=360, height=220)
        frame.pack(padx=15, pady=(0, 15), fill="both", expand=True)
        for spk in sorted(speakers):
            info = speakers[spk]
            pct = (info["duration"] / total_dur * 100) if total_dur else 0
            ctk.CTkLabel(frame, text=f"{spk}:  {info['duration']:.2f}s  ({pct:.1f}%)  —  {info['count']} segments",
                          anchor="w").pack(anchor="w", padx=5, pady=2)


# ---------------------------------------------------------------------------
# Batch processing window
# ---------------------------------------------------------------------------

class _BatchWindow(ctk.CTkToplevel):
    """Queue multiple audio files and process them sequentially."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Batch Processing")
        self.geometry("620x440")
        self.grab_set()
        self._parent = parent
        self._jobs: list = []

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(top, text="Add files\u2026", width=100, command=self._add_files).pack(side="left")
        ctk.CTkButton(top, text="Run all", width=80, command=self._run_all).pack(side="left", padx=10)
        self._status_label = ctk.CTkLabel(top, text="", anchor="w")
        self._status_label.pack(side="left", fill="x", expand=True)

        # Overall progress
        self._overall_bar = ctk.CTkProgressBar(self, width=580)
        self._overall_bar.set(0)
        self._overall_bar.pack(padx=10, pady=(0, 5))

        self._list_frame = ctk.CTkScrollableFrame(self, width=580, height=300)
        self._list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._rows: list[dict] = []  # {"label": CTkLabel, "bar": CTkProgressBar}

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Select audio files",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All", "*.*")])
        if not paths:
            return
        from secretary.batch import BatchJob
        for p in paths:
            job = BatchJob(audio_path=norm_path_for_windows(p))
            self._jobs.append(job)
            rf = ctk.CTkFrame(self._list_frame, fg_color="transparent")
            rf.pack(fill="x", padx=5, pady=2)
            lbl = ctk.CTkLabel(rf, text=f"{Path(p).name}  \u2014  queued", anchor="w")
            lbl.pack(fill="x")
            bar = ctk.CTkProgressBar(rf, width=540, height=10)
            bar.set(0)
            bar.pack(fill="x", pady=(0, 2))
            self._rows.append({"label": lbl, "bar": bar})

    def _run_all(self):
        if not self._jobs:
            return
        from secretary.batch import run_batch

        model = self._parent._model_combo.get().strip() or DEFAULT_WHISPER_MODEL
        lang = self._parent._lang_entry.get().strip() or None
        total = len(self._jobs)

        def _progress(idx: int, msg: str):
            self.after(0, lambda: self._update_row(idx, msg, total))

        def _bg():
            run_batch(self._jobs, model_name=model, language=lang, progress_callback=_progress)
            self.after(0, lambda: self._status_label.configure(text="Done"))
            self.after(0, lambda: self._overall_bar.set(1.0))

        self._status_label.configure(text="Running\u2026")
        threading.Thread(daemon=True, target=_bg).start()

    def _update_row(self, idx: int, msg: str, total: int):
        if idx < len(self._rows):
            job = self._jobs[idx]
            row = self._rows[idx]
            row["label"].configure(text=f"{Path(job.audio_path).name}  \u2014  {msg}")
            if job.status == "done":
                row["bar"].set(1.0)
            elif job.status == "running":
                row["bar"].set(0.5)
            elif job.status == "failed":
                row["bar"].set(1.0)
                row["bar"].configure(progress_color="red")
        # Overall progress
        done = sum(1 for j in self._jobs if j.status in ("done", "failed"))
        self._overall_bar.set(done / max(total, 1))


# ---------------------------------------------------------------------------
# Summarize dialog
# ---------------------------------------------------------------------------

class _SummarizeDialog(ctk.CTkToplevel):
    def __init__(self, parent, segments, label_map):
        super().__init__(parent)
        self.title("Summarize Transcript")
        self.geometry("560x400")
        self.grab_set()
        self._segments = segments
        self._label_map = label_map

        cfg = get_config()
        opt = ctk.CTkFrame(self, fg_color="transparent")
        opt.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(opt, text="Ollama model:").pack(side="left")
        self._model_entry = ctk.CTkEntry(opt, width=120, placeholder_text="llama3")
        self._model_entry.insert(0, cfg.get("OLLAMA_MODEL", "llama3"))
        self._model_entry.pack(side="left", padx=5)
        ctk.CTkLabel(opt, text="Endpoint:").pack(side="left")
        self._ep_entry = ctk.CTkEntry(opt, width=200, placeholder_text="http://localhost:11434")
        self._ep_entry.insert(0, cfg.get("OLLAMA_ENDPOINT", "http://localhost:11434"))
        self._ep_entry.pack(side="left", padx=5)
        ctk.CTkButton(opt, text="Summarize", width=90, command=self._run).pack(side="left", padx=5)

        self._text = ctk.CTkTextbox(self, width=540, height=280)
        self._text.pack(padx=10, pady=(0, 10), fill="both", expand=True)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(padx=10, pady=(0, 10))
        ctk.CTkButton(btn_row, text="Copy", width=80, command=self._copy).pack(side="left")

    def _run(self):
        from secretary.summarize import summarize_transcript
        model = self._model_entry.get().strip() or "llama3"
        endpoint = self._ep_entry.get().strip() or "http://localhost:11434"
        save_config({"OLLAMA_MODEL": model, "OLLAMA_ENDPOINT": endpoint})
        self._text.delete("1.0", "end")
        self._text.insert("1.0", "Generating summary…")

        def _bg():
            try:
                result = summarize_transcript(self._segments, self._label_map, model=model, endpoint=endpoint)
                self.after(0, lambda: self._show(result))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._show(f"Error: {err}"))

        threading.Thread(daemon=True, target=_bg).start()

    def _show(self, text: str):
        self._text.delete("1.0", "end")
        self._text.insert("1.0", text)

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._text.get("1.0", "end"))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class SecretaryApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Secretary — Transcription & Diarization")
        self.geometry("1060x700")
        self.minsize(800, 500)

        self.audio_path: str | None = None
        self.segments: list[Segment] = []
        self.speaker_label_map: dict[str, str] = {}
        self._last_transcribe_result: dict | None = None
        self._progress_var = ctk.StringVar(value="")
        self._run_button: ctk.CTkButton | None = None
        self._rediar_button: ctk.CTkButton | None = None
        self._table_frame: ctk.CTkScrollableFrame | None = None
        self._table_widgets: list[dict] = []
        self._speaker_combo: ctk.CTkComboBox | None = None
        self._merge_from_combo: ctk.CTkComboBox | None = None
        self._merge_to_combo: ctk.CTkComboBox | None = None

        # Search state
        self._search_matches: list[int] = []
        self._search_idx: int = -1

        # Undo / redo
        self._undo_stack: list[list[Segment]] = []
        self._redo_stack: list[list[Segment]] = []

        self._build_ui()

        # Keyboard shortcuts
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("<Control-y>", lambda e: self._redo())

        # First-run: check token
        self.after(200, self._check_first_run)

    # ── First-run ──────────────────────────────────────────────────────

    def _check_first_run(self):
        if not get_hf_token():
            dlg = _FirstRunDialog(self)
            self.wait_window(dlg)

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        # Top: file and run
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=10)

        self._file_label = ctk.CTkLabel(top, text="No file selected", anchor="w")
        self._file_label.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(top, text="Select audio…", width=120, command=self._on_select_file).pack(side="right", padx=(5, 0))
        self._run_button = ctk.CTkButton(top, text="Transcribe & diarize", width=160, command=self._on_run_pipeline, state="disabled")
        self._run_button.pack(side="right")
        self._rediar_button = ctk.CTkButton(top, text="Re-diarize", width=100, command=self._on_rediarize, state="disabled")
        self._rediar_button.pack(side="right", padx=(0, 5))

        # Progress
        self._progress = ctk.CTkLabel(self, textvariable=self._progress_var, anchor="w")
        self._progress.pack(fill="x", padx=10, pady=(0, 5))

        # Options row: model, language, min segment, theme
        opt = ctk.CTkFrame(self, fg_color="transparent")
        opt.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(opt, text="Whisper model:").pack(side="left", padx=(0, 5))
        self._model_combo = ctk.CTkComboBox(
            opt, values=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "large"], width=120)
        self._model_combo.set(DEFAULT_WHISPER_MODEL)
        self._model_combo.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(opt, text="Language (blank=auto):").pack(side="left", padx=(0, 5))
        self._lang_entry = ctk.CTkEntry(opt, width=80, placeholder_text="auto")
        self._lang_entry.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(opt, text="Min seg (s):").pack(side="left", padx=(0, 3))
        self._min_seg_entry = ctk.CTkEntry(opt, width=50, placeholder_text="0.0")
        self._min_seg_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(opt, text="Mode:").pack(side="left", padx=(0, 3))
        self._mode_combo = ctk.CTkComboBox(opt, values=["Fast", "Accurate"], width=100)
        self._mode_combo.set("Fast")
        self._mode_combo.pack(side="left", padx=(0, 10))

        self._vad_var = ctk.StringVar(value="off")
        self._vad_check = ctk.CTkSwitch(opt, text="VAD pre-filter", variable=self._vad_var, onvalue="on", offvalue="off")
        self._vad_check.pack(side="left", padx=(0, 10))

        # Theme toggle
        self._theme_var = ctk.StringVar(value="on" if get_config().get("theme", "dark") == "dark" else "off")
        self._theme_switch = ctk.CTkSwitch(opt, text="Dark mode", variable=self._theme_var, onvalue="on", offvalue="off", command=self._on_theme_toggle)
        self._theme_switch.pack(side="right")

        # Search row
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkLabel(search_frame, text="Search:").pack(side="left", padx=(0, 5))
        self._search_entry = ctk.CTkEntry(search_frame, width=200, placeholder_text="Find in transcript…")
        self._search_entry.pack(side="left", padx=(0, 5))
        self._search_entry.bind("<Return>", lambda e: self._on_search())
        ctk.CTkButton(search_frame, text="Find", width=60, command=self._on_search).pack(side="left", padx=(0, 3))
        ctk.CTkButton(search_frame, text="Next", width=60, command=self._on_search_next).pack(side="left", padx=(0, 3))
        ctk.CTkButton(search_frame, text="Prev", width=60, command=self._on_search_prev).pack(side="left")
        self._search_info = ctk.CTkLabel(search_frame, text="", width=120)
        self._search_info.pack(side="left", padx=10)

        # Segment table area
        table_container = ctk.CTkFrame(self, fg_color="transparent")
        table_container.pack(fill="both", expand=True, padx=10, pady=5)

        header = ctk.CTkFrame(table_container, fg_color="transparent")
        header.pack(fill="x")
        for text, w in [("", 8), ("Start", 70), ("End", 70), ("", 36), ("", 30), ("Speaker", 120)]:
            ctk.CTkLabel(header, text=text, width=w).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Text", anchor="w").pack(side="left", fill="x", expand=True, padx=2)

        self._table_frame = ctk.CTkScrollableFrame(table_container, fg_color="transparent")
        self._table_frame.pack(fill="both", expand=True)

        # Speaker edit area
        edit_frame = ctk.CTkFrame(self, fg_color="transparent")
        edit_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(edit_frame, text="Label speaker:").pack(side="left", padx=(0, 5))
        self._label_speaker_combo = ctk.CTkComboBox(edit_frame, values=[""], width=150, state="disabled")
        self._label_speaker_combo.pack(side="left", padx=(0, 5))
        self._label_name_entry = ctk.CTkEntry(edit_frame, width=120, placeholder_text="Display name")
        self._label_name_entry.pack(side="left", padx=(0, 5))
        ctk.CTkButton(edit_frame, text="Apply label", width=90, command=self._on_apply_label).pack(side="left", padx=(0, 15))
        ctk.CTkLabel(edit_frame, text="Merge:").pack(side="left", padx=(0, 5))
        self._merge_from_combo = ctk.CTkComboBox(edit_frame, values=[""], width=150, state="disabled")
        self._merge_from_combo.pack(side="left", padx=(0, 5))
        ctk.CTkLabel(edit_frame, text="\u2192").pack(side="left", padx=2)
        self._merge_to_combo = ctk.CTkComboBox(edit_frame, values=[""], width=150, state="disabled")
        self._merge_to_combo.pack(side="left", padx=(0, 5))
        ctk.CTkButton(edit_frame, text="Merge speakers", width=110, command=self._on_merge_speakers).pack(side="left")

        # Export, playback, undo, stats
        export_frame = ctk.CTkFrame(self, fg_color="transparent")
        export_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(export_frame, text="Export TXT", width=80, command=lambda: self._export("txt")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(export_frame, text="Export SRT", width=80, command=lambda: self._export("srt")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(export_frame, text="Export VTT", width=80, command=lambda: self._export("vtt")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(export_frame, text="Export JSON", width=80, command=lambda: self._export("json")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(export_frame, text="Open in player", width=110, command=self._on_open_at_time).pack(side="left", padx=(10, 4))
        ctk.CTkButton(export_frame, text="Load project\u2026", width=100, command=self._on_load_project).pack(side="left", padx=(0, 4))
        ctk.CTkButton(export_frame, text="Stats", width=60, command=self._on_stats).pack(side="left", padx=(10, 4))
        ctk.CTkButton(export_frame, text="Summarize", width=80, command=self._on_summarize).pack(side="left", padx=(4, 4))
        ctk.CTkButton(export_frame, text="Batch", width=60, command=self._on_batch).pack(side="left", padx=(4, 4))
        ctk.CTkButton(export_frame, text="Logs", width=50, command=self._on_open_logs).pack(side="left", padx=(4, 0))
        ctk.CTkButton(export_frame, text="Undo", width=60, command=self._undo).pack(side="right", padx=(4, 0))
        ctk.CTkButton(export_frame, text="Redo", width=60, command=self._redo).pack(side="right", padx=(4, 0))

    # ── File selection ─────────────────────────────────────────────────

    def _on_select_file(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("WAV", "*.wav"), ("MP3", "*.mp3"), ("All", "*.*")],
        )
        if path:
            self.audio_path = norm_path_for_windows(path)
            name = Path(self.audio_path).name
            dur = get_audio_duration(path)
            dur_s = f"{dur:.1f}s" if dur > 0 else "?"
            self._file_label.configure(text=f"{name} ({dur_s})")
            self._run_button.configure(state="normal")

    # ── Pipeline ───────────────────────────────────────────────────────

    def _on_run_pipeline(self):
        if not self.audio_path or not os.path.isfile(self.audio_path):
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        if not get_hf_token():
            messagebox.showerror("Missing token", "Set HUGGINGFACE_TOKEN in config.json, .env, or ~/.secretary/config.json.")
            return
        self._run_button.configure(state="disabled")
        self._rediar_button.configure(state="disabled")
        self._progress_var.set("Running\u2026")

        min_seg = 0.0
        try:
            min_seg = float(self._min_seg_entry.get())
        except (ValueError, TypeError):
            pass

        mode = self._mode_combo.get()
        use_vad = self._vad_var.get() == "on"

        def run():
            try:
                model = self._model_combo.get().strip()
                lang = self._lang_entry.get().strip() or None
                cb = lambda msg: self.after(0, lambda: self._progress_var.set(msg))
                logger.info("Pipeline start: mode=%s model=%s lang=%s vad=%s", mode, model, lang, use_vad)
                if mode == "Accurate":
                    segs = run_pipeline_accurate(
                        self.audio_path, model_name=model or DEFAULT_WHISPER_MODEL,
                        language=lang, use_vad=use_vad, progress_callback=cb)
                else:
                    segs = run_pipeline(
                        self.audio_path, model_name=model or DEFAULT_WHISPER_MODEL,
                        language=lang, use_vad=use_vad, progress_callback=cb)
                if min_seg > 0:
                    segs = merge_short_segments(segs, min_seg)
                logger.info("Pipeline done: %d segments", len(segs))
                self.after(0, lambda: self._on_pipeline_done(segs, None))
            except Exception as e:
                err_msg = str(e)
                logger.error("Pipeline failed: %s", err_msg)
                log_pipeline_error(err_msg, traceback.format_exc())
                self.after(0, lambda: self._on_pipeline_done([], err_msg))

        threading.Thread(daemon=True, target=run).start()

    def _on_pipeline_done(self, segments: list[Segment], error: str | None):
        self._run_button.configure(state="normal")
        if error:
            self._progress_var.set("")
            msg = error
            if any(k in error.lower() for k in ("token", "auth", "401", "403", "gated", "accept")):
                msg += "\n\nCheck config.json or .env for HUGGINGFACE_TOKEN."
            messagebox.showerror("Pipeline failed", msg)
            return
        self._push_undo()
        self.segments = segments
        self._progress_var.set("")
        self._refresh_table()
        self._refresh_speaker_combos()
        self._rediar_button.configure(state="normal" if self.audio_path else "disabled")

    # ── Re-diarize ─────────────────────────────────────────────────────

    def _on_rediarize(self):
        if not self.audio_path or not self.segments:
            messagebox.showinfo("Re-diarize", "Run the full pipeline first.")
            return
        self._run_button.configure(state="disabled")
        self._rediar_button.configure(state="disabled")
        self._progress_var.set("Re-diarizing\u2026")
        # Build a minimal transcribe_result from current segments
        transcribe_result = {
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in self.segments
            ]
        }

        min_seg = 0.0
        try:
            min_seg = float(self._min_seg_entry.get())
        except (ValueError, TypeError):
            pass

        def run():
            try:
                segs = run_diarization_only(
                    self.audio_path,
                    transcribe_result,
                    progress_callback=lambda msg: self.after(0, lambda: self._progress_var.set(msg)),
                )
                if min_seg > 0:
                    segs = merge_short_segments(segs, min_seg)
                self.after(0, lambda: self._on_pipeline_done(segs, None))
            except Exception as e:
                err_msg = str(e)
                log_pipeline_error(err_msg, traceback.format_exc())
                self.after(0, lambda: self._on_pipeline_done([], err_msg))

        threading.Thread(daemon=True, target=run).start()

    # ── Undo / Redo ────────────────────────────────────────────────────

    def _push_undo(self):
        self._undo_stack.append(snapshot_segments(self.segments))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(snapshot_segments(self.segments))
        self.segments = self._undo_stack.pop()
        self._refresh_table()
        self._refresh_speaker_combos()

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(snapshot_segments(self.segments))
        self.segments = self._redo_stack.pop()
        self._refresh_table()
        self._refresh_speaker_combos()

    # ── Table ──────────────────────────────────────────────────────────

    def _refresh_table(self):
        for w in self._table_widgets:
            frame = w.get("_frame")
            if frame is not None:
                frame.destroy()
            else:
                for k, widget in w.items():
                    if isinstance(widget, tk.Widget):
                        widget.destroy()
        self._table_widgets.clear()

        speakers = list(sorted({s.speaker_id for s in self.segments}))
        for idx, seg in enumerate(self.segments):
            row: dict = {}
            row["_seg"] = seg
            # Speaker color strip
            spk_idx = speakers.index(seg.speaker_id) if seg.speaker_id in speakers else 0
            color = SPEAKER_COLORS[spk_idx % len(SPEAKER_COLORS)]

            row_frame = ctk.CTkFrame(self._table_frame, height=32, fg_color="transparent")
            row_frame.pack(fill="x", pady=0)
            row_frame.pack_propagate(False)

            color_strip = ctk.CTkFrame(row_frame, width=6, fg_color=color, corner_radius=0)
            color_strip.pack(side="left", fill="y")
            row["color_strip"] = color_strip

            start_l = ctk.CTkLabel(row_frame, text=f"{seg.start:.2f}", width=60, height=26)
            start_l.pack(side="left", padx=(4, 2))
            row["start"] = start_l

            end_l = ctk.CTkLabel(row_frame, text=f"{seg.end:.2f}", width=60, height=26)
            end_l.pack(side="left", padx=2)
            row["end"] = end_l

            play_btn = ctk.CTkButton(row_frame, text="\u25b6", width=28, height=24, command=lambda s=seg: self._play_segment(s))
            play_btn.pack(side="left", padx=2)
            row["play"] = play_btn

            split_btn = ctk.CTkButton(row_frame, text="\u2702", width=28, height=24, command=lambda i=idx: self._on_split_segment(i))
            split_btn.pack(side="left", padx=2)
            row["split"] = split_btn

            combo = ctk.CTkComboBox(row_frame, values=speakers, width=120, height=26,
                                     command=lambda v, i=idx: self._on_speaker_change(i, v))
            combo.set(seg.speaker_id)
            combo.pack(side="left", padx=2)
            row["speaker"] = combo

            text_var = ctk.StringVar(value=seg.text)
            text_var.trace_add("write", lambda *a, s=seg, tv=text_var: setattr(s, "text", tv.get()))
            text_e = ctk.CTkEntry(row_frame, textvariable=text_var, height=26)
            text_e.pack(side="left", fill="x", expand=True, padx=(2, 4))
            row["text"] = text_e
            row["_frame"] = row_frame

            self._table_widgets.append(row)

    def _on_speaker_change(self, row_idx: int, value: str):
        self._push_undo()
        seg = self.segments[row_idx]
        seg.speaker_id = value
        # Update color strip for this row
        speakers = list(sorted({s.speaker_id for s in self.segments}))
        spk_idx = speakers.index(value) if value in speakers else 0
        color = SPEAKER_COLORS[spk_idx % len(SPEAKER_COLORS)]
        if row_idx < len(self._table_widgets):
            strip = self._table_widgets[row_idx].get("color_strip")
            if strip:
                strip.configure(fg_color=color)

    # ── Split segment ──────────────────────────────────────────────────

    def _on_split_segment(self, idx: int):
        if idx < 0 or idx >= len(self.segments):
            return
        seg = self.segments[idx]
        mid = round((seg.start + seg.end) / 2, 2)
        val = simpledialog.askfloat("Split segment", f"Split at time (s):", initialvalue=mid,
                                     minvalue=seg.start, maxvalue=seg.end, parent=self)
        if val is None:
            return
        self._push_undo()
        a, b = split_segment(seg, val)
        self.segments[idx:idx + 1] = [a, b]
        self._refresh_table()
        self._refresh_speaker_combos()

    # ── Speaker combos ─────────────────────────────────────────────────

    def _refresh_speaker_combos(self):
        speakers = list(sorted({s.speaker_id for s in self.segments}))
        if not speakers:
            speakers = [""]
        self._label_speaker_combo.configure(values=speakers, state="normal" if speakers != [""] else "disabled")
        if speakers and speakers != [""]:
            self._label_speaker_combo.set(speakers[0])
        self._merge_from_combo.configure(values=speakers, state="normal" if speakers != [""] else "disabled")
        self._merge_to_combo.configure(values=speakers, state="normal" if speakers != [""] else "disabled")
        if speakers and speakers != [""]:
            self._merge_from_combo.set(speakers[0])
            self._merge_to_combo.set(speakers[1] if len(speakers) > 1 else speakers[0])

    def _on_apply_label(self):
        spk = self._label_speaker_combo.get()
        name = self._label_name_entry.get().strip()
        if not spk or not name:
            return
        self._push_undo()
        self.speaker_label_map[spk] = name
        for seg in self.segments:
            if seg.speaker_id == spk:
                seg.speaker_id = name
        self._refresh_table()
        self._refresh_speaker_combos()

    def _on_merge_speakers(self):
        from_id = self._merge_from_combo.get()
        to_id = self._merge_to_combo.get()
        if not from_id or not to_id or from_id == to_id:
            messagebox.showinfo("Merge", "Select two different speakers to merge.")
            return
        self._push_undo()
        for seg in self.segments:
            if seg.speaker_id == from_id:
                seg.speaker_id = to_id
        if from_id in self.speaker_label_map:
            del self.speaker_label_map[from_id]
        self._refresh_table()
        self._refresh_speaker_combos()

    # ── Search ─────────────────────────────────────────────────────────

    def _on_search(self):
        query = self._search_entry.get().strip().lower()
        if not query:
            self._search_matches.clear()
            self._search_idx = -1
            self._search_info.configure(text="")
            self._refresh_table()
            return
        self._search_matches = [i for i, s in enumerate(self.segments) if query in s.text.lower()]
        if self._search_matches:
            self._search_idx = 0
            self._highlight_search()
        else:
            self._search_idx = -1
            self._search_info.configure(text="0 matches")
            self._refresh_table()

    def _on_search_next(self):
        if not self._search_matches:
            return
        self._search_idx = (self._search_idx + 1) % len(self._search_matches)
        self._highlight_search()

    def _on_search_prev(self):
        if not self._search_matches:
            return
        self._search_idx = (self._search_idx - 1) % len(self._search_matches)
        self._highlight_search()

    def _highlight_search(self):
        # Reset all row colors then highlight matches
        for i, w in enumerate(self._table_widgets):
            frame = w.get("_frame")
            if frame is None:
                continue
            if i in self._search_matches:
                frame.configure(fg_color=("gray85", "gray30"))
            else:
                frame.configure(fg_color="transparent")
        # Scroll to current match
        if 0 <= self._search_idx < len(self._search_matches):
            match_row = self._search_matches[self._search_idx]
            if match_row < len(self._table_widgets):
                w = self._table_widgets[match_row].get("_frame")
                if w:
                    # Highlight current match more strongly
                    w.configure(fg_color=("gold", "#665500"))
                    # Scroll into view
                    self._table_frame._parent_canvas.yview_moveto(
                        match_row / max(len(self._table_widgets), 1))
        self._search_info.configure(text=f"{self._search_idx + 1}/{len(self._search_matches)}")

    # ── Export ─────────────────────────────────────────────────────────

    def _export(self, fmt: str):
        if not self.segments:
            messagebox.showinfo("Export", "No transcript to export.")
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}")], initialdir=os.path.expanduser("~"))
        if not path:
            return
        try:
            if fmt == "txt":
                to_txt(self.segments, path, label_map=self.speaker_label_map)
            elif fmt == "srt":
                to_srt(self.segments, path, label_map=self.speaker_label_map)
            elif fmt == "vtt":
                to_vtt(self.segments, path, label_map=self.speaker_label_map)
            elif fmt == "json":
                to_json(self.segments, path, label_map=self.speaker_label_map,
                         audio_path=norm_path_for_windows(self.audio_path) if self.audio_path else None)
            messagebox.showinfo("Export", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ── Playback ───────────────────────────────────────────────────────

    def _on_open_at_time(self):
        if not self.audio_path:
            messagebox.showinfo("Playback", "No audio file loaded.")
            return
        path = norm_path_for_windows(self.audio_path)
        if not os.path.isfile(path):
            messagebox.showerror("Playback", f"File not found:\n{path}")
            return
        import subprocess, platform
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)

    def _play_segment(self, seg: Segment):
        if not self.audio_path:
            messagebox.showinfo("Playback", "No audio file loaded.")
            return
        path = norm_path_for_windows(self.audio_path)
        if not os.path.isfile(path):
            messagebox.showerror("Playback", f"File not found:\n{path}")
            return
        start, end = seg.start, seg.end

        def _run():
            try:
                import sounddevice as sd
                chunk, sr = None, None
                try:
                    import soundfile as sf
                    with sf.SoundFile(path) as f:
                        sr = f.samplerate
                        f.seek(int(start * sr))
                        n = int((end - start) * sr)
                        n = min(n, len(f) - f.tell())
                        if n <= 0:
                            return
                        chunk = f.read(n, dtype="float32")
                except Exception:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        warnings.simplefilter("ignore", category=UserWarning)
                        import librosa
                        y, sr = librosa.load(path, sr=None, mono=True, dtype="float32")
                    n0, n1 = int(start * sr), int(end * sr)
                    n1 = min(n1, len(y))
                    chunk = y[n0:n1] if n1 > n0 else None
                if chunk is None or sr is None:
                    return
                if chunk.ndim == 2:
                    chunk = chunk.mean(axis=1)
                sd.play(chunk, sr)
                sd.wait()
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda m=err_msg: messagebox.showerror("Playback failed", m))

        threading.Thread(target=_run, daemon=True).start()

    # ── Stats ──────────────────────────────────────────────────────────

    def _on_stats(self):
        if not self.segments:
            messagebox.showinfo("Stats", "No segments.")
            return
        _StatsDialog(self, self.segments)

    # ── Theme ──────────────────────────────────────────────────────────

    def _on_theme_toggle(self):
        mode = "dark" if self._theme_var.get() == "on" else "light"
        ctk.set_appearance_mode(mode)
        save_config({"theme": mode})

    # ── Batch ───────────────────────────────────────────────────────────

    def _on_batch(self):
        _BatchWindow(self)

    # ── Summarize ──────────────────────────────────────────────────────

    def _on_summarize(self):
        if not self.segments:
            messagebox.showinfo("Summarize", "No transcript to summarize.")
            return
        _SummarizeDialog(self, self.segments, self.speaker_label_map)

    # ── Logs ───────────────────────────────────────────────────────────

    def _on_open_logs(self):
        log_dir = str(get_log_dir())
        import subprocess, platform
        if platform.system() == "Windows":
            os.startfile(log_dir)
        elif platform.system() == "Darwin":
            subprocess.run(["open", log_dir], check=False)
        else:
            subprocess.run(["xdg-open", log_dir], check=False)

    # ── Load project ───────────────────────────────────────────────────

    def _on_load_project(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("JSON project", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            segments, label_map, audio_path = load_json(path)
            self._push_undo()
            self.segments = segments
            self.speaker_label_map = label_map
            if audio_path:
                self.audio_path = norm_path_for_windows(audio_path)
                self._file_label.configure(text=Path(self.audio_path).name)
            self._refresh_table()
            self._refresh_speaker_combos()
            messagebox.showinfo("Load", "Project loaded.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))


def main():
    cfg = get_config()
    ctk.set_appearance_mode(cfg.get("theme", "dark"))
    ctk.set_default_color_theme("blue")
    app = SecretaryApp()
    app.mainloop()


if __name__ == "__main__":
    main()
