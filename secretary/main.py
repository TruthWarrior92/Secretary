"""Secretary GUI: upload audio, transcribe + diarize, edit segments, export."""
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

# Add project root for config
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import DEFAULT_WHISPER_MODEL, get_hf_token
from secretary.diarize_utils import diarize_text
from secretary.export import to_json, to_srt, to_txt, to_vtt, load_json
from secretary.models import Segment, get_display_speaker
from secretary.pipeline import load_diarization_pipeline, load_whisper, run_pipeline


# Supported audio extensions
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def get_audio_duration(path: str) -> float:
    """Return duration in seconds; 0 if unreadable."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, duration=0)
            return len(y) / sr if sr else 0
        except Exception:
            return 0.0


class SecretaryApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Secretary — Transcription & Diarization")
        self.geometry("1000x650")
        self.minsize(800, 500)

        self.audio_path: str | None = None
        self.segments: list[Segment] = []
        self.speaker_label_map: dict[str, str] = {}
        self._progress_var = ctk.StringVar(value="")
        self._run_button: ctk.CTkButton | None = None
        self._table_frame: ctk.CTkScrollableFrame | None = None
        self._table_widgets: list[dict] = []
        self._speaker_combo: ctk.CTkComboBox | None = None
        self._merge_from_combo: ctk.CTkComboBox | None = None
        self._merge_to_combo: ctk.CTkComboBox | None = None

        self._build_ui()

    def _build_ui(self):
        # Top: file and run
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=10)

        self._file_label = ctk.CTkLabel(top, text="No file selected", anchor="w")
        self._file_label.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(top, text="Select audio…", width=120, command=self._on_select_file).pack(
            side="right", padx=(5, 0)
        )
        self._run_button = ctk.CTkButton(
            top, text="Transcribe & diarize", width=160, command=self._on_run_pipeline, state="disabled"
        )
        self._run_button.pack(side="right")

        # Progress
        self._progress = ctk.CTkLabel(self, textvariable=self._progress_var, anchor="w")
        self._progress.pack(fill="x", padx=10, pady=(0, 5))

        # Options row: model, language
        opt = ctk.CTkFrame(self, fg_color="transparent")
        opt.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(opt, text="Whisper model:").pack(side="left", padx=(0, 5))
        self._model_combo = ctk.CTkComboBox(
            opt, values=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "large"], width=120
        )
        self._model_combo.set(DEFAULT_WHISPER_MODEL)
        self._model_combo.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(opt, text="Language (blank=auto):").pack(side="left", padx=(0, 5))
        self._lang_entry = ctk.CTkEntry(opt, width=80, placeholder_text="auto")
        self._lang_entry.pack(side="left")

        # Segment table area
        table_container = ctk.CTkFrame(self, fg_color="transparent")
        table_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Table header
        header = ctk.CTkFrame(table_container, fg_color="transparent")
        header.pack(fill="x")
        for col, (text, w) in enumerate([("Start", 70), ("End", 70), ("Speaker", 120), ("Text", 400)]):
            ctk.CTkLabel(header, text=text, width=w if col < 3 else 0).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Text", anchor="w").pack(side="left", fill="x", expand=True, padx=2)

        self._table_frame = ctk.CTkScrollableFrame(table_container, fg_color="transparent")
        self._table_frame.pack(fill="both", expand=True)

        # Speaker edit area: label speakers, correct, merge
        edit_frame = ctk.CTkFrame(self, fg_color="transparent")
        edit_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(edit_frame, text="Label speaker:").pack(side="left", padx=(0, 5))
        self._label_speaker_combo = ctk.CTkComboBox(edit_frame, values=[""], width=100, state="disabled")
        self._label_speaker_combo.pack(side="left", padx=(0, 5))
        self._label_name_entry = ctk.CTkEntry(edit_frame, width=120, placeholder_text="Display name")
        self._label_name_entry.pack(side="left", padx=(0, 5))
        ctk.CTkButton(edit_frame, text="Apply label", width=90, command=self._on_apply_label).pack(side="left", padx=(0, 15))

        ctk.CTkLabel(edit_frame, text="Merge:").pack(side="left", padx=(0, 5))
        self._merge_from_combo = ctk.CTkComboBox(edit_frame, values=[""], width=100, state="disabled")
        self._merge_from_combo.pack(side="left", padx=(0, 5))
        ctk.CTkLabel(edit_frame, text="→").pack(side="left", padx=2)
        self._merge_to_combo = ctk.CTkComboBox(edit_frame, values=[""], width=100, state="disabled")
        self._merge_to_combo.pack(side="left", padx=(0, 5))
        ctk.CTkButton(edit_frame, text="Merge speakers", width=110, command=self._on_merge_speakers).pack(side="left", padx=(0, 15))

        # Export and playback
        export_frame = ctk.CTkFrame(self, fg_color="transparent")
        export_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(export_frame, text="Export TXT", width=90, command=lambda: self._export("txt")).pack(side="left", padx=(0, 5))
        ctk.CTkButton(export_frame, text="Export SRT", width=90, command=lambda: self._export("srt")).pack(side="left", padx=(0, 5))
        ctk.CTkButton(export_frame, text="Export VTT", width=90, command=lambda: self._export("vtt")).pack(side="left", padx=(0, 5))
        ctk.CTkButton(export_frame, text="Export JSON", width=90, command=lambda: self._export("json")).pack(side="left", padx=(0, 5))
        ctk.CTkButton(export_frame, text="Open at time (player)", width=160, command=self._on_open_at_time).pack(side="left", padx=(15, 0))
        ctk.CTkButton(export_frame, text="Load project…", width=100, command=self._on_load_project).pack(side="left", padx=(5, 0))

    def _on_select_file(self):
        path = filedialog.askopenfilename(
            title="Select audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                ("WAV", "*.wav"),
                ("MP3", "*.mp3"),
                ("All", "*.*"),
            ],
        )
        if path:
            self.audio_path = path
            name = Path(path).name
            dur = get_audio_duration(path)
            dur_s = f"{dur:.1f}s" if dur > 0 else "?"
            self._file_label.configure(text=f"{name} ({dur_s})")
            self._run_button.configure(state="normal")

    def _on_run_pipeline(self):
        if not self.audio_path or not os.path.isfile(self.audio_path):
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        if not get_hf_token():
            messagebox.showerror(
                "Missing token",
                "Set HUGGINGFACE_TOKEN in .env or ~/.secretary/config.\n"
                "Create token: https://hf.co/settings/tokens\n"
                "Accept terms: https://huggingface.co/pyannote/speaker-diarization",
            )
            return
        self._run_button.configure(state="disabled")
        self._progress_var.set("Running…")

        def run():
            try:
                model = self._model_combo.get().strip()
                lang = self._lang_entry.get().strip() or None
                segs = run_pipeline(
                    self.audio_path,
                    model_name=model or DEFAULT_WHISPER_MODEL,
                    language=lang,
                    progress_callback=lambda msg: self.after(0, lambda: self._progress_var.set(msg)),
                )
                self.after(0, lambda: self._on_pipeline_done(segs, None))
            except Exception as e:
                self.after(0, lambda: self._on_pipeline_done([], str(e)))

        threading.Thread(daemon=True, target=run).start()

    def _on_pipeline_done(self, segments: list[Segment], error: str | None):
        self._run_button.configure(state="normal")
        if error:
            self._progress_var.set("")
            messagebox.showerror("Pipeline failed", error)
            return
        self.segments = segments
        self._progress_var.set("")
        self._refresh_table()
        self._refresh_speaker_combos()

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
        for seg in self.segments:
            row: dict = {}
            row_frame = ctk.CTkFrame(self._table_frame, fg_color="transparent")
            row_frame.pack(fill="x")

            start_l = ctk.CTkLabel(row_frame, text=f"{seg.start:.2f}", width=70)
            start_l.pack(side="left", padx=2, pady=2)
            row["start"] = start_l

            end_l = ctk.CTkLabel(row_frame, text=f"{seg.end:.2f}", width=70)
            end_l.pack(side="left", padx=2, pady=2)
            row["end"] = end_l

            combo = ctk.CTkComboBox(
                row_frame,
                values=speakers,
                width=120,
                command=lambda v, s=seg: setattr(s, "speaker_id", v),
            )
            combo.set(seg.speaker_id)
            combo.pack(side="left", padx=2, pady=2)
            row["speaker"] = combo

            text_var = ctk.StringVar(value=seg.text)
            text_var.trace_add("write", lambda *a, seg=seg: setattr(seg, "text", text_var.get()))
            text_e = ctk.CTkEntry(row_frame, textvariable=text_var, width=400)
            text_e.pack(side="left", fill="x", expand=True, padx=2, pady=2)
            row["text"] = text_e
            row["_frame"] = row_frame

            self._table_widgets.append(row)

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
        self.speaker_label_map[spk] = name
        self._refresh_table()

    def _on_merge_speakers(self):
        from_id = self._merge_from_combo.get()
        to_id = self._merge_to_combo.get()
        if not from_id or not to_id or from_id == to_id:
            messagebox.showinfo("Merge", "Select two different speakers to merge.")
            return
        for seg in self.segments:
            if seg.speaker_id == from_id:
                seg.speaker_id = to_id
        if from_id in self.speaker_label_map:
            del self.speaker_label_map[from_id]
        self._refresh_table()
        self._refresh_speaker_combos()

    def _export(self, fmt: str):
        if not self.segments:
            messagebox.showinfo("Export", "No transcript to export. Run Transcribe & diarize first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}")],
        )
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
                to_json(
                    self.segments,
                    path,
                    label_map=self.speaker_label_map,
                    audio_path=self.audio_path,
                )
            messagebox.showinfo("Export", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _on_open_at_time(self):
        if not self.segments:
            messagebox.showinfo("Playback", "No segments. Run pipeline first.")
            return
        # Find first selected or first segment; open system default player at timestamp if possible
        start = self.segments[0].start
        if self.audio_path and os.path.isfile(self.audio_path):
            import subprocess
            import platform
            path = self.audio_path
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
            messagebox.showinfo("Playback", f"Opened audio. Seek to {start:.1f}s in your player.")
        else:
            messagebox.showinfo("Playback", f"No audio path; seek to {start:.1f}s in your player.")

    def _on_load_project(self):
        path = filedialog.askopenfilename(filetypes=[("JSON project", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            segments, label_map, audio_path = load_json(path)
            self.segments = segments
            self.speaker_label_map = label_map
            if audio_path:
                self.audio_path = audio_path
                self._file_label.configure(text=Path(audio_path).name)
            self._refresh_table()
            self._refresh_speaker_combos()
            messagebox.showinfo("Load", "Project loaded.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = SecretaryApp()
    app.mainloop()


if __name__ == "__main__":
    main()
