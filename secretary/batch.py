"""Batch processing: queue multiple audio files and run the pipeline on each."""
from dataclasses import dataclass, field
from typing import Callable

from secretary.models import Segment
from secretary.pipeline import run_pipeline


@dataclass
class BatchJob:
    audio_path: str
    status: str = "queued"  # queued | running | done | failed
    segments: list[Segment] = field(default_factory=list)
    error: str | None = None


def run_batch(
    jobs: list[BatchJob],
    *,
    model_name: str = "base",
    language: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    """Run pipeline on each job sequentially. progress_callback(job_index, message)."""
    for i, job in enumerate(jobs):
        if job.status == "done":
            continue
        job.status = "running"
        if progress_callback:
            progress_callback(i, f"Running {job.audio_path}…")
        try:
            job.segments = run_pipeline(
                job.audio_path,
                model_name=model_name,
                language=language,
                progress_callback=lambda msg, idx=i: progress_callback(idx, msg) if progress_callback else None,
            )
            job.status = "done"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
        if progress_callback:
            progress_callback(i, job.status)
