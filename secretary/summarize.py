"""Summarize a transcript using a local Ollama model."""
import json
import urllib.request
import urllib.error

from secretary.models import Segment, get_display_speaker


def build_prompt(segments: list[Segment], label_map: dict[str, str] | None = None) -> str:
    """Build a plain-text transcript for the LLM prompt."""
    label_map = label_map or {}
    lines = []
    for seg in segments:
        spk = get_display_speaker(seg.speaker_id, label_map)
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {spk}: {seg.text}")
    return "\n".join(lines)


def summarize_transcript(
    segments: list[Segment],
    label_map: dict[str, str] | None = None,
    *,
    model: str = "llama3",
    endpoint: str = "http://localhost:11434",
) -> str:
    """Call Ollama REST API to summarize the transcript. Returns the summary string."""
    transcript = build_prompt(segments, label_map)
    system_prompt = (
        "You are a meeting summarizer. Given the transcript below, produce a concise summary "
        "with key points, action items, and decisions. Use bullet points."
    )
    payload = json.dumps({
        "model": model,
        "prompt": f"{system_prompt}\n\n---\n\n{transcript}\n\n---\n\nSummary:",
        "stream": False,
    }).encode("utf-8")

    url = f"{endpoint.rstrip('/')}/api/generate"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach Ollama at {url}: {e}") from e
