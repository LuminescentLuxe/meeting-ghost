"""
Sprint 1 — Speech-to-text using faster-whisper.

Converts a .wav / .mp3 / .m4a file into a list of word-level segments
with start/end timestamps. These timestamps are used in Sprint 2 to
align speaker identities from pyannote.

Usage:
    from meeting_ghost.transcribe import transcribe
    segments = transcribe("meeting.wav")
    # -> [{"start": 0.0, "end": 1.2, "text": "Let's get started"}, ...]
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json


def transcribe(
    audio_path: str | Path,
    model_size: str = "base",
    device: str = "auto",
    language: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Transcribe an audio file and return time-stamped segments.

    Args:
        audio_path:  Path to the audio file (.wav, .mp3, .m4a, etc.)
        model_size:  Whisper model size. Options: tiny, base, small, medium, large-v2.
                     'base' is the sweet spot for Colab free tier — fast and accurate enough.
        device:      'auto' picks GPU if available, else CPU.
                     Pass 'cpu' to force CPU (slower but always works).
        language:    ISO 639-1 code like 'en'. None = auto-detect.
        verbose:     Print progress to stdout.

    Returns:
        List of dicts: [{"start": float, "end": float, "text": str}, ...]
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed.\n"
            "Run: pip install faster-whisper"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Resolve device
    if device == "auto":
        import torch
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
        compute_type = "int8"  # int8 works on all hardware

    if verbose:
        print(f"[transcribe] Loading Whisper '{model_size}' on {resolved_device} ({compute_type})")

    model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)

    if verbose:
        print(f"[transcribe] Transcribing: {audio_path.name}")

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,   # needed for speaker-word alignment in Sprint 2
        beam_size=5,
        vad_filter=True,        # skip silent portions automatically
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    results = []
    for seg in segments_iter:
        results.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  seg.text.strip(),
            # word-level timestamps stored for Sprint 2 merge step
            "words": [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in (seg.words or [])
            ],
        })
        if verbose:
            print(f"  [{seg.start:.1f}s → {seg.end:.1f}s] {seg.text.strip()}")

    if verbose:
        duration = info.duration
        print(f"\n[transcribe] Done. {len(results)} segments · {duration:.1f}s audio · lang={info.language}")

    return results


def save_transcript(segments: list[dict], output_path: str | Path) -> None:
    """Save transcript segments to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"[transcribe] Saved → {output_path}")


def load_transcript(path: str | Path) -> list[dict]:
    """Load a previously saved transcript from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_test_audio(output_path: str | Path = "test_meeting.wav") -> Path:
    """
    Generate a short synthetic test audio file using gTTS (Google Text-to-Speech).
    This lets you test the pipeline WITHOUT recording a real meeting.

    Requires: pip install gTTS pydub

    The generated audio simulates a meeting with 3 typical lines so you can
    immediately verify classify.py picks up decisions and tasks.
    """
    try:
        from gtts import gTTS
        import io
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pip install pydub")
    except ImportError:
        raise ImportError("pip install gTTS pydub")

    lines = [
        "Okay everyone, let's get started with today's standup.",
        "We have decided to push the release to next Friday.",
        "Ali, can you write the deployment runbook by Wednesday?",
        "Does anyone have concerns about the new authentication flow?",
        "I think we should keep the old login page for now.",
        "Alright, that's everything. Thanks everyone.",
    ]

    output_path = Path(output_path)
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=600)  # 0.6s pause between lines

    for line in lines:
        buf = io.BytesIO()
        gTTS(text=line, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        seg = AudioSegment.from_mp3(buf)
        combined += seg + silence

    combined.export(str(output_path), format="wav")
    print(f"[generate_test_audio] Saved test audio → {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m meeting_ghost.transcribe <audio_file> [model_size]")
        print("       python -m meeting_ghost.transcribe --generate-test")
        sys.exit(1)

    if sys.argv[1] == "--generate-test":
        audio = generate_test_audio("test_meeting.wav")
        segments = transcribe(audio)
    else:
        audio = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "base"
        segments = transcribe(audio, model_size=model)

    save_transcript(segments, "transcript.json")
    print(f"\nFull transcript:\n{'─'*50}")
    for s in segments:
        print(f"[{s['start']:.1f}s] {s['text']}")
