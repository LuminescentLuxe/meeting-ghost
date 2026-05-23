"""
Sprint 1 Test — Run this to verify your transcription pipeline works.

Usage:
    python test_sprint1.py

What it does:
    1. Checks all Sprint 1 dependencies are installed
    2. Generates a synthetic test audio file (no mic needed)
    3. Runs faster-whisper transcription on it
    4. Prints results and confirms the output structure is correct

Expected output:
    ✅ faster-whisper installed
    ✅ torch installed
    🎙️  Generating test audio...
    🔊 Transcribing...
    [0.0s] Okay everyone let's get started...
    ...
    ✅ Sprint 1 PASSED — X segments transcribed
"""

import sys


def check_imports():
    """Check all Sprint 1 deps are available."""
    missing = []
    for pkg, import_name in [
        ("faster-whisper", "faster_whisper"),
        ("torch", "torch"),
        ("gTTS", "gtts"),
        ("pydub", "pydub"),
    ]:
        try:
            __import__(import_name)
            print(f"  ✅ {pkg} installed")
        except ImportError:
            print(f"  ❌ {pkg} MISSING — run: pip install {pkg}")
            missing.append(pkg)
    return missing


def main():
    print("=" * 50)
    print("Meeting Ghost — Sprint 1 Test")
    print("=" * 50)

    print("\n📦 Checking dependencies...")
    missing = check_imports()

    if missing:
        print(f"\n❌ Install missing packages and re-run.")
        sys.exit(1)

    print("\n🎙️  Generating test audio...")
    from meeting_ghost.transcribe import generate_test_audio, transcribe, save_transcript

    audio_path = generate_test_audio("test_meeting.wav")
    print(f"  Created: {audio_path}")

    print("\n🔊 Transcribing (this downloads Whisper 'base' ~150MB on first run)...")
    segments = transcribe(audio_path, model_size="base", verbose=True)

    print("\n📄 Full transcript:")
    print("─" * 40)
    for s in segments:
        print(f"  [{s['start']:5.1f}s → {s['end']:5.1f}s]  {s['text']}")

    print("\n💾 Saving to transcript.json...")
    save_transcript(segments, "transcript.json")

    # Validate structure
    assert isinstance(segments, list), "segments should be a list"
    assert len(segments) > 0, "should have at least 1 segment"
    assert "start" in segments[0], "each segment needs 'start'"
    assert "end" in segments[0], "each segment needs 'end'"
    assert "text" in segments[0], "each segment needs 'text'"
    assert "words" in segments[0], "each segment needs 'words' (for Sprint 2)"

    print(f"\n✅ Sprint 1 PASSED — {len(segments)} segments transcribed")
    print("   Output saved to: transcript.json")
    print("\n→ Next: run Sprint 2 (diarization) once this is working.")


if __name__ == "__main__":
    main()
