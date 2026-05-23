"""
Speaker diarization: figuring out *who* spoke *when*.

Whisper tells us what was said and at what time. It does NOT tell us who said
it. pyannote.audio segments the audio into speaker turns (SPEAKER_00,
SPEAKER_01, ...). The merge step then combines the two.

NOTE: pyannote's pretrained pipeline is gated on Hugging Face. You must:
  1. Make a free HF account
  2. Accept the user conditions on the model page
     (huggingface.co/pyannote/speaker-diarization-3.1)
  3. Create a read token and pass it below (in Colab: store as a secret).
This is free — the gating is just a license click-through.
"""

from dataclasses import dataclass


@dataclass
class SpeakerTurn:
    speaker: str   # e.g. "SPEAKER_00"
    start: float   # seconds
    end: float


def diarize(audio_path: str, hf_token: str, device: str = "cuda"):
    """
    Return a list of SpeakerTurn covering the whole file, in time order.

    If you already know how many people are in the recording, passing
    num_speakers makes results noticeably cleaner — see the kwarg below.
    """
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device))

    # If you know the count, e.g. a 4-person study group:
    #   diarization = pipeline(audio_path, num_speakers=4)
    diarization = pipeline(audio_path)

    turns: list[SpeakerTurn] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(speaker=speaker, start=turn.start, end=turn.end))
    turns.sort(key=lambda t: t.start)
    return turns
