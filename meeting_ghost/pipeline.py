"""
End-to-end pipeline: audio file -> catch-up.

Run this once everything is installed and your HF + Anthropic keys are set.
Designed so the input is just a file path — when you tackle 'live' later, you
swap this single entry point for a streaming source and the rest is unchanged.
"""

import json
import os

from meeting_ghost.transcribe import transcribe
from meeting_ghost.diarize import diarize
from meeting_ghost.merge import merge
from meeting_ghost.classify import (
    ZeroShotClassifier,
    FineTunedClassifier,
    classify_utterances,
)
from meeting_ghost.catchup import generate_catchup


def run(
    audio_path: str,
    your_name: str | None = None,
    use_finetuned: bool = False,
    whisper_size: str = "small",
    device: str = "cuda",
):
    hf_token = os.environ["HF_TOKEN"]  # set this in Colab secrets

    print("1/4 transcribing...")
    words = transcribe(audio_path, model_size=whisper_size, device=device)

    print("2/4 diarizing (who spoke when)...")
    turns = diarize(audio_path, hf_token=hf_token, device=device)

    print("3/4 merging + classifying...")
    utterances = merge(words, turns)
    clf = FineTunedClassifier() if use_finetuned else ZeroShotClassifier(
        device=0 if device == "cuda" else -1
    )
    classified = classify_utterances(utterances, clf)

    print("4/4 generating catch-up...")
    result = generate_catchup(classified, your_name=your_name)

    return {"classified": classified, **result}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--name", default=None)
    p.add_argument("--finetuned", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    out = run(
        args.audio,
        your_name=args.name,
        use_finetuned=args.finetuned,
        device="cpu" if args.cpu else "cuda",
    )
    print("\n===== CATCH-UP =====\n")
    print(out["summary"])
    if out["your_items"]:
        print("\n===== THINGS FOR YOU =====")
        for item in out["your_items"]:
            print(f"  - {item['text']}")
    with open("output.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n(full output written to output.json)")
