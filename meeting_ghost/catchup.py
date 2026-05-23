"""
The 2-minute catch-up generator.

Takes the classified transcript, drops the small_talk, and asks Claude Haiku to
write a tight catch-up. This is the one place an LLM clearly beats anything you'd
build by hand, and at Haiku prices a run costs a fraction of a cent.

The 'your name' feature is here too: pass your_name and we surface every line
where you were mentioned or assigned something — the "interrupt only when
relevant" promise, done as post-processing first.
"""

import json

CATCHUP_PROMPT = """You are a meeting catch-up assistant. Below is a classified \
transcript of a meeting. Each line has a speaker, the text, and a label \
(decision, task_assigned, important_question, small_talk).

Write a catch-up someone could read in under 2 minutes. Use these sections, \
omitting any that are empty:
- Decisions made
- Tasks assigned (include who, if known)
- Open questions
Keep it tight and skimmable. Do not invent anything not in the transcript.

Transcript:
{transcript}
"""


def _format_transcript(classified, drop_small_talk=True):
    lines = []
    for item in classified:
        if drop_small_talk and item["label"] == "small_talk":
            continue
        lines.append(f"[{item['label']}] {item['speaker']}: {item['text']}")
    return "\n".join(lines)


def find_mentions(classified, your_name: str):
    """Lines where you were named or (heuristically) given a task."""
    name_lower = your_name.lower()
    hits = []
    for item in classified:
        text_lower = item["text"].lower()
        if name_lower in text_lower or (
            item["label"] == "task_assigned" and name_lower in text_lower
        ):
            hits.append(item)
    return hits


def generate_catchup(classified, your_name=None, model="claude-haiku-4-5-20251001"):
    """
    Returns a dict: {"summary": str, "your_items": list}.
    Requires ANTHROPIC_API_KEY in the environment.
    """
    import anthropic

    client = anthropic.Anthropic()
    transcript = _format_transcript(classified)

    msg = client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": CATCHUP_PROMPT.format(transcript=transcript),
        }],
    )
    summary = "".join(b.text for b in msg.content if b.type == "text")

    your_items = find_mentions(classified, your_name) if your_name else []
    return {"summary": summary, "your_items": your_items}


if __name__ == "__main__":
    # Demo with a canned classified transcript — no audio or API needed to eyeball
    # the formatting (the API call itself needs a key).
    demo = [
        {"speaker": "SPEAKER_00", "text": "We will launch Friday.", "label": "decision", "start": 0, "end": 2},
        {"speaker": "SPEAKER_00", "text": "Can you handle the slides, Ali?", "label": "task_assigned", "start": 3, "end": 5},
        {"speaker": "SPEAKER_01", "text": "Nice weather today huh.", "label": "small_talk", "start": 5, "end": 7},
    ]
    print(_format_transcript(demo))
    print("---")
    print("Mentions of 'Ali':", json.dumps(find_mentions(demo, "Ali"), indent=2))
