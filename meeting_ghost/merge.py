"""
The merge step: align Whisper words with pyannote speaker turns, then group
consecutive same-speaker words into utterances (sentence-ish chunks).

This is the fiddly part of the whole pipeline. The two models work on slightly
different time grids, so a word's timestamp won't perfectly nest inside one
speaker turn. Strategy: assign each word to the speaker turn it *overlaps most*
with. Then collapse runs of same-speaker words into utterances, breaking on
sentence-ending punctuation so downstream classification gets clean sentences.
"""

from dataclasses import dataclass


@dataclass
class Utterance:
    speaker: str
    text: str
    start: float
    end: float


def _speaker_for_word(word_start, word_end, turns):
    """Pick the speaker turn with maximum temporal overlap with this word."""
    best_speaker = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(word_end, turn.end) - max(word_start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    # Fallback: if a word overlaps no turn (gaps happen), attach to the nearest.
    if best_speaker is None and turns:
        best_speaker = min(
            turns, key=lambda t: min(abs(t.start - word_start), abs(t.end - word_end))
        ).speaker
    return best_speaker


def merge(words, turns):
    """
    words: list[Word] from transcribe.py
    turns: list[SpeakerTurn] from diarize.py
    Returns: list[Utterance]
    """
    if not words:
        return []

    # 1. Tag every word with a speaker.
    tagged = [
        (w, _speaker_for_word(w.start, w.end, turns)) for w in words
    ]

    # 2. Group consecutive words, breaking when the speaker changes OR when a
    #    word ends a sentence (so utterances stay sentence-sized for the NLP step).
    utterances: list[Utterance] = []
    cur_speaker = tagged[0][1]
    cur_words = []
    cur_start = tagged[0][0].start

    def flush(end_time):
        if cur_words:
            utterances.append(
                Utterance(
                    speaker=cur_speaker,
                    text=" ".join(cur_words).strip(),
                    start=cur_start,
                    end=end_time,
                )
            )

    for word, speaker in tagged:
        if speaker != cur_speaker:
            flush(word.start)
            cur_speaker = speaker
            cur_words = []
            cur_start = word.start
        cur_words.append(word.text)
        if word.text.endswith((".", "?", "!")):
            flush(word.end)
            cur_words = []
            cur_start = word.end  # next word starts a new utterance

    flush(words[-1].end)
    return [u for u in utterances if u.text]
