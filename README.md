# Meeting Ghost 🤖

> AI that attends boring meetings for you. Listens, classifies, and generates a 2-minute catch-up.

**Pipeline:**
`Audio → Transcription → Speaker labels → Classification → Catch-up summary`

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/meeting-ghost
cd meeting-ghost
pip install -r requirements.txt
cp .env.example .env  # add your keys
```

**Keys needed:**
- `HF_TOKEN` — free at [huggingface.co](https://huggingface.co) · accept [pyannote license](https://hf.co/pyannote/speaker-diarization-3.1)
- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com)

---

## Sprint status

| Sprint | What | Status |
|--------|------|--------|
| 1 | Transcription (faster-whisper) | ✅ Done |
| 2 | Speaker diarization (pyannote) | 🔲 Next |
| 3 | Sentence classifier (zero-shot) | 🔲 Todo |
| 4 | Catch-up generator (Claude API) | 🔲 Todo |
| 5 | Fine-tune your own classifier | 🔲 Todo |
| 6 | Portfolio polish + demo | 🔲 Todo |

---

## Sprint 1 — Run it

```bash
# Test with synthetic audio (no mic needed)
python test_sprint1.py

# Or transcribe your own file
python -m meeting_ghost.transcribe my_meeting.wav
```

---

## Tech stack

`faster-whisper` · `pyannote.audio` · `HuggingFace Transformers (DistilBERT)` · `Anthropic Claude API`
