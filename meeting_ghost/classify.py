"""
Sentence classification — the heart of the project and the part that proves
ML skill (not just API-calling) on your CV.

Two paths, deliberately:
  1. ZeroShotClassifier — works on day one, no training data. Uses an NLI model
     to score each label as a hypothesis. This is your BASELINE.
  2. FineTunedClassifier — trained on YOUR labeled conversations. This is the
     thing you show off. The portfolio story is: "here is my baseline, here is
     my fine-tuned model, here are the numbers showing it's better."

Labels are your four categories. Keep them consistent everywhere.
"""

LABELS = ["decision", "task_assigned", "important_question", "small_talk"]

# Human-readable descriptions help the zero-shot model a lot — the raw label
# "task_assigned" is less meaningful to an NLI model than a natural sentence.
ZEROSHOT_HYPOTHESES = {
    "decision": "This states a decision that was made.",
    "task_assigned": "This assigns a task or responsibility to someone.",
    "important_question": "This asks an important question that needs an answer.",
    "small_talk": "This is casual small talk or chit-chat.",
}


class ZeroShotClassifier:
    """Baseline. No training required. Slower but works immediately."""

    def __init__(self, model="facebook/bart-large-mnli", device=0):
        from transformers import pipeline
        self.pipe = pipeline("zero-shot-classification", model=model, device=device)
        self.candidate_labels = list(ZEROSHOT_HYPOTHESES.values())
        self._desc_to_label = {v: k for k, v in ZEROSHOT_HYPOTHESES.items()}

    def predict(self, text: str) -> str:
        out = self.pipe(text, self.candidate_labels)
        return self._desc_to_label[out["labels"][0]]


class FineTunedClassifier:
    """
    Your trained model. Loads a DistilBERT fine-tuned by train_classifier.py.
    Tiny, fast, and runs on CPU fine — good for the 'live' aspiration later.
    """

    def __init__(self, model_dir="models/classifier"):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        idx = int(logits.argmax(dim=-1))
        return self.model.config.id2label[idx]


def classify_utterances(utterances, classifier):
    """Attach a .label to each utterance. Returns list of dicts for easy JSON."""
    results = []
    for u in utterances:
        results.append({
            "speaker": u.speaker,
            "text": u.text,
            "start": u.start,
            "end": u.end,
            "label": classifier.predict(u.text),
        })
    return results
