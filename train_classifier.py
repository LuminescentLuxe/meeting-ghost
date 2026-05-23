"""
Fine-tune DistilBERT on YOUR labeled conversations, and — critically for the
portfolio — evaluate it against the zero-shot baseline so you have real numbers.

The CV-impressing output of this script is the printed comparison table:
  zero-shot accuracy/F1  vs  fine-tuned accuracy/F1.
Put that table in your README.

Expected data format: a CSV at data/labeled.csv with columns:
    text,label
where label is one of: decision, task_assigned, important_question, small_talk

Aim for ~300+ labeled sentences to start; more small_talk than anything else is
fine and realistic — just report per-class F1 so imbalance is visible.
"""

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

LABELS = ["decision", "task_assigned", "important_question", "small_talk"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

MODEL = "distilbert-base-uncased"


def load_data(path="data/labeled.csv"):
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
    return train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )


def evaluate_zeroshot(test_df):
    """Baseline numbers, computed on the SAME test set for a fair comparison."""
    from meeting_ghost.classify import ZeroShotClassifier
    clf = ZeroShotClassifier(device=-1)
    preds = [clf.predict(t) for t in test_df["text"]]
    truth = list(test_df["label"])
    return preds, truth


def train(train_df, test_df, out_dir="models/classifier", epochs=4):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )

    tok = AutoTokenizer.from_pretrained(MODEL)

    def prep(df):
        ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128),
                    batched=True)
        ds = ds.map(lambda b: {"labels": label2id[b["label"]]})
        return ds

    train_ds, test_ds = prep(train_df), prep(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    ft_preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    ft_preds = [id2label[i] for i in ft_preds]
    return ft_preds, list(test_df["label"])


if __name__ == "__main__":
    train_df, test_df = load_data()

    print("Evaluating zero-shot baseline...")
    zs_preds, truth = evaluate_zeroshot(test_df)

    print("Fine-tuning...")
    ft_preds, _ = train(train_df, test_df)

    print("\n" + "=" * 50)
    print("RESULTS  (put this table in your README)")
    print("=" * 50)
    print(f"{'model':<14}{'accuracy':>10}{'macro-F1':>10}")
    for name, preds in [("zero-shot", zs_preds), ("fine-tuned", ft_preds)]:
        acc = accuracy_score(truth, preds)
        f1 = f1_score(truth, preds, average="macro")
        print(f"{name:<14}{acc:>10.3f}{f1:>10.3f}")
    print("\nPer-class (fine-tuned):")
    print(classification_report(truth, ft_preds))
