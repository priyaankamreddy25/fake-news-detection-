"""
Train DistilBERT once on datasets/True.csv + datasets/Fake.csv,
then save model + tokenizer to ./saved_model for the web app.

Quick (recommended):
  python train_and_save.py --max-samples 5000

Full dataset (slow on CPU):
  python train_and_save.py --full
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets", help="Folder containing True.csv and Fake.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "saved_model", help="Where to save model + tokenizer")
    p.add_argument("--max-samples", type=int, default=5000, help="Max rows (balanced) for quick training (ignored if --full)")
    p.add_argument("--full", action="store_true", help="Train on all rows (slow on CPU)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=None, help="Default: 32 on GPU, 16 on CPU")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    true_csv = data_dir / "True.csv"
    fake_csv = data_dir / "Fake.csv"
    if not true_csv.is_file() or not fake_csv.is_file():
        raise SystemExit(f"Missing CSVs. Expected {true_csv} and {fake_csv}")

    use_cuda = torch.cuda.is_available()
    batch_size = args.batch_size if args.batch_size is not None else (32 if use_cuda else 16)
    print(f"Device: {'cuda' if use_cuda else 'cpu'} | batch_size={batch_size} | fp16={use_cuda}")

    # Read only the columns we need (your CSVs often contain many empty "Unnamed:" columns)
    real_df = pd.read_csv(true_csv, usecols=["title", "text"])
    fake_df = pd.read_csv(fake_csv, usecols=["title", "text"])
    real_df["label"] = 1  # REAL
    fake_df["label"] = 0  # FAKE
    data = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    if not args.full and len(data) > args.max_samples:
        # balanced subset (keeps both classes represented)
        n_each = args.max_samples // 2
        real = data[data["label"] == 1].sample(min(n_each, (data["label"] == 1).sum()), random_state=42)
        fake = data[data["label"] == 0].sample(min(n_each, (data["label"] == 0).sum()), random_state=42)
        data = pd.concat([real, fake], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Using balanced subset: {len(data)} rows (cap {args.max_samples}).")
    else:
        print(f"Using dataset: {len(data)} rows")

    if "title" not in data.columns or "text" not in data.columns:
        raise SystemExit("CSV must contain columns: title, text")

    data["content"] = data["title"].astype(str) + " [SEP] " + data["text"].astype(str)
    data = data[["content", "label"]]
    data["content"] = data["content"].apply(clean_text)

    texts = data["content"].tolist()
    labels = data["label"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_length)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "results"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=str(args.output_dir / "logs"),
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=bool(use_cuda),
        dataloader_pin_memory=bool(use_cuda),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(val_dataset).predictions.argmax(axis=1)
    print(classification_report(val_labels, preds, digits=4))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "label_config.json").write_text(
        json.dumps({"id2label": {"0": "FAKE", "1": "REAL"}, "max_length": args.max_length}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

