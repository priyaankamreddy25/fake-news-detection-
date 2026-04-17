"""
Web application (Flask) for Fake News Detection.

First train once (creates ./saved_model):
  python train_and_save.py --max-samples 5000

Then run the website:
  python app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

app = Flask(__name__)

_tokenizer: DistilBertTokenizer | None = None
_model: DistilBertForSequenceClassification | None = None
_device: torch.device | None = None
_max_length = 128
_id2label = {0: "FAKE", 1: "REAL"}


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def load_artifacts() -> None:
    global _tokenizer, _model, _device, _max_length, _id2label
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Missing {MODEL_DIR}. Run: python train_and_save.py")

    cfg_path = MODEL_DIR / "label_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        _max_length = int(cfg.get("max_length", 128))
        id2label = cfg.get("id2label")
        if isinstance(id2label, dict) and "0" in id2label and "1" in id2label:
            _id2label = {0: str(id2label["0"]), 1: str(id2label["1"])}

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
    _model.eval()


@app.before_request
def _ensure_loaded():
    if _tokenizer is None or _model is None:
        load_artifacts()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    text = (payload.get("text") or "").strip()
    if not title and not text:
        return jsonify({"error": "Please enter a headline and/or article text."}), 400

    combined = (title + " [SEP] " + text).strip() if title and text else (title or text)
    combined = clean_text(combined)

    assert _tokenizer is not None and _model is not None and _device is not None
    inputs = _tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=_max_length,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    label = _id2label.get(pred_id, str(pred_id))
    confidence = float(probs[pred_id].item())

    return jsonify(
        {
            "label": label,
            "prediction_id": pred_id,
            "confidence": confidence,
            "probabilities": {
                _id2label.get(0, "FAKE"): float(probs[0].item()),
                _id2label.get(1, "REAL"): float(probs[1].item()),
            },
            "device": str(_device),
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)

