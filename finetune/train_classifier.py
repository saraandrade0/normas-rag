"""
Fine-tuning de BERT para classificação de categorias de normas técnicas.

Classifica perguntas em categorias (eletrica, acessibilidade, estrutural, etc.)
para pré-filtrar documentos antes da busca vetorial.

Modelo base: neuralmind/bert-base-portuguese-cased
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset


# ─── CONFIG ──────────────────────────────────────────────
BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
CATEGORIES = [
    "eletrica",
    "acessibilidade",
    "estrutural",
    "hidraulica",
    "incendio",
    "urbanismo",
    "geral",
]
CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2CAT = {i: c for c, i in CAT2ID.items()}


def load_data(path: str) -> list[dict]:
    """Carrega dataset JSONL: {"text": "...", "label": "eletrica"}"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            item["label_id"] = CAT2ID[item["label"]]
            data.append(item)
    print(f"Dataset: {len(data)} exemplos, {len(set(d['label'] for d in data))} categorias")
    return data


def tokenize_data(data: list[dict], tokenizer):
    """Tokeniza textos e cria Dataset do HuggingFace."""
    texts = [d["text"] for d in data]
    labels = [d["label_id"] for d in data]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    return dataset


def compute_metrics(eval_pred):
    """Calcula acurácia para avaliação durante treino."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train(data_path: str, output_dir: str, epochs: int = 5, batch_size: int = 16):
    """Pipeline completo de fine-tuning."""

    # 1. Carrega dados
    data = load_data(data_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=[d["label"] for d in data])
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # 2. Tokeniza
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = tokenize_data(train_data, tokenizer)
    val_dataset = tokenize_data(val_data, tokenizer)

    # 3. Carrega modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CATEGORIES),
        id2label=ID2CAT,
        label2id=CAT2ID,
    )

    # 4. Treina
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\nTreinando {BASE_MODEL} por {epochs} epochs...")
    trainer.train()

    # 5. Avalia
    val_preds = trainer.predict(val_dataset)
    preds = np.argmax(val_preds.predictions, axis=-1)
    labels = [d["label_id"] for d in val_data]
    print("\n" + classification_report(labels, preds, target_names=CATEGORIES))

    # 6. Salva modelo e tokenizer
    model_path = Path(output_dir) / "best"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"\n✓ Modelo salvo em {model_path}")

    # Salva mapeamento de categorias
    with open(model_path / "categories.json", "w") as f:
        json.dump({"cat2id": CAT2ID, "id2cat": ID2CAT}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning do classificador de normas")
    parser.add_argument("--data", default="data/categorias.jsonl")
    parser.add_argument("--output", default="finetune/model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size)
