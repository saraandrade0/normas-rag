"""
Wrapper do classificador BERT fine-tuned.
Classifica perguntas em categorias de normas para pré-filtrar a busca.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NormaClassifier:
    """
    Classificador de perguntas por categoria de norma técnica.
    Usa BERT fine-tuned para prever a categoria antes da busca vetorial,
    reduzindo o espaço de busca e melhorando a relevância.
    """

    def __init__(self, model_path: str = "finetune/model/best"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            ).to(self.device)
            self.model.eval()

            # Carrega mapeamento de categorias
            cat_path = self.model_path / "categories.json"
            if cat_path.exists():
                with open(cat_path) as f:
                    cats = json.load(f)
                    self.id2cat = {int(k): v for k, v in cats["id2cat"].items()}
            else:
                self.id2cat = self.model.config.id2label

            self.ready = True
            print(f"✓ Classificador carregado: {model_path}")
        else:
            self.ready = False
            print(f"⚠ Classificador não encontrado em {model_path}. Busca sem pré-filtro.")

    def classify(self, text: str, threshold: float = 0.6) -> str | None:
        """
        Classifica uma pergunta e retorna a categoria.
        Retorna None se a confiança for abaixo do threshold (busca sem filtro).
        """
        if not self.ready:
            return None

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_id = torch.max(probs, dim=-1)

        if confidence.item() < threshold:
            return None  # Confiança baixa → busca em todas as categorias

        return self.id2cat[pred_id.item()]
