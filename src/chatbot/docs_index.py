"""
Section-level search over EPM's own documentation (docs/*.md) for the
``explain`` intent. Answers are verbatim excerpts with their source file and
heading, so nothing is invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"

# Concept → doc file stem; boosts sections of that file for common names.
CONCEPT_FILES: Dict[str, str] = {
    "linear algebra": "01_linear_algebra", "eigen": "01_linear_algebra", "matrix": "01_linear_algebra",
    "gradient descent": "02_gradient_descent", "adam": "02_gradient_descent", "momentum": "02_gradient_descent",
    "logistic regression": "03_logistic_regression", "sigmoid": "03_logistic_regression",
    "neural network": "04_neural_networks", "mlp": "04_neural_networks", "perceptron": "04_neural_networks",
    "backprop": "05_backpropagation", "backpropagation": "05_backpropagation", "chain rule": "05_backpropagation",
    "embedding": "06_embeddings", "word2vec": "06_embeddings", "tf-idf": "06_embeddings", "tfidf": "06_embeddings",
    "attention": "07_attention", "self-attention": "07_attention", "transformer": "08_transformers",
    "time series": "09_time_series", "lstm": "09_time_series", "gru": "09_time_series", "forecasting": "09_time_series",
    "multimodal": "09b_multimodal_fusion", "fusion": "09b_multimodal_fusion", "gmu": "09b_multimodal_fusion",
    "vae": "10_generative_models", "autoencoder": "10_generative_models", "elbo": "10_generative_models",
    "variational": "10_generative_models", "generative": "10_generative_models",
    "diffusion": "11_diffusion", "ddpm": "11_diffusion", "ddim": "11_diffusion",
    "flow matching": "12_flow_matching", "normalizing flow": "12_flow_matching", "cnf": "12_flow_matching",
    "optimal transport": "12_flow_matching", "compression": "13_model_compression",
    "distillation": "13_model_compression", "quantization": "13_model_compression", "quantisation": "13_model_compression",
    "pruning": "13_model_compression", "complexity": "14_complexity_analysis", "linear attention": "14_complexity_analysis",
    "flops": "14_complexity_analysis", "dataset registry": "15_datasets_and_benchmarking",
    "leakage": "15_datasets_and_benchmarking", "benchmark": "15_datasets_and_benchmarking",
    "mase": "15_datasets_and_benchmarking", "pareto": "15_datasets_and_benchmarking",
    "h1": "stage13_research_report", "hypothesis": "stage13_research_report", "chatbot": "16_chatbot",
}


@dataclass
class Section:
    file: str
    title: str
    body: str


def _sections(path: Path) -> List[Section]:
    text = path.read_text(encoding="utf-8", errors="replace")
    doc_title = path.stem
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        doc_title = m.group(1).strip()
    parts = re.split(r"^(#{2,3}\s+.+)$", text, flags=re.M)
    out = [Section(path.name, doc_title, parts[0])]
    for i in range(1, len(parts) - 1, 2):
        out.append(Section(path.name, f"{doc_title} › {parts[i].lstrip('#').strip()}", parts[i + 1]))
    return [s for s in out if len(s.body.strip()) > 40]


class DocsIndex:
    def __init__(self, docs_dir: Optional[Path] = None) -> None:
        self.docs_dir = docs_dir or DOCS_DIR
        self.sections: List[Section] = []
        for p in sorted(self.docs_dir.glob("*.md")) if self.docs_dir.exists() else []:
            self.sections.extend(_sections(p))
        self._vec = None
        if self.sections:
            from sklearn.feature_extraction.text import TfidfVectorizer

            corpus = [(s.title + " ") * 3 + s.body for s in self.sections]
            self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self._mat = self._vec.fit_transform(corpus)

    def topics(self) -> List[str]:
        seen = []
        for s in self.sections:
            t = s.title.split(" › ")[0]
            if t not in seen:
                seen.append(t)
        return seen

    def search(self, query: str, k: int = 1, min_score: float = 0.08) -> List[Dict[str, object]]:
        if not self._vec or not query.strip():
            return []
        q = query.lower()
        scores = (self._mat @ self._vec.transform([q]).T).toarray().ravel()
        for concept, stem in CONCEPT_FILES.items():
            if re.search(r"(?<!\w)" + re.escape(concept) + r"(?!\w)", q):
                for i, s in enumerate(self.sections):
                    if s.file.startswith(stem):
                        scores[i] += 0.25
                        title = s.title.split(" › ")[-1].lower()
                        if concept in title and "conditional" not in title:
                            scores[i] += 0.15
        order = scores.argsort()[::-1][:k]
        out = []
        for i in order:
            if scores[i] < min_score:
                continue
            text = _excerpt(self.sections[i].body)
            j = i + 1  # thin intro sections: continue into the next section of the same file
            while len(text) < 300 and j < len(self.sections) and self.sections[j].file == self.sections[i].file:
                nxt = _excerpt(self.sections[j].body, 900 - len(text))
                text = (text + "\n\n" + "**" + self.sections[j].title.split(" › ")[-1] + "**\n" + nxt).strip()
                j += 1
            out.append({"file": self.sections[i].file, "title": self.sections[i].title,
                        "excerpt": text, "score": round(float(scores[i]), 3)})
        return out


def _excerpt(body: str, limit: int = 900) -> str:
    """First paragraphs of a section, without code fences or tables."""
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    lines = [ln.rstrip() for ln in body.splitlines()]
    keep = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith(("|", "#", "---", "<"))
            and not re.match(r"^>\s*\*", ln.strip())]
    text = "\n".join(keep).strip()
    if len(text) > limit:
        cut = text[:limit]
        text = cut[: max(cut.rfind("\n"), cut.rfind(". ") + 1, limit // 2)].rstrip() + " …"
    return text


_INDEX: Optional[DocsIndex] = None


def get_index() -> DocsIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = DocsIndex()
    return _INDEX
