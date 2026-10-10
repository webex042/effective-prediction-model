"""
EPM Stage 4 — NLP, Embeddings & Self-Attention Demo
===================================================

This script demonstrates all Stage 4 modules:
    1. Classical NLP Pipeline: Text -> Cleaning -> Tokenization -> TF-IDF -> Classifier
    2. Dense Word Embeddings: Token IDs -> Embedding Lookup -> Pooling -> Neural Classifier
    3. Scaled Dot-Product Self-Attention from Scratch: Q, K, V matrix math & √d_k scaling
    4. Attention Heatmap Visualization: Inspecting where the model attends
    5. Transformer Sequence Classifier: Positional Encoding + Multi-Head Attention
    6. Comprehensive NLP Pipeline Benchmark & Trade-off Analysis

Run:
    python demo_stage4.py
"""

import sys
import io
import os
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Reconfigure stdout for utf-8 safety on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.reproducibility import set_seed
from src.logger import setup_logging
from src.nlp import (
    clean_text,
    simple_tokenize,
    Vocabulary,
    TFIDFVectorizerScratch,
    PyTorchEmbeddingClassifier,
)
from src.transformers_ import (
    scaled_dot_product_attention_numpy,
    MultiHeadAttention,
    TextTransformerClassifier,
)
from src.classical_ml.logistic_regression import LogisticRegressionScratch
from src.evaluation.metrics import accuracy

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage4")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  EPM - Stage 4: Natural Language Processing & Self-Attention")
print("  From Classical TF-IDF to Dense Embeddings & Transformers")
print("=" * 72)


# =====================================================================
# 1. INDUSTRIAL MAINTENANCE DATASET GENERATION
# =====================================================================

print("\n" + "-" * 72)
print("  1. DOMAIN CORPUS: INDUSTRIAL EQUIPMENT MAINTENANCE LOGS")
print("-" * 72)

# Templates for 3 classes: 0 = Nominal, 1 = Warning, 2 = Critical Fault
nominal_templates = [
    "routine shift inspection completed operating temperatures normal",
    "oil lubrication levels checked and replenished nominal condition",
    "motor running smoothly baseline vibration measurements stable",
    "cooling water pump pressure within normal operating parameters",
    "scheduled maintenance completed no abnormalities found in gearbox",
    "bearing temperature at 45 degrees celsius well within safe limits",
    "system reboot completed successfully all sensors reporting green",
    "filter cleaned airflow nominal motor drawing standard electrical load",
]

warning_templates = [
    "minor rattling sound noticed in gearbox during morning start up",
    "coolant pressure slightly lower than recommended manufacturer baseline",
    "motor temperature gradually rising to upper advisory threshold",
    "mild mechanical vibration detected at high rotational rpm",
    "lubricant viscosity degraded scheduled for oil replacement next shift",
    "slight shaft misalignment detected during acoustic sensor scan",
    "periodic pressure fluctuation observed in primary hydraulic line",
    "unusual belt squeaking noise detected under heavy industrial load",
]

critical_templates = [
    "severe high frequency vibration detected motor bearing catastrophic failure",
    "cooling pump failure oil temperature exceeded 110 degrees emergency shutdown",
    "intense screeching noise and burning smell coming from main gearbox",
    "hydraulic pressure collapsed completely piston jammed in cylinder",
    "critical bearing seizure detected imminent catastrophic breakdown",
    "rapid thermal spike detected inside turbine housing manual trip required",
    "structural cracking sound heard excessive shaft oscillation above limit",
    "smoke emitted from motor stator winding immediate power disconnect",
]

corpus_texts = []
corpus_labels = []
rng = np.random.default_rng(42)

# Synthesize 360 maintenance log reports
for _ in range(120):
    corpus_texts.append(rng.choice(nominal_templates))
    corpus_labels.append(0)
    corpus_texts.append(rng.choice(warning_templates))
    corpus_labels.append(1)
    corpus_texts.append(rng.choice(critical_templates))
    corpus_labels.append(2)

indices = rng.permutation(len(corpus_texts))
corpus_texts = [corpus_texts[i] for i in indices]
corpus_labels = np.array([corpus_labels[i] for i in indices])

split_idx = int(0.8 * len(corpus_texts))
train_texts, test_texts = corpus_texts[:split_idx], corpus_texts[split_idx:]
train_labels, test_labels = corpus_labels[:split_idx], corpus_labels[split_idx:]

print(f"  Generated {len(corpus_texts)} maintenance log reports across 3 severity classes:")
print("    Class 0: Nominal Operation")
print("    Class 1: Warning / Minor Wear")
print("    Class 2: Critical / Imminent Failure")
print(f"  Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
print(f"  Sample Log: \"{train_texts[0]}\" -> Class {train_labels[0]}")


# =====================================================================
# 2. CLASSICAL NLP PIPELINE: TF-IDF + CLASSIFIER
# =====================================================================

print("\n" + "-" * 72)
print("  2. CLASSICAL NLP: TF-IDF VECTORIZER FROM SCRATCH")
print("-" * 72)

tfidf = TFIDFVectorizerScratch()
X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf = tfidf.transform(test_texts)

print(f"  Vocabulary size: {len(tfidf.get_feature_names())} terms")
print(f"  TF-IDF Matrix shape: {X_train_tfidf.shape}")

# Train Classical Logistic Regression (from Stage 2)
clf_tfidf = LogisticRegressionScratch(learning_rate=0.2, n_iterations=800)
start_t = time.perf_counter()
clf_tfidf.fit(X_train_tfidf, train_labels)
train_time_tfidf = time.perf_counter() - start_t

start_inf = time.perf_counter()
preds_tfidf = clf_tfidf.predict(X_test_tfidf)
infer_time_tfidf = (time.perf_counter() - start_inf) / len(test_texts) * 1e6
acc_tfidf = accuracy(test_labels, preds_tfidf)

print(f"  TF-IDF + Logistic Regression Test Accuracy: {acc_tfidf * 100:.2f}%")
print(f"  Training time: {train_time_tfidf:.4f}s | Latency: {infer_time_tfidf:.2f} µs/query")


# =====================================================================
# 3. DENSE EMBEDDINGS & BAG-OF-EMBEDDINGS
# =====================================================================

print("\n" + "-" * 72)
print("  3. DENSE EMBEDDINGS & BAG-OF-EMBEDDINGS NEURAL CLASSIFIER")
print("-" * 72)

vocab = Vocabulary(min_freq=1)
vocab.build_vocab(train_texts)
max_seq_len = 16

X_train_ids = np.array([vocab.encode(t, max_length=max_seq_len) for t in train_texts])
X_test_ids = np.array([vocab.encode(t, max_length=max_seq_len) for t in test_texts])

# PyTorch Bag-of-Embeddings
emb_model = PyTorchEmbeddingClassifier(
    vocab_size=len(vocab), embedding_dim=32, hidden_dim=32, num_classes=3, pad_idx=vocab.pad_id
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(emb_model.parameters(), lr=0.01)
train_dataset = TensorDataset(torch.from_numpy(X_train_ids).long(), torch.from_numpy(train_labels).long())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

start_t = time.perf_counter()
emb_model.train()
for epoch in range(25):
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(emb_model(bx), by)
        loss.backward()
        optimizer.step()
train_time_emb = time.perf_counter() - start_t

start_inf = time.perf_counter()
preds_emb = emb_model.predict(X_test_ids)
infer_time_emb = (time.perf_counter() - start_inf) / len(test_texts) * 1e6
acc_emb = accuracy(test_labels, preds_emb)

print(f"  Bag-of-Embeddings Test Accuracy:            {acc_emb * 100:.2f}%")
print(f"  Training time: {train_time_emb:.4f}s | Latency: {infer_time_emb:.2f} µs/query")


# =====================================================================
# 4. SCALED DOT-PRODUCT ATTENTION FROM SCRATCH & VISUALIZATION
# =====================================================================

print("\n" + "-" * 72)
print("  4. SCALED DOT-PRODUCT ATTENTION FROM SCRATCH & HEATMAP")
print("-" * 72)

# Sample sentence for attention demonstration
sample_sentence = "severe vibration detected in motor bearing"
sample_tokens = sample_sentence.split()
L = len(sample_tokens)
d_k = 8

# Initialize simulated learned representations
rng = np.random.default_rng(42)
# Q and K with high affinity between 'vibration' and 'bearing'
Q_demo = rng.normal(0, 1, size=(L, d_k))
K_demo = rng.normal(0, 1, size=(L, d_k))
V_demo = rng.normal(0, 1, size=(L, d_k))

# Strong dot-product correlation between token 1 ("vibration") and token 5 ("bearing")
Q_demo[1] = np.array([2.0, 1.5, -1.0, 0.5, 2.0, -0.5, 1.0, 1.5])
K_demo[5] = np.array([1.8, 1.4, -0.9, 0.4, 1.9, -0.4, 0.9, 1.4])

output_attn, attn_matrix = scaled_dot_product_attention_numpy(Q_demo, K_demo, V_demo)

print(f"  Input Tokens: {sample_tokens}")
print(f"  Scaled Dot-Product Formula: softmax( Q K^T / √d_k ) V")
print(f"  Key Dimension d_k: {d_k} (Scaling factor 1/√{d_k} = {1.0 / np.sqrt(d_k):.4f})")
print(f"  Attention Matrix Shape: {attn_matrix.shape} (Row sums = {np.sum(attn_matrix, axis=1)})")
print(f"\n  Attention from 'vibration' to all words:\n  " +
      ", ".join([f"{sample_tokens[j]}: {attn_matrix[1, j]:.3f}" for j in range(L)]))

# Plot Attention Heatmap
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
im = ax.imshow(attn_matrix, cmap="Blues", interpolation="nearest")

ax.set_xticks(range(L))
ax.set_yticks(range(L))
ax.set_xticklabels(sample_tokens, rotation=45, ha="right", fontsize=10, fontweight="bold")
ax.set_yticklabels(sample_tokens, fontsize=10, fontweight="bold")
ax.set_title("Self-Attention Alignment Matrix: Attention(Q, K, V)", fontweight="bold", fontsize=11)
ax.set_xlabel("Key Tokens (Information Sources)")
ax.set_ylabel("Query Tokens (Attending Words)")

for i in range(L):
    for j in range(L):
        text = ax.text(j, i, f"{attn_matrix[i, j]:.2f}",
                       ha="center", va="center", color="black" if attn_matrix[i, j] < 0.5 else "white", fontsize=9)

plt.colorbar(im, ax=ax, label="Attention Weight A_ij")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "attention_heatmap.png")
plt.close(fig)
print(f"\n  [Plot saved]: {PLOTS_DIR / 'attention_heatmap.png'}")


# =====================================================================
# 5. TRANSFORMER SEQUENCE CLASSIFIER
# =====================================================================

print("\n" + "-" * 72)
print("  5. COMPLETE TRANSFORMER SEQUENCE CLASSIFIER")
print("-" * 72)

transformer_model = TextTransformerClassifier(
    vocab_size=len(vocab),
    num_classes=3,
    d_model=32,
    num_heads=4,
    d_ff=64,
    max_seq_len=max_seq_len,
    pad_idx=vocab.pad_id,
    dropout=0.1,
)

opt_trans = optim.Adam(transformer_model.parameters(), lr=0.005)
start_t = time.perf_counter()
transformer_model.train()
for epoch in range(25):
    for bx, by in train_loader:
        opt_trans.zero_grad()
        logits, _ = transformer_model(bx)
        loss = criterion(logits, by)
        loss.backward()
        opt_trans.step()
train_time_trans = time.perf_counter() - start_t

start_inf = time.perf_counter()
preds_trans = transformer_model.predict(X_test_ids)
infer_time_trans = (time.perf_counter() - start_inf) / len(test_texts) * 1e6
acc_trans = accuracy(test_labels, preds_trans)

print(f"  Transformer Classifier Test Accuracy:       {acc_trans * 100:.2f}%")
print(f"  Training time: {train_time_trans:.4f}s | Latency: {infer_time_trans:.2f} µs/query")


# =====================================================================
# 6. PIPELINE BENCHMARK & COMPARISON
# =====================================================================

print("\n" + "-" * 72)
print("  6. NLP PIPELINE COMPARISON SUMMARY")
print("-" * 72)

models = ["TF-IDF + Logistic Reg", "Bag-of-Embeddings", "Transformer (Self-Attention)"]
accuracies = [acc_tfidf * 100, acc_emb * 100, acc_trans * 100]
latencies = [infer_time_tfidf, infer_time_emb, infer_time_trans]

print(f"  {'Model Architecture':<30} | {'Test Accuracy':<14} | {'Latency (µs)':<14}")
print(f"  {'-'*30}-|-{'-'*14}-|-{'-'*14}")
for m, a, l in zip(models, accuracies, latencies):
    print(f"  {m:<30} | {a:<13.2f}% | {l:<14.2f}")

# Plot Comparison Bar Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
colors = ["#3b82f6", "#10b981", "#8b5cf6"]

ax1.bar(models, accuracies, color=colors, edgecolor="black", width=0.5)
ax1.set_title("NLP Predictive Accuracy Comparison", fontweight="bold")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(80, 102)
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontweight="bold")
ax1.grid(True, axis="y", alpha=0.3)
ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=9)

ax2.bar(models, latencies, color=colors, edgecolor="black", width=0.5)
ax2.set_title("Inference Latency (Lower is better)", fontweight="bold")
ax2.set_ylabel("Latency (µs / sample)")
for i, v in enumerate(latencies):
    ax2.text(i, v + 1.0, f"{v:.1f}µs", ha="center", fontweight="bold")
ax2.grid(True, axis="y", alpha=0.3)
ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=9)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "tfidf_vs_embedding_vs_transformer.png")
plt.close(fig)
print(f"\n  [Plot saved]: {PLOTS_DIR / 'tfidf_vs_embedding_vs_transformer.png'}")


# =====================================================================
# STAGE 4 SUMMARY
# =====================================================================

print("\n" + "=" * 72)
print("  STAGE 4 COMPLETE: NLP, EMBEDDINGS & TRANSFORMERS")
print("=" * 72)
print("  Summary of Key Discoveries:")
print("  1. Progression of Representation:")
print("     TF-IDF captures surface lexical frequencies with sub-microsecond latency;")
print("     Embeddings capture semantic geometry; Self-Attention dynamically binds words.")
print("  2. Scaled Dot-Product Mechanics:")
print("     The 1/√d_k scaling factor mathematically counteracts dot-product variance growth,")
print("     preventing the softmax function from vanishing gradients.")
print("  3. The Foundation for Multimodal Learning:")
print("     The text representations developed here (Embeddings + Attention) provide")
print("     the exact textual encoder ready to be fused with sensor time-series in Stage 6!")
print("=" * 72)
print("  Ready for Stage 5: Time-Series Modeling (LSTM, GRU, Transformers)")
print("=" * 72)
