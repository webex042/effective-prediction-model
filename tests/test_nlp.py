"""
Unit Tests for Stage 4: NLP, Embeddings, Self-Attention & Transformers
======================================================================

Verifies:
  1. Text cleaning, tokenization, and Vocabulary encoding/decoding
  2. TF-IDF from scratch (TF, IDF smoothing, L2 normalization)
  3. Embedding layer forward lookup and backward accumulation
  4. Scaled Dot-Product Self-Attention (NumPy & PyTorch parity, softmax row sum = 1, masking)
  5. Multi-Head Attention shapes and forward pass
  6. Positional Encoding and TextTransformerClassifier
"""

import numpy as np
import pytest
import torch

from src.nlp.preprocessing import clean_text, simple_tokenize, Vocabulary
from src.nlp.tfidf import TFIDFVectorizerScratch
from src.nlp.embeddings import EmbeddingLayerScratch, PyTorchEmbeddingClassifier
from src.transformers_.attention import (
    scaled_dot_product_attention_numpy,
    scaled_dot_product_attention_torch,
    MultiHeadAttention,
)
from src.transformers_.transformer_block import (
    PositionalEncoding,
    TransformerBlock,
    TextTransformerClassifier,
)


# =============================================================================
# 1. Preprocessing & Vocabulary Tests
# =============================================================================

class TestPreprocessing:
    def test_clean_text(self):
        raw = "  Machine ALARM! Vibration @ 87.5% - urgent!!  "
        cleaned = clean_text(raw)
        assert cleaned == "machine alarm vibration 87 5 urgent"

    def test_simple_tokenize(self):
        text = "Bearing temperature is 95C."
        tokens = simple_tokenize(text)
        assert tokens == ["bearing", "temperature", "is", "95c"]

    def test_vocabulary_encoding_and_padding(self):
        corpus = [
            "bearing vibration high",
            "temperature normal",
            "pump pressure low",
        ]
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(corpus)

        assert vocab.pad_id == 0
        assert vocab.unk_id == 1

        # Encode with padding
        encoded = vocab.encode("bearing vibration unknown", max_length=5)
        assert len(encoded) == 5
        assert encoded[0] == vocab.token2id["bearing"]
        assert encoded[1] == vocab.token2id["vibration"]
        assert encoded[2] == vocab.unk_id
        assert encoded[3] == vocab.pad_id
        assert encoded[4] == vocab.pad_id

        # Decode
        decoded = vocab.decode(encoded, skip_special=True)
        assert decoded == "bearing vibration"


# =============================================================================
# 2. TF-IDF Vectorizer Tests
# =============================================================================

class TestTFIDF:
    def test_tfidf_vectorizer_shapes_and_norm(self):
        corpus = [
            "machine vibration excessive",
            "routine inspection normal",
            "oil pump failure vibration",
        ]
        tfidf = TFIDFVectorizerScratch()
        X = tfidf.fit_transform(corpus)

        assert X.shape[0] == 3
        assert X.shape[1] == len(tfidf.get_feature_names())

        # Each row should be L2-normalized to 1.0
        row_norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(row_norms, 1.0, atol=1e-5)

    def test_idf_values_higher_for_rare_words(self):
        corpus = [
            "vibration in engine",
            "vibration in motor",
            "vibration in pump failure",
        ]
        tfidf = TFIDFVectorizerScratch()
        tfidf.fit(corpus)

        # "vibration" appears in all 3 docs; "failure" appears in only 1
        idx_vib = tfidf.vocabulary_["vibration"]
        idx_fail = tfidf.vocabulary_["failure"]

        assert tfidf.idf_diag_[idx_fail] > tfidf.idf_diag_[idx_vib]


# =============================================================================
# 3. Embedding Layer Tests
# =============================================================================

class TestEmbeddings:
    def test_embedding_scratch_forward_and_backward(self):
        vocab_size, embed_dim = 10, 8
        layer = EmbeddingLayerScratch(vocab_size, embed_dim)

        token_ids = np.array([[2, 4, 1], [0, 4, 3]])  # (2, 3)
        embeds = layer.forward(token_ids)

        assert embeds.shape == (2, 3, embed_dim)
        # Check lookup matches row
        np.testing.assert_array_equal(embeds[0, 1], layer.W[4])

        # Backward
        grad_out = np.ones_like(embeds)
        layer.backward(grad_out)

        # Token 4 appeared twice -> should receive gradient of 2.0
        np.testing.assert_allclose(layer.dW[4], 2.0)
        # Token 9 did not appear -> gradient 0.0
        np.testing.assert_allclose(layer.dW[9], 0.0)


# =============================================================================
# 4. Scaled Dot-Product Self-Attention Tests
# =============================================================================

class TestAttention:
    def test_scaled_dot_product_attention_numpy(self):
        B, L, d_k = 2, 4, 8
        rng = np.random.default_rng(42)
        Q = rng.normal(0, 1, size=(B, L, d_k))
        K = rng.normal(0, 1, size=(B, L, d_k))
        V = rng.normal(0, 1, size=(B, L, d_k))

        output, attn_weights = scaled_dot_product_attention_numpy(Q, K, V)

        assert output.shape == (B, L, d_k)
        assert attn_weights.shape == (B, L, L)

        # Probabilities must sum to 1.0 across keys (last dimension)
        row_sums = np.sum(attn_weights, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_attention_masking(self):
        B, L, d_k = 1, 3, 4
        Q = np.ones((B, L, d_k))
        K = np.ones((B, L, d_k))
        V = np.ones((B, L, d_k))

        # Mask token 2 out (e.g. padding token)
        mask = np.array([[True, True, False]])
        _, attn_weights = scaled_dot_product_attention_numpy(Q, K, V, mask=mask)

        # Weight for token 2 should be virtually 0.0
        assert attn_weights[0, 0, 2] < 1e-6
        # Remaining valid tokens should split attention equally
        np.testing.assert_allclose(attn_weights[0, 0, 0], 0.5, atol=1e-4)
        np.testing.assert_allclose(attn_weights[0, 0, 1], 0.5, atol=1e-4)

    def test_numpy_and_pytorch_attention_parity(self):
        B, L, d_k = 2, 3, 4
        rng = np.random.default_rng(42)
        Q_np = rng.normal(0, 1, (B, L, d_k)).astype(np.float32)
        K_np = rng.normal(0, 1, (B, L, d_k)).astype(np.float32)
        V_np = rng.normal(0, 1, (B, L, d_k)).astype(np.float32)

        out_np, weights_np = scaled_dot_product_attention_numpy(Q_np, K_np, V_np)

        Q_pt = torch.from_numpy(Q_np)
        K_pt = torch.from_numpy(K_np)
        V_pt = torch.from_numpy(V_np)

        out_pt, weights_pt = scaled_dot_product_attention_torch(Q_pt, K_pt, V_pt)

        np.testing.assert_allclose(out_np, out_pt.numpy(), atol=1e-5)
        np.testing.assert_allclose(weights_np, weights_pt.numpy(), atol=1e-5)


# =============================================================================
# 5. Multi-Head Attention & Transformer Tests
# =============================================================================

class TestTransformer:
    def test_multi_head_attention_forward(self):
        B, L, d_model, num_heads = 2, 5, 16, 4
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(B, L, d_model)

        out, attn_weights = mha(x, x, x)
        assert out.shape == (B, L, d_model)
        assert attn_weights.shape == (B, num_heads, L, L)

    def test_positional_encoding_shape(self):
        pe = PositionalEncoding(d_model=16, max_len=50)
        x = torch.zeros(2, 10, 16)
        out = pe(x)
        assert out.shape == (2, 10, 16)
        # Verify not all zeros (encodings were added)
        assert torch.abs(out).sum() > 0

    def test_text_transformer_classifier_forward(self):
        vocab_size = 20
        model = TextTransformerClassifier(
            vocab_size=vocab_size,
            num_classes=3,
            d_model=16,
            num_heads=2,
            d_ff=32,
            max_seq_len=10,
        )

        token_ids = torch.randint(0, vocab_size, (4, 8))
        logits, attn_weights = model(token_ids)

        assert logits.shape == (4, 3)
        assert attn_weights.shape == (4, 2, 8, 8)
