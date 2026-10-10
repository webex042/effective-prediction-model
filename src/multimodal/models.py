"""
EPM Multimodal Encoders & Fusion Architectures
==============================================

Implements unimodal encoders, multimodal fusion mechanisms, and classification
heads for aligning continuous sensor telemetry and discrete textual incident logs.

FUSION TAXONOMY:
  1. Concatenation Fusion (Feature-level early/mid concatenation)
  2. Gated Multimodal Fusion (GMU — dynamic adaptive channel-wise gating)
  3. Cross-Attention Fusion (Bidirectional cross-modal queries)
  4. Unimodal Ablations (Sensor-Only and Text-Only benchmarks)
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Modality Encoders
# =============================================================================

class SensorEncoder(nn.Module):
    """
    Encodes multivariate temporal sensor telemetry sequences into dense representations.

    Input shape:  (batch_size, window_size, sensor_dim) e.g. (B, 24, 5)
    Output:
      latent:    (batch_size, latent_dim) — pooled sequence embedding
      sequence:  (batch_size, window_size, latent_dim) — step-wise embeddings
    """

    def __init__(
        self,
        sensor_dim: int = 5,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional

        # 1D conv for local temporal feature smoothing
        self.conv1d = nn.Conv1d(
            in_channels=sensor_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )

        # Recurrent sequence processor
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Linear(gru_out_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, sensor_dim)

        Returns
        -------
        latent : torch.Tensor of shape (batch_size, latent_dim)
        sequence : torch.Tensor of shape (batch_size, seq_len, latent_dim)
        """
        # x: (B, W, D) -> Conv1d expects (B, D, W)
        conv_in = x.transpose(1, 2)
        conv_out = F.gelu(self.conv1d(conv_in)).transpose(1, 2)  # (B, W, H)

        gru_out, _ = self.gru(conv_out)  # (B, W, gru_out_dim)
        seq_emb = self.layer_norm(self.proj(gru_out))  # (B, W, latent_dim)

        # Temporal average pooling for summary latent vector
        latent = torch.mean(seq_emb, dim=1)  # (B, latent_dim)
        latent = self.dropout(latent)

        return latent, seq_emb


class TextEncoder(nn.Module):
    """
    Encodes discrete tokenized text sequences into dense representations.

    Input shape:  (batch_size, seq_len) e.g. (B, 16) token IDs
    Output:
      latent:    (batch_size, latent_dim) — pooled sentence embedding
      sequence:  (batch_size, seq_len, latent_dim) — token-wise embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int = 0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Linear(gru_out_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        token_ids : torch.Tensor of shape (batch_size, seq_len)

        Returns
        -------
        latent : torch.Tensor of shape (batch_size, latent_dim)
        sequence : torch.Tensor of shape (batch_size, seq_len, latent_dim)
        """
        # Embed tokens: (B, L, embed_dim)
        embeds = self.embedding(token_ids)
        embeds = self.dropout(embeds)

        gru_out, _ = self.gru(embeds)  # (B, L, gru_out_dim)
        seq_emb = self.layer_norm(self.proj(gru_out))  # (B, L, latent_dim)

        # Mask-aware average pooling (ignoring padding tokens where token_id == 0)
        mask = (token_ids != self.padding_idx).unsqueeze(-1).float()  # (B, L, 1)
        sum_emb = torch.sum(seq_emb * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        latent = sum_emb / denom  # (B, latent_dim)
        latent = self.dropout(latent)

        return latent, seq_emb


# =============================================================================
# 2. Fusion Mechanisms
# =============================================================================

class ConcatenationFusion(nn.Module):
    """
    Baseline Early/Mid Concatenation Fusion.

    Concatenates latent vectors [h_sensor || h_text] and projects through an MLP:
        h_fused = LayerNorm(W2 * GELU(W1 * [h_s || h_t] + b1) + b2)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        fused_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
        )

    def forward(
        self,
        h_sensor: torch.Tensor,
        h_text: torch.Tensor,
        seq_sensor: Optional[torch.Tensor] = None,
        seq_text: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        concat = torch.cat([h_sensor, h_text], dim=-1)
        fused = self.mlp(concat)
        return fused, {"fusion_type": "concat"}


class GatedMultimodalFusion(nn.Module):
    """
    Gated Multimodal Unit (GMU) Fusion.

    Adaptively learns dynamic gating weights between sensor telemetry and text reports:
        h_s_tilde = tanh(W_s * h_sensor + b_s)
        h_t_tilde = tanh(W_t * h_text + b_t)
        z = sigmoid(W_z * [h_sensor || h_text] + b_z)
        h_fused = z * h_s_tilde + (1 - z) * h_t_tilde

    Properties:
      - z in (0, 1)^d represents the gate value per latent dimension.
      - Mean gate z_bar in [0, 1] quantifies overall sensor reliance.
      - When sensor inputs are degraded or corrupted, the network dynamically
        shifts z -> 0 to rely on the qualitative text log.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        fused_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fused_dim = fused_dim

        # Candidate transformations
        self.transform_sensor = nn.Linear(latent_dim, fused_dim)
        self.transform_text = nn.Linear(latent_dim, fused_dim)

        # Modal gating network: produces channel-wise gate z in (0, 1)
        self.gate_linear = nn.Linear(latent_dim * 2, fused_dim)

        self.layer_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_sensor: torch.Tensor,
        h_text: torch.Tensor,
        seq_sensor: Optional[torch.Tensor] = None,
        seq_text: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Candidate activations
        s_cand = torch.tanh(self.transform_sensor(h_sensor))
        t_cand = torch.tanh(self.transform_text(h_text))

        # Dynamic multimodal gate
        concat_features = torch.cat([h_sensor, h_text], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat_features))  # (B, fused_dim)

        # Gated convex combination
        fused = gate * s_cand + (1.0 - gate) * t_cand
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        aux = {
            "fusion_type": "gated",
            "gate": gate,
            "sensor_weight": gate.mean(dim=-1, keepdim=True),  # (B, 1)
            "text_weight": 1.0 - gate.mean(dim=-1, keepdim=True),
        }
        return fused, aux


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Modal Attention Fusion.

    Performs cross-attention where:
      1. Sensor tokens query Text tokens: Q_sensor, K_text, V_text
      2. Text tokens query Sensor tokens: Q_text, K_sensor, V_sensor

    Captures fine-grained cross-modal alignments, e.g. correlating a specific
    thermal spike in time with operator notes about "bearing overheating".
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_heads: int = 4,
        fused_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        # Multi-Head Cross Attention: Sensor queries Text
        self.cross_attn_s2t = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Multi-Head Cross Attention: Text queries Sensor
        self.cross_attn_t2s = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_s = nn.LayerNorm(latent_dim)
        self.norm_t = nn.LayerNorm(latent_dim)

        # Final projection combining cross-attended pooled vectors
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
        )

    def forward(
        self,
        h_sensor: torch.Tensor,
        h_text: torch.Tensor,
        seq_sensor: Optional[torch.Tensor] = None,
        seq_text: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # If sequences are not provided, expand latents to (B, 1, D)
        if seq_sensor is None:
            seq_sensor = h_sensor.unsqueeze(1)
        if seq_text is None:
            seq_text = h_text.unsqueeze(1)

        # 1. Sensor queries Text
        attn_s2t_out, s2t_weights = self.cross_attn_s2t(
            query=seq_sensor,
            key=seq_text,
            value=seq_text,
        )
        s_enhanced = self.norm_s(seq_sensor + attn_s2t_out)
        s_pooled = s_enhanced.mean(dim=1)  # (B, latent_dim)

        # 2. Text queries Sensor
        attn_t2s_out, t2s_weights = self.cross_attn_t2s(
            query=seq_text,
            key=seq_sensor,
            value=seq_sensor,
        )
        t_enhanced = self.norm_t(seq_text + attn_t2s_out)
        t_pooled = t_enhanced.mean(dim=1)  # (B, latent_dim)

        # 3. Project joint cross-attended features
        fused = self.mlp(torch.cat([s_pooled, t_pooled], dim=-1))

        aux = {
            "fusion_type": "cross_attention",
            "s2t_weights": s2t_weights,
            "t2s_weights": t2s_weights,
        }
        return fused, aux


# =============================================================================
# 3. Unified Multimodal Classifier & Unimodal Baselines
# =============================================================================

class MultimodalClassifier(nn.Module):
    """
    Unified Multimodal Equipment Health Classifier.

    Integrates SensorEncoder + TextEncoder + customizable FusionModule + Classifier Head.

    Parameters
    ----------
    vocab_size : int
        Size of vocabulary for text logs.
    sensor_dim : int
        Number of sensor channels (default 5).
    latent_dim : int
        Internal representation dimensionality (default 64).
    num_classes : int
        Number of health diagnostic classes (default 3: Nominal, Warning, Critical).
    fusion_type : {'concat', 'gated', 'cross_attention'}
        Multimodal fusion strategy.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        sensor_dim: int = 5,
        latent_dim: int = 64,
        num_classes: int = 3,
        fusion_type: Literal["concat", "gated", "cross_attention"] = "gated",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.sensor_dim = sensor_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        # Encoders
        self.sensor_encoder = SensorEncoder(
            sensor_dim=sensor_dim,
            hidden_dim=latent_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=latent_dim,
            hidden_dim=latent_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Fusion
        if fusion_type == "concat":
            self.fusion = ConcatenationFusion(
                latent_dim=latent_dim, fused_dim=latent_dim, dropout=dropout
            )
        elif fusion_type == "gated":
            self.fusion = GatedMultimodalFusion(
                latent_dim=latent_dim, fused_dim=latent_dim, dropout=dropout
            )
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                latent_dim=latent_dim, num_heads=4, fused_dim=latent_dim, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from 'concat', 'gated', 'cross_attention'.")

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes),
        )

    def forward(
        self,
        sensor_x: torch.Tensor,
        text_x: torch.Tensor,
        missing_modality: Optional[Literal["sensor", "text"]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional simulated missing modality for robustness testing.

        Parameters
        ----------
        sensor_x : torch.Tensor of shape (B, W, D)
        text_x : torch.Tensor of shape (B, L)
        missing_modality : Optional[str]
            If 'sensor', zero out sensor latent representations.
            If 'text', zero out text latent representations.

        Returns
        -------
        logits : torch.Tensor of shape (B, num_classes)
        aux : Dict[str, Any] with embeddings and fusion diagnostics.
        """
        h_sensor, seq_sensor = self.sensor_encoder(sensor_x)
        h_text, seq_text = self.text_encoder(text_x)

        # Handle missing modality ablation
        if missing_modality == "sensor":
            h_sensor = torch.zeros_like(h_sensor)
            seq_sensor = torch.zeros_like(seq_sensor)
        elif missing_modality == "text":
            h_text = torch.zeros_like(h_text)
            seq_text = torch.zeros_like(seq_text)

        # Fusion
        fused, aux = self.fusion(h_sensor, h_text, seq_sensor, seq_text)

        # Logits
        logits = self.classifier(fused)

        aux["h_sensor"] = h_sensor
        aux["h_text"] = h_text
        aux["h_fused"] = fused

        return logits, aux


class SensorOnlyClassifier(nn.Module):
    """Ablation model relying purely on multivariate sensor telemetry."""

    def __init__(
        self,
        sensor_dim: int = 5,
        latent_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sensor_encoder = SensorEncoder(
            sensor_dim=sensor_dim,
            hidden_dim=latent_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes),
        )

    def forward(
        self, sensor_x: torch.Tensor, text_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h_sensor, seq_sensor = self.sensor_encoder(sensor_x)
        logits = self.classifier(h_sensor)
        return logits, {"h_sensor": h_sensor}


class TextOnlyClassifier(nn.Module):
    """Ablation model relying purely on technician textual logs."""

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=latent_dim,
            hidden_dim=latent_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes),
        )

    def forward(
        self, sensor_x: Optional[torch.Tensor] = None, text_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if text_x is None:
            raise ValueError("text_x is required for TextOnlyClassifier")
        h_text, seq_text = self.text_encoder(text_x)
        logits = self.classifier(h_text)
        return logits, {"h_text": h_text}
