"""
EPM Multimodal Equipment Health Dataset
=======================================

Generates paired multimodal datasets combining:
  1. High-frequency multivariate sensor telemetry (Temperature, Vibration, Pressure, RPM, Load)
  2. Free-form textual maintenance logs and technician inspection reports
  3. Ground-truth equipment health severity label (Nominal, Warning, Critical Fault)
"""

from __future__ import annotations

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from src.nlp.preprocessing import Vocabulary
from src.timeseries.dataset import generate_multivariate_sensor_telemetry, create_sliding_windows


# Domain-specific text log templates correlated with physical sensor states
NOMINAL_LOGS = [
    "routine shift inspection completed operating temperatures normal",
    "oil lubrication levels checked and replenished nominal condition",
    "motor running smoothly baseline vibration measurements stable",
    "cooling water pump pressure within normal operating parameters",
    "scheduled maintenance completed no abnormalities found in gearbox",
    "bearing temperature at 45 degrees celsius well within safe limits",
    "system reboot completed successfully all sensors reporting green",
]

WARNING_LOGS = [
    "minor rattling sound noticed in gearbox during morning start up",
    "coolant pressure slightly lower than recommended manufacturer baseline",
    "motor temperature gradually rising to upper advisory threshold",
    "mild mechanical vibration detected at high rotational rpm",
    "lubricant viscosity degraded scheduled for oil replacement next shift",
    "slight shaft misalignment detected during acoustic sensor scan",
    "periodic pressure fluctuation observed in primary hydraulic line",
]

CRITICAL_LOGS = [
    "severe high frequency vibration detected motor bearing catastrophic failure",
    "cooling pump failure oil temperature exceeded 110 degrees emergency shutdown",
    "intense screeching noise and burning smell coming from main gearbox",
    "hydraulic pressure collapsed completely piston jammed in cylinder",
    "critical bearing seizure detected imminent catastrophic breakdown",
    "rapid thermal spike detected inside turbine housing manual trip required",
    "structural cracking sound heard excessive shaft oscillation above limit",
]


class MultimodalEquipmentDataset(Dataset):
    """
    PyTorch Dataset providing aligned (sensor_window, token_ids, label) triplets.

    Parameters
    ----------
    sensor_windows : np.ndarray of shape (N, W, D)
        Historical telemetry windows.
    text_token_ids : np.ndarray of shape (N, L)
        Tokenized maintenance text sequences.
    labels : np.ndarray of shape (N,)
        Integer class labels {0, 1, 2}.
    """

    def __init__(
        self,
        sensor_windows: np.ndarray,
        text_token_ids: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.sensor_windows = torch.from_numpy(sensor_windows).float()
        self.text_token_ids = torch.from_numpy(text_token_ids).long()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sensor_windows[idx], self.text_token_ids[idx], self.labels[idx]


def generate_paired_multimodal_dataset(
    n_samples: int = 600,
    window_size: int = 24,
    max_text_len: int = 16,
    normalize_sensors: bool = True,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Vocabulary]:
    """
    Generate synthetically aligned sensor telemetry windows and textual incident reports.

    Simulates:
      - 3 Balanced Health States:
        0 = Nominal Operating State
        1 = Incipient Mechanical Warning
        2 = Severe / Critical Breakdown
      - Realistic sensor noise and occasional ambiguous reports
    """
    rng = np.random.default_rng(random_seed)

    sensor_windows = np.zeros((n_samples, window_size, 5), dtype=np.float32)
    raw_texts = []
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Choose class: 0, 1, or 2
        cls = i % 3
        labels[i] = cls

        # 1. Generate correlated sensor window (24 steps, 5 channels)
        # Channels: [temperature, vibration, pressure, rpm, load]
        t_seq = np.linspace(0, 1, window_size)

        if cls == 0:  # Nominal
            temp = 45.0 + 3.0 * np.sin(2 * np.pi * t_seq) + rng.normal(0, 0.5, window_size)
            vib = 1.2 + rng.normal(0, 0.1, window_size)
            press = 50.0 + rng.normal(0, 1.0, window_size)
            rpm = 1800.0 + rng.normal(0, 20.0, window_size)
            load = 50.0 + 10.0 * np.sin(2 * np.pi * t_seq) + rng.normal(0, 2.0, window_size)
            text = rng.choice(NOMINAL_LOGS)

        elif cls == 1:  # Warning
            temp = 62.0 + 5.0 * t_seq + rng.normal(0, 1.0, window_size)
            vib = 2.8 + 0.5 * t_seq + rng.normal(0, 0.2, window_size)
            press = 44.0 - 3.0 * t_seq + rng.normal(0, 1.5, window_size)
            rpm = 1750.0 + rng.normal(0, 30.0, window_size)
            load = 65.0 + rng.normal(0, 5.0, window_size)
            text = rng.choice(WARNING_LOGS)

        else:  # Critical Fault
            temp = 85.0 + 15.0 * (t_seq ** 2) + rng.normal(0, 2.0, window_size)
            vib = 5.5 + 4.0 * (t_seq ** 2) + rng.normal(0, 0.4, window_size)
            press = 32.0 - 10.0 * t_seq + rng.normal(0, 2.5, window_size)
            rpm = 2300.0 + rng.normal(0, 50.0, window_size)
            load = 88.0 + rng.normal(0, 6.0, window_size)
            text = rng.choice(CRITICAL_LOGS)

        sensor_windows[i] = np.column_stack([temp, vib, press, rpm, load])
        raw_texts.append(text)

    # Standardize sensor channels (zero mean, unit variance)
    if normalize_sensors:
        ch_mean = np.mean(sensor_windows, axis=(0, 1), keepdims=True)
        ch_std = np.std(sensor_windows, axis=(0, 1), keepdims=True) + 1e-6
        sensor_windows = ((sensor_windows - ch_mean) / ch_std).astype(np.float32)

    # Shuffle paired data
    perm = rng.permutation(n_samples)
    sensor_windows = sensor_windows[perm]
    raw_texts = [raw_texts[p] for p in perm]
    labels = labels[perm]

    # Build vocabulary
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(raw_texts)

    # Encode texts into padded token IDs
    text_token_ids = np.array([vocab.encode(t, max_length=max_text_len) for t in raw_texts], dtype=np.int64)

    return sensor_windows, text_token_ids, labels, raw_texts, vocab
