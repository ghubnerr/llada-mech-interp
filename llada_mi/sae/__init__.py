"""
Sparse Autoencoder (SAE) module for LLaDA mechanistic interpretability.

This module provides:
- SAE model implementations (model.py)
- Training utilities and distributed training (train.py)
- Dataset utilities for The Pile and custom datasets (dataset.py)
"""

from .model import LLaDASAE
from .dataset import PileDataset, StaticTextDataset, CustomTextDataset, create_dataset
from .train import SAETrainingConfig, extract_llada_activations

__all__ = [
    "LLaDASAE",
    "PileDataset",
    "StaticTextDataset",
    "CustomTextDataset",
    "create_dataset",
    "SAETrainingConfig",
    "extract_llada_activations",
]
