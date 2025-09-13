"""
Test script to verify SAE activation normalization is working correctly.
"""

import torch
import torch.nn.functional as F
from llada_mi.sae.model import LLaDASAE
import pytest


def test_activation_normalization():
    """Test that activation normalization reduces reconstruction loss."""

    torch.manual_seed(42)

    batch_size, seq_len, d_model = 8, 32, 4096

    activations = torch.randn(batch_size, seq_len, d_model) * 10.0  # Large scale
    activations[:, :16] *= 0.1  # Make first half smaller
    activations[:, 16:] *= 5.0  # Make second half larger

    assert -50 < activations.mean().item() < 50
    assert 0 < activations.std().item() < 200
    assert -300 < activations.min().item() < 300
    assert -300 < activations.max().item() < 300

    activations_flat = activations.view(-1, d_model)  # (batch_size * seq_len, d_model)

    sae_no_norm = LLaDASAE(
        d_model=d_model,
        d_sae=d_model * 4,  # 4x overcomplete
        k_sparse=64,
        normalize_activations=False,
    )

    recon_no_norm, sparse_acts_no_norm, pre_acts_no_norm = sae_no_norm(activations_flat)

    loss_dict_no_norm = sae_no_norm.compute_loss(
        activations_flat, recon_no_norm, sparse_acts_no_norm
    )

    sae_with_norm = LLaDASAE(
        d_model=d_model,
        d_sae=d_model * 4,  # 4x overcomplete
        k_sparse=64,
        normalize_activations=True,
        activation_norm_eps=1e-6,
    )

    sae_with_norm.encoder.weight.data = sae_no_norm.encoder.weight.data.clone()
    sae_with_norm.encoder.bias.data = sae_no_norm.encoder.bias.data.clone()
    sae_with_norm.decoder.weight.data = sae_no_norm.decoder.weight.data.clone()
    sae_with_norm.decoder.bias.data = sae_no_norm.decoder.bias.data.clone()

    recon_with_norm, sparse_acts_with_norm, pre_acts_with_norm = sae_with_norm(
        activations_flat
    )

    loss_dict_with_norm = sae_with_norm.compute_loss(
        activations_flat, recon_with_norm, sparse_acts_with_norm
    )

    recon_improvement = (
        loss_dict_no_norm["recon_loss"].item()
        - loss_dict_with_norm["recon_loss"].item()
    )
    assert recon_improvement > 0
    improvement_ratio = recon_improvement / loss_dict_no_norm["recon_loss"].item() * 100
    assert improvement_ratio > 0

    orig_mean = activations_flat.mean().item()
    recon_no_norm_mean = recon_no_norm.mean().item()
    recon_with_norm_mean = recon_with_norm.mean().item()
    assert abs(orig_mean - recon_no_norm_mean) < 10.0
    assert abs(orig_mean - recon_with_norm_mean) < 10.0

    orig_std = activations_flat.std().item()
    recon_no_norm_std = recon_no_norm.std().item()
    recon_with_norm_std = recon_with_norm.std().item()
    assert abs(orig_std - recon_no_norm_std) < 50.0
    assert abs(orig_std - recon_with_norm_std) < 50.0

    x_norm, x_mean, x_std = sae_with_norm._normalize_activations(activations_flat)
    x_denorm = sae_with_norm._denormalize_activations(x_norm, x_mean, x_std)

    assert x_denorm.shape == activations_flat.shape
    round_trip_error = F.mse_loss(activations_flat, x_denorm).item()
    assert round_trip_error < 1e-6
    assert abs(x_norm.mean().item()) < 0.1
    assert abs(x_norm.std().item() - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
