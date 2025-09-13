"""
Comprehensive tests for LLaDA Sparse Autoencoder (SAE) model.

This test suite covers the LLaDA SAE implementation specifically designed for
diffusion language models. Tests include initialization, forward pass, loss computation,
sparsity constraints, weight management, and integration with diffusion model components.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llada_mi.sae.model import LLaDASAE


@pytest.fixture
def llada_sae_config(sample_inference_tensors):
    """Configuration for SAE that matches llada model dimensions."""
    return {
        "d_model": sample_inference_tensors["hidden_dim"],
        "d_sae": sample_inference_tensors["hidden_dim"] * 4,
        "k_sparse": 64,
    }


@pytest.fixture
def llada_sample_sae(device, llada_sae_config):
    """Create a sample SAE with llada model dimensions."""
    return LLaDASAE(**llada_sae_config).to(device)


class TestLLaDASAEInitialization:
    """Test SAE model initialization and configuration."""

    def test_basic_initialization(self, device):
        """Test basic SAE initialization with default parameters."""
        d_model = 4096  # Use llada model hidden dimension
        d_sae = 16384  # 4x expansion ratio
        k_sparse = 64

        sae = LLaDASAE(d_model=d_model, d_sae=d_sae, k_sparse=k_sparse).to(device)

        assert sae.d_model == d_model
        assert sae.d_sae == d_sae
        assert sae.k_sparse == k_sparse
        assert not sae.tie_weights
        assert sae.normalize_decoder
        assert sae.l2_coefficient == 1e-6
        assert isinstance(sae.encoder, nn.Linear)
        assert sae.encoder.in_features == d_model
        assert sae.encoder.out_features == d_sae
        assert sae.encoder.bias is not None

        assert isinstance(sae.decoder, nn.Linear)
        assert sae.decoder.in_features == d_sae
        assert sae.decoder.out_features == d_model
        assert sae.decoder.bias is not None

    def test_tied_weights_initialization(self, device):
        """Test SAE initialization with tied weights."""
        d_model = 4096
        d_sae = 16384

        sae = LLaDASAE(
            d_model=d_model, d_sae=d_sae, tie_weights=True, bias_decoder=True
        ).to(device)

        assert sae.tie_weights
        assert hasattr(sae, "encoder")
        assert not hasattr(sae, "decoder")
        assert hasattr(sae, "decoder_bias")
        assert sae.decoder_bias is not None
        assert sae.decoder_bias.shape == (d_model,)

    def test_no_decoder_bias_initialization(self, device):
        """Test SAE initialization without decoder bias."""
        sae = LLaDASAE(d_model=4096, d_sae=16384, bias_decoder=False).to(device)

        assert sae.decoder.bias is None

    def test_custom_parameters(self, device):
        """Test SAE initialization with custom parameters."""
        d_model = 4096
        d_sae = 12288
        k_sparse = 128
        l2_coeff = 1e-4

        sae = LLaDASAE(
            d_model=d_model,
            d_sae=d_sae,
            k_sparse=k_sparse,
            tie_weights=True,
            normalize_decoder=False,
            bias_decoder=False,
            l2_coefficient=l2_coeff,
        ).to(device)

        assert sae.d_model == d_model
        assert sae.d_sae == d_sae
        assert sae.k_sparse == k_sparse
        assert sae.tie_weights
        assert not sae.normalize_decoder
        assert sae.l2_coefficient == l2_coeff
        assert sae.decoder_bias is None

    def test_weight_initialization_values(self, device, random_seed):
        """Test that weights are properly initialized."""
        sae = LLaDASAE(d_model=4096, d_sae=16384).to(device)

        assert not torch.allclose(
            sae.encoder.weight, torch.zeros_like(sae.encoder.weight)
        )
        assert torch.allclose(sae.encoder.bias, torch.zeros_like(sae.encoder.bias))

        assert not torch.allclose(
            sae.decoder.weight, torch.zeros_like(sae.decoder.weight)
        )
        assert torch.allclose(sae.decoder.bias, torch.zeros_like(sae.decoder.bias))

        assert torch.isfinite(sae.encoder.weight).all()
        assert torch.isfinite(sae.decoder.weight).all()


class TestLLaDASAEForwardPass:
    """Test SAE forward pass functionality."""

    @pytest.fixture
    def sample_sae(self, device):
        """Create a sample SAE for testing with llada model dimensions."""
        return LLaDASAE(d_model=4096, d_sae=16384, k_sparse=64).to(device)

    @pytest.fixture
    def sample_input(self, device, sample_inference_tensors):
        """Create sample input tensor for testing using llada model dimensions."""
        batch_size = 1  # Match inference tensor batch size
        seq_length = sample_inference_tensors["total_length"]
        d_model = sample_inference_tensors["hidden_dim"]
        return torch.randn(batch_size, seq_length, d_model, device=device)

    def test_forward_pass_shapes(
        self, sample_sae, sample_input, tensor_assertion_helpers
    ):
        """Test that forward pass produces correct output shapes."""
        reconstruction, sparse_acts, pre_acts = sample_sae(sample_input)

        batch_size, seq_length, d_model = sample_input.shape
        d_sae = sample_sae.d_sae

        tensor_assertion_helpers["assert_shape"](
            reconstruction, (batch_size, seq_length, d_model)
        )
        tensor_assertion_helpers["assert_shape"](
            sparse_acts, (batch_size, seq_length, d_sae)
        )
        tensor_assertion_helpers["assert_shape"](
            pre_acts, (batch_size, seq_length, d_sae)
        )

        tensor_assertion_helpers["assert_device"](reconstruction, sample_input.device)
        tensor_assertion_helpers["assert_device"](sparse_acts, sample_input.device)
        tensor_assertion_helpers["assert_device"](pre_acts, sample_input.device)
        tensor_assertion_helpers["assert_finite"](reconstruction)
        tensor_assertion_helpers["assert_finite"](sparse_acts)
        tensor_assertion_helpers["assert_finite"](pre_acts)

    def test_k_sparse_constraint(self, sample_sae, sample_input):
        """Test that k-sparse constraint is properly enforced."""
        _, sparse_acts, _ = sample_sae(sample_input)

        non_zero_counts = (sparse_acts != 0).sum(dim=-1)  # (B, T)

        expected_count = sample_sae.k_sparse
        assert (non_zero_counts == expected_count).all(), (
            f"Expected {expected_count} non-zero elements per position, got {non_zero_counts}"
        )

    def test_top_k_selection(self, device, llada_sample_sae):
        """Test that top-k selection works correctly."""
        batch_size, seq_length, d_model = 1, 1, llada_sample_sae.d_model
        x = torch.randn(batch_size, seq_length, d_model, device=device)

        sample_sae = llada_sample_sae  # Use llada dimensions

        pre_acts = sample_sae.encoder(x)

        sparse_acts = sample_sae._apply_k_sparse(pre_acts)

        _, top_k_indices = torch.topk(pre_acts[0, 0], sample_sae.k_sparse)

        non_zero_indices = torch.nonzero(sparse_acts[0, 0]).squeeze(-1)

        top_k_indices_sorted = torch.sort(top_k_indices)[0]
        non_zero_indices_sorted = torch.sort(non_zero_indices)[0]

        assert torch.equal(top_k_indices_sorted, non_zero_indices_sorted)

    def test_tied_weights_forward(self, device, sample_input):
        """Test forward pass with tied weights."""
        sae = LLaDASAE(
            d_model=4096,
            d_sae=16384,
            k_sparse=64,
            tie_weights=True,
            normalize_decoder=False,
        ).to(device)

        sae.eval()
        reconstruction, sparse_acts, pre_acts = sae(sample_input)

        assert reconstruction.shape == sample_input.shape
        assert sparse_acts.shape == (
            sample_input.shape[0],
            sample_input.shape[1],
            sae.d_sae,
        )

        assert not hasattr(sae, "decoder")
        assert hasattr(sae, "decoder_bias")

        expected_reconstruction = F.linear(
            sparse_acts, sae.encoder.weight.t(), sae.decoder_bias
        )

        assert torch.allclose(reconstruction, expected_reconstruction, rtol=1e-4)

    def test_decoder_normalization(self, device, sample_input):
        """Test that decoder weight normalization works during training."""
        sae = LLaDASAE(
            d_model=4096, d_sae=16384, k_sparse=64, normalize_decoder=True
        ).to(device)

        sae.train()

        reconstruction, _, _ = sae(sample_input)

        weight_norms = torch.norm(sae.decoder.weight, dim=1)
        expected_norms = torch.ones_like(weight_norms)

        assert torch.allclose(weight_norms, expected_norms, atol=1e-6)

    def test_eval_mode_no_normalization(self, device, sample_input):
        """Test that decoder weights are not normalized in eval mode."""
        sae = LLaDASAE(
            d_model=4096, d_sae=16384, k_sparse=64, normalize_decoder=True
        ).to(device)

        sae.eval()

        original_weights = sae.decoder.weight.clone()

        reconstruction, _, _ = sae(sample_input)

        assert torch.equal(sae.decoder.weight, original_weights)


class TestLLaDASAELossComputation:
    """Test SAE loss computation and metrics."""

    @pytest.fixture
    def sample_sae(self, device):
        """Create a sample SAE for testing with llada model dimensions."""
        return LLaDASAE(d_model=4096, d_sae=16384, k_sparse=64, l2_coefficient=1e-4).to(
            device
        )

    @pytest.fixture
    def sample_data(self, device, sample_inference_tensors):
        """Create sample data for loss testing using llada model dimensions."""
        batch_size = 1  # Match inference tensor batch size
        seq_length = sample_inference_tensors["total_length"]
        d_model = sample_inference_tensors["hidden_dim"]
        x = torch.randn(batch_size, seq_length, d_model, device=device)
        return x

    def test_loss_computation_shapes(self, sample_sae, sample_data):
        """Test that loss computation returns correct shapes and types."""
        reconstruction, sparse_acts, pre_acts = sample_sae(sample_data)
        loss_dict = sample_sae.compute_loss(sample_data, reconstruction, sparse_acts)

        required_keys = [
            "total_loss",
            "recon_loss",
            "weight_decay_loss",
            "sparsity_ratio",
            "l0_norm",
            "max_recon_error",
            "mean_recon_error",
        ]

        for key in required_keys:
            assert key in loss_dict, f"Missing key: {key}"

        scalar_keys = ["total_loss", "recon_loss", "weight_decay_loss"]
        for key in scalar_keys:
            assert loss_dict[key].dim() == 0, f"{key} should be scalar"
            assert torch.isfinite(loss_dict[key]), f"{key} should be finite"

    def test_reconstruction_loss(self, sample_sae, sample_data):
        """Test reconstruction loss calculation."""
        reconstruction, sparse_acts, _ = sample_sae(sample_data)
        loss_dict = sample_sae.compute_loss(sample_data, reconstruction, sparse_acts)

        expected_recon_loss = F.mse_loss(reconstruction, sample_data)

        assert torch.allclose(loss_dict["recon_loss"], expected_recon_loss, rtol=1e-5)
        assert loss_dict["recon_loss"] >= 0, (
            "Reconstruction loss should be non-negative"
        )

    def test_weight_decay_loss(self, sample_sae, sample_data):
        """Test weight decay loss calculation."""
        reconstruction, sparse_acts, _ = sample_sae(sample_data)
        loss_dict = sample_sae.compute_loss(sample_data, reconstruction, sparse_acts)

        expected_weight_decay = (
            torch.norm(sample_sae.encoder.weight) ** 2
            + torch.norm(sample_sae.decoder.weight) ** 2
        )

        assert torch.allclose(
            loss_dict["weight_decay_loss"], expected_weight_decay, rtol=1e-5
        )
        assert loss_dict["weight_decay_loss"] >= 0, (
            "Weight decay loss should be non-negative"
        )

    def test_tied_weights_loss(self, device, sample_data):
        """Test loss computation with tied weights."""
        sae = LLaDASAE(
            d_model=4096,
            d_sae=16384,
            k_sparse=64,
            tie_weights=True,
            l2_coefficient=1e-4,
        ).to(device)

        reconstruction, sparse_acts, _ = sae(sample_data)
        loss_dict = sae.compute_loss(sample_data, reconstruction, sparse_acts)

        expected_weight_decay = torch.norm(sae.encoder.weight) ** 2

        assert torch.allclose(
            loss_dict["weight_decay_loss"], expected_weight_decay, rtol=1e-5
        )

    def test_total_loss_composition(self, sample_sae, sample_data):
        """Test that total loss is correctly composed of components."""
        reconstruction, sparse_acts, _ = sample_sae(sample_data)
        loss_dict = sample_sae.compute_loss(sample_data, reconstruction, sparse_acts)

        expected_total = (
            loss_dict["recon_loss"]
            + sample_sae.l2_coefficient * loss_dict["weight_decay_loss"]
        )

        assert torch.allclose(loss_dict["total_loss"], expected_total, rtol=1e-5)

    def test_sparsity_metrics(self, sample_sae, sample_data):
        """Test sparsity-related metrics."""
        reconstruction, sparse_acts, _ = sample_sae(sample_data)
        loss_dict = sample_sae.compute_loss(sample_data, reconstruction, sparse_acts)

        expected_sparsity_ratio = (sparse_acts != 0).float().mean()
        assert torch.allclose(
            loss_dict["sparsity_ratio"], expected_sparsity_ratio, rtol=1e-5
        )

        expected_l0_norm = (sparse_acts != 0).sum(dim=-1).float().mean()
        assert torch.allclose(loss_dict["l0_norm"], expected_l0_norm, rtol=1e-5)

        assert torch.allclose(
            loss_dict["l0_norm"],
            torch.tensor(sample_sae.k_sparse, dtype=torch.float),
            rtol=1e-5,
        )

    def test_reconstruction_error_metrics(self, sample_sae, sample_data):
        """Test reconstruction error metrics."""
        reconstruction, sparse_acts, _ = sample_sae(sample_data)
        loss_dict = sample_sae.compute_loss(sample_data, reconstruction, sparse_acts)

        recon_error = torch.norm(sample_data - reconstruction, dim=-1)  # (B, T)
        expected_max_error = recon_error.max()
        expected_mean_error = recon_error.mean()

        assert torch.allclose(
            loss_dict["max_recon_error"], expected_max_error, rtol=1e-5
        )
        assert torch.allclose(
            loss_dict["mean_recon_error"], expected_mean_error, rtol=1e-5
        )

        assert loss_dict["max_recon_error"] >= 0
        assert loss_dict["mean_recon_error"] >= 0


class TestLLaDASAEFeatureAnalysis:
    """Test SAE feature activation analysis methods."""

    @pytest.fixture
    def sample_sae(self, device):
        """Create a sample SAE for testing with llada model dimensions."""
        return LLaDASAE(d_model=4096, d_sae=16384, k_sparse=64).to(device)

    @pytest.fixture
    def sample_input(self, device, sample_inference_tensors):
        """Create sample input for feature analysis using llada model dimensions."""
        batch_size = 1  # Match inference tensor batch size
        seq_length = sample_inference_tensors["total_length"]
        d_model = sample_inference_tensors["hidden_dim"]
        return torch.randn(batch_size, seq_length, d_model, device=device)

    def test_get_feature_activations_basic(
        self, sample_sae, sample_input, tensor_assertion_helpers
    ):
        """Test basic feature activation retrieval."""
        sparse_acts, active_indices = sample_sae.get_feature_activations(
            sample_input, return_indices=False
        )

        batch_size, seq_length = sample_input.shape[:2]
        d_sae = sample_sae.d_sae

        tensor_assertion_helpers["assert_shape"](
            sparse_acts, (batch_size, seq_length, d_sae)
        )
        assert active_indices is None

        non_zero_counts = (sparse_acts != 0).sum(dim=-1)
        assert (non_zero_counts == sample_sae.k_sparse).all()

    def test_get_feature_activations_with_indices(
        self, sample_sae, sample_input, tensor_assertion_helpers
    ):
        """Test feature activation retrieval with indices."""
        sparse_acts, active_indices = sample_sae.get_feature_activations(
            sample_input, return_indices=True
        )

        batch_size, seq_length = sample_input.shape[:2]
        k_sparse = sample_sae.k_sparse

        tensor_assertion_helpers["assert_shape"](
            active_indices, (batch_size, seq_length, k_sparse)
        )

        for b in range(batch_size):
            for t in range(seq_length):
                indices = active_indices[b, t]
                acts = sparse_acts[b, t]

                assert (acts[indices] != 0).all()

                mask = torch.ones(acts.shape[0], dtype=torch.bool, device=acts.device)
                mask[indices] = False
                assert (acts[mask] == 0).all()

    def test_no_grad_context(self, sample_sae, sample_input):
        """Test that feature analysis doesn't require gradients."""
        sample_input.requires_grad_(True)

        sparse_acts, active_indices = sample_sae.get_feature_activations(
            sample_input, return_indices=True
        )

        assert not sparse_acts.requires_grad
        assert not active_indices.requires_grad


class TestLLaDASAEDiffusionIntegration:
    """Test SAE integration with diffusion model components."""

    def test_diffusion_hidden_states_processing(
        self, device, tensor_assertion_helpers, sample_inference_tensors
    ):
        """Test SAE processing of diffusion model hidden states using llada model shapes."""
        steps = sample_inference_tensors["steps"]
        layers = sample_inference_tensors["num_layers"]
        batch_size = 1  # Match inference tensor batch size
        seq_len = sample_inference_tensors["total_length"]
        hidden_dim = sample_inference_tensors["hidden_dim"]

        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        all_hidden_states = torch.randn(
            steps, layers, batch_size, seq_len, hidden_dim, device=device
        )

        for layer_idx in range(layers):
            layer_hidden_states = all_hidden_states[:, layer_idx]

            layer_reconstructions = []
            layer_sparse_acts = []

            for step_idx in range(steps):
                step_hidden = layer_hidden_states[step_idx]

                reconstruction, sparse_acts, _ = sae(step_hidden)
                layer_reconstructions.append(reconstruction)
                layer_sparse_acts.append(sparse_acts)

            layer_reconstructions = torch.stack(layer_reconstructions)
            layer_sparse_acts = torch.stack(layer_sparse_acts)

            tensor_assertion_helpers["assert_shape"](
                layer_reconstructions, (steps, batch_size, seq_len, hidden_dim)
            )
            tensor_assertion_helpers["assert_shape"](
                layer_sparse_acts, (steps, batch_size, seq_len, sae.d_sae)
            )

    def test_step_agnostic_processing(
        self, device, random_seed, sample_inference_tensors
    ):
        """Test that SAE treats all diffusion steps equally (step-agnostic)."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(
            d_model=hidden_dim,
            d_sae=hidden_dim * 4,
            k_sparse=64,
            normalize_decoder=False,  # Disable normalization for deterministic behavior
        ).to(device)

        sae.eval()  # Use eval mode to prevent normalization

        batch_size = 1  # Match inference tensor batch size
        seq_len = sample_inference_tensors["total_length"]

        hidden_state = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        results = []
        for _ in range(5):
            reconstruction, sparse_acts, pre_acts = sae(hidden_state)
            results.append((reconstruction, sparse_acts, pre_acts))

        for i in range(1, len(results)):
            assert torch.allclose(results[0][0], results[i][0], rtol=1e-5)
            assert torch.allclose(results[0][1], results[i][1], rtol=1e-5)
            assert torch.allclose(results[0][2], results[i][2], rtol=1e-5)

    def test_batch_processing_consistency(
        self, device, random_seed, sample_inference_tensors
    ):
        """Test that batch processing gives same results as individual processing."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(
            d_model=hidden_dim,
            d_sae=hidden_dim * 4,
            k_sparse=64,
            normalize_decoder=False,
        ).to(device)

        sae.eval()  # Use eval mode to prevent normalization

        batch_size = 4  # Test with multiple samples
        seq_len = sample_inference_tensors["total_length"]
        batch_input = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        batch_reconstruction, batch_sparse_acts, batch_pre_acts = sae(batch_input)

        individual_results = []
        for i in range(batch_size):
            single_input = batch_input[i : i + 1]
            reconstruction, sparse_acts, pre_acts = sae(single_input)
            individual_results.append((reconstruction, sparse_acts, pre_acts))

        for i in range(batch_size):
            batch_recon_slice = batch_reconstruction[i : i + 1]
            indiv_recon = individual_results[i][0]
            batch_sparse_slice = batch_sparse_acts[i : i + 1]
            indiv_sparse = individual_results[i][1]
            batch_pre_slice = batch_pre_acts[i : i + 1]
            indiv_pre = individual_results[i][2]

            assert batch_recon_slice.shape == indiv_recon.shape
            assert batch_sparse_slice.shape == indiv_sparse.shape
            assert batch_pre_slice.shape == indiv_pre.shape

            assert torch.allclose(
                batch_recon_slice, indiv_recon, rtol=1e-3, atol=1e-5
            ), f"Reconstruction mismatch at sample {i}"
            assert torch.allclose(
                batch_sparse_slice, indiv_sparse, rtol=1e-3, atol=1e-5
            ), f"Sparse acts mismatch at sample {i}"
            assert torch.allclose(batch_pre_slice, indiv_pre, rtol=1e-3, atol=1e-5), (
                f"Pre-acts mismatch at sample {i}"
            )


class TestLLaDASAEEdgeCases:
    """Test SAE behavior in edge cases and error conditions."""

    def test_zero_input(self, device, sample_inference_tensors):
        """Test SAE behavior with zero input."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        with torch.no_grad():
            sae.encoder.bias.data.normal_(0, 0.1)

        batch_size = 1  # Match inference tensor batch size
        seq_len = sample_inference_tensors["total_length"]
        zero_input = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
        reconstruction, sparse_acts, pre_acts = sae(zero_input)

        non_zero_counts = (sparse_acts != 0).sum(dim=-1)

        assert (non_zero_counts == sae.k_sparse).all(), (
            f"Expected {sae.k_sparse} non-zero elements, got {non_zero_counts}"
        )

        assert torch.isfinite(reconstruction).all()
        assert torch.isfinite(sparse_acts).all()
        assert torch.isfinite(pre_acts).all()

        for b in range(zero_input.shape[0]):
            for t in range(zero_input.shape[1]):
                kth_largest = torch.topk(pre_acts[b, t], sae.k_sparse)[0][-1]
                non_zero_mask = sparse_acts[b, t] != 0
                assert (sparse_acts[b, t][non_zero_mask] >= kth_largest - 1e-6).all()

    def test_single_token_sequence(self, device, sample_inference_tensors):
        """Test SAE with single token sequences."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        d_sae = hidden_dim * 4
        sae = LLaDASAE(d_model=hidden_dim, d_sae=d_sae, k_sparse=64).to(device)

        single_token_input = torch.randn(1, 1, hidden_dim, device=device)
        reconstruction, sparse_acts, pre_acts = sae(single_token_input)

        assert reconstruction.shape == (1, 1, hidden_dim)
        assert sparse_acts.shape == (1, 1, d_sae)
        assert pre_acts.shape == (1, 1, d_sae)

        assert (sparse_acts != 0).sum() == sae.k_sparse

    def test_large_k_sparse(self, device, sample_inference_tensors):
        """Test SAE behavior when k_sparse is large relative to d_sae."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        d_sae = 100
        k_sparse = 90  # 90% of features active

        sae = LLaDASAE(d_model=hidden_dim, d_sae=d_sae, k_sparse=k_sparse).to(device)

        seq_len = 4  # Use small sequence length for this test
        input_tensor = torch.randn(1, seq_len, hidden_dim, device=device)
        reconstruction, sparse_acts, pre_acts = sae(input_tensor)

        non_zero_counts = (sparse_acts != 0).sum(dim=-1)
        assert (non_zero_counts == k_sparse).all()

    def test_k_sparse_equals_d_sae(self, device, sample_inference_tensors):
        """Test SAE when k_sparse equals d_sae (no sparsity)."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        d_sae = 64
        k_sparse = d_sae  # No sparsity constraint

        sae = LLaDASAE(d_model=hidden_dim, d_sae=d_sae, k_sparse=k_sparse).to(device)

        seq_len = 4  # Use small sequence length for this test
        input_tensor = torch.randn(1, seq_len, hidden_dim, device=device)
        reconstruction, sparse_acts, pre_acts = sae(input_tensor)

        assert torch.equal(sparse_acts, pre_acts)
        assert (sparse_acts != 0).all()

    def test_gradient_flow(self, device, sample_inference_tensors):
        """Test that gradients flow properly through the SAE."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        seq_len = 4  # Use small sequence length for this test
        input_tensor = torch.randn(
            1, seq_len, hidden_dim, device=device, requires_grad=True
        )
        reconstruction, sparse_acts, pre_acts = sae(input_tensor)

        loss = reconstruction.sum()
        loss.backward()

        assert sae.encoder.weight.grad is not None
        assert sae.encoder.bias.grad is not None
        assert sae.decoder.weight.grad is not None
        if sae.decoder.bias is not None:
            assert sae.decoder.bias.grad is not None

        assert input_tensor.grad is not None


class TestLLaDASAETraining:
    """Test SAE training behavior and optimization."""

    def test_training_mode_effects(self, device, sample_inference_tensors):
        """Test that training mode affects decoder normalization."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(
            d_model=hidden_dim,
            d_sae=hidden_dim * 4,
            k_sparse=64,
            normalize_decoder=True,
        ).to(device)

        seq_len = 4  # Use small sequence length for this test
        input_tensor = torch.randn(1, seq_len, hidden_dim, device=device)

        sae.train()
        reconstruction_train, _, _ = sae(input_tensor)

        weight_norms = torch.norm(sae.decoder.weight, dim=1)
        assert torch.allclose(weight_norms, torch.ones_like(weight_norms), atol=1e-6)

        sae.eval()
        original_weights = sae.decoder.weight.clone()
        reconstruction_eval, _, _ = sae(input_tensor)

        assert torch.equal(sae.decoder.weight, original_weights)

    def test_loss_backward_compatibility(self, device, sample_inference_tensors):
        """Test that loss computation works with autograd."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        batch_size = 1  # Match inference tensor batch size
        seq_len = sample_inference_tensors["total_length"]
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        reconstruction, sparse_acts, pre_acts = sae(input_tensor)

        loss_dict = sae.compute_loss(input_tensor, reconstruction, sparse_acts)
        total_loss = loss_dict["total_loss"]

        total_loss.backward()

        assert sae.encoder.weight.grad is not None
        assert sae.decoder.weight.grad is not None

    def test_parameter_count(self, device, sample_inference_tensors):
        """Test parameter counting for different configurations."""
        d_model = sample_inference_tensors["hidden_dim"]
        d_sae = d_model * 4

        sae1 = LLaDASAE(
            d_model=d_model, d_sae=d_sae, tie_weights=False, bias_decoder=True
        ).to(device)
        expected_params1 = d_model * d_sae + d_sae + d_sae * d_model + d_model
        actual_params1 = sum(p.numel() for p in sae1.parameters())
        assert actual_params1 == expected_params1

        sae2 = LLaDASAE(
            d_model=d_model, d_sae=d_sae, tie_weights=True, bias_decoder=True
        ).to(device)
        expected_params2 = (
            d_model * d_sae  # encoder weight (shared)
            + d_sae
            + d_model  # decoder bias
        )
        actual_params2 = sum(p.numel() for p in sae2.parameters())
        assert actual_params2 == expected_params2

        sae3 = LLaDASAE(
            d_model=d_model, d_sae=d_sae, tie_weights=True, bias_decoder=False
        ).to(device)
        expected_params3 = d_model * d_sae + d_sae
        actual_params3 = sum(p.numel() for p in sae3.parameters())
        assert actual_params3 == expected_params3


class TestLLaDASAEFlattenedFormat:
    """Test SAE with flattened (total_tokens, hidden_size) input format."""

    def test_flattened_input_forward_pass(self, device, sample_inference_tensors):
        """Test forward pass with flattened 2D input."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        # Create flattened input: (total_tokens, hidden_dim)
        total_tokens = 20  # e.g., 4 sequences of 5 tokens each
        flattened_input = torch.randn(total_tokens, hidden_dim, device=device)

        reconstruction, sparse_acts, pre_acts = sae(flattened_input)

        # Check output shapes
        assert reconstruction.shape == (total_tokens, hidden_dim)
        assert sparse_acts.shape == (total_tokens, sae.d_sae)
        assert pre_acts.shape == (total_tokens, sae.d_sae)

        # Check sparsity constraint
        non_zero_counts = (sparse_acts != 0).sum(dim=-1)
        assert (non_zero_counts == sae.k_sparse).all()

    def test_flatten_unflatten_methods(self, device, sample_inference_tensors):
        """Test static methods for flattening and unflattening tensors."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        batch_size, seq_length = 2, 8

        # Create 3D tensor
        original_tensor = torch.randn(batch_size, seq_length, hidden_dim, device=device)

        # Test flattening
        flattened, original_shape = LLaDASAE.flatten_sequence_tensor(original_tensor)
        assert flattened.shape == (batch_size * seq_length, hidden_dim)
        assert original_shape == (batch_size, seq_length)

        # Test unflattening
        unflattened = LLaDASAE.unflatten_sequence_tensor(flattened, original_shape)
        assert unflattened.shape == original_tensor.shape
        assert torch.allclose(unflattened, original_tensor)

    def test_forward_flattened_method(self, device, sample_inference_tensors):
        """Test forward_flattened method with 3D input."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        batch_size, seq_length = 2, 8
        input_3d = torch.randn(batch_size, seq_length, hidden_dim, device=device)

        # Use forward_flattened method
        reconstruction, sparse_acts, pre_acts = sae.forward_flattened(input_3d)

        # Check output shapes match input
        assert reconstruction.shape == input_3d.shape
        assert sparse_acts.shape == (batch_size, seq_length, sae.d_sae)
        assert pre_acts.shape == (batch_size, seq_length, sae.d_sae)

        # Check sparsity constraint
        non_zero_counts = (sparse_acts != 0).sum(dim=-1)
        assert (non_zero_counts == sae.k_sparse).all()

    def test_forward_flattened_with_2d_input(self, device, sample_inference_tensors):
        """Test forward_flattened method with 2D input (should use regular forward)."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(
            d_model=hidden_dim,
            d_sae=hidden_dim * 4,
            k_sparse=64,
            normalize_decoder=False,  # Disable normalization for deterministic comparison
        ).to(device)
        sae.eval()  # Set to eval mode to prevent any normalization differences

        total_tokens = 16
        input_2d = torch.randn(total_tokens, hidden_dim, device=device)

        # forward_flattened should handle 2D input by calling regular forward
        reconstruction_flat, sparse_acts_flat, pre_acts_flat = sae.forward_flattened(
            input_2d
        )
        reconstruction_reg, sparse_acts_reg, pre_acts_reg = sae.forward(input_2d)

        assert torch.allclose(reconstruction_flat, reconstruction_reg)
        assert torch.allclose(sparse_acts_flat, sparse_acts_reg)
        assert torch.allclose(pre_acts_flat, pre_acts_reg)

    def test_equivalence_3d_vs_flattened(
        self, device, sample_inference_tensors, random_seed
    ):
        """Test that 3D input and its flattened equivalent produce same results."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(
            d_model=hidden_dim,
            d_sae=hidden_dim * 4,
            k_sparse=64,
            normalize_decoder=False,  # Disable for deterministic comparison
        ).to(device)
        sae.eval()

        batch_size, seq_length = 2, 8
        input_3d = torch.randn(batch_size, seq_length, hidden_dim, device=device)

        # Process as 3D
        recon_3d, sparse_3d, pre_3d = sae.forward_flattened(input_3d)

        # Process as flattened 2D
        input_2d, original_shape = LLaDASAE.flatten_sequence_tensor(input_3d)
        recon_2d_flat, sparse_2d_flat, pre_2d_flat = sae.forward(input_2d)

        # Unflatten 2D results back to 3D
        recon_2d = LLaDASAE.unflatten_sequence_tensor(recon_2d_flat, original_shape)
        sparse_2d = LLaDASAE.unflatten_sequence_tensor(sparse_2d_flat, original_shape)
        pre_2d = LLaDASAE.unflatten_sequence_tensor(pre_2d_flat, original_shape)

        # Results should be identical
        assert torch.allclose(recon_3d, recon_2d, rtol=1e-5)
        assert torch.allclose(sparse_3d, sparse_2d, rtol=1e-5)
        assert torch.allclose(pre_3d, pre_2d, rtol=1e-5)

    def test_loss_computation_with_flattened_input(
        self, device, sample_inference_tensors
    ):
        """Test loss computation with flattened 2D inputs."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        total_tokens = 16
        input_2d = torch.randn(total_tokens, hidden_dim, device=device)

        reconstruction, sparse_acts, pre_acts = sae(input_2d)
        loss_dict = sae.compute_loss(input_2d, reconstruction, sparse_acts)

        # Check all required loss components are present
        required_keys = [
            "total_loss",
            "recon_loss",
            "weight_decay_loss",
            "sparsity_ratio",
            "l0_norm",
            "max_recon_error",
            "mean_recon_error",
        ]
        for key in required_keys:
            assert key in loss_dict
            assert torch.isfinite(loss_dict[key])

    def test_feature_activations_with_flattened_input(
        self, device, sample_inference_tensors
    ):
        """Test get_feature_activations with flattened input."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        total_tokens = 16
        input_2d = torch.randn(total_tokens, hidden_dim, device=device)

        # Test without indices
        sparse_acts, indices = sae.get_feature_activations(
            input_2d, return_indices=False
        )
        assert sparse_acts.shape == (total_tokens, sae.d_sae)
        assert indices is None

        # Test with indices
        sparse_acts, indices = sae.get_feature_activations(
            input_2d, return_indices=True
        )
        assert sparse_acts.shape == (total_tokens, sae.d_sae)
        assert indices.shape == (total_tokens, sae.k_sparse)

        # Verify indices point to non-zero activations
        for i in range(total_tokens):
            token_indices = indices[i]
            token_acts = sparse_acts[i]
            assert (token_acts[token_indices] != 0).all()

    def test_batch_processing_different_shapes(self, device, sample_inference_tensors):
        """Test processing batches with different total token counts."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        configs = [
            (1, 24),
            (2, 12),
            (3, 8),
            (4, 6),
        ]

        results = []
        for batch_size, seq_length in configs:
            input_3d = torch.randn(batch_size, seq_length, hidden_dim, device=device)

            input_2d, _ = LLaDASAE.flatten_sequence_tensor(input_3d)
            reconstruction, sparse_acts, pre_acts = sae(input_2d)

            results.append(
                {
                    "config": (batch_size, seq_length),
                    "total_tokens": batch_size * seq_length,
                    "shapes": {
                        "input": input_2d.shape,
                        "reconstruction": reconstruction.shape,
                        "sparse_acts": sparse_acts.shape,
                        "pre_acts": pre_acts.shape,
                    },
                }
            )

        total_tokens = results[0]["total_tokens"]
        for result in results:
            assert result["total_tokens"] == total_tokens
            assert result["shapes"]["input"][0] == total_tokens
            assert result["shapes"]["reconstruction"][0] == total_tokens
            assert result["shapes"]["sparse_acts"][0] == total_tokens

    def test_gradient_flow_flattened_input(self, device, sample_inference_tensors):
        """Test gradient flow with flattened input format."""
        hidden_dim = sample_inference_tensors["hidden_dim"]
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)

        total_tokens = 16
        input_2d = torch.randn(
            total_tokens, hidden_dim, device=device, requires_grad=True
        )

        reconstruction, sparse_acts, pre_acts = sae(input_2d)
        loss_dict = sae.compute_loss(input_2d, reconstruction, sparse_acts)

        # Backpropagate
        loss_dict["total_loss"].backward()

        # Check gradients exist
        assert input_2d.grad is not None
        assert sae.encoder.weight.grad is not None
        assert sae.encoder.bias.grad is not None
        assert sae.decoder.weight.grad is not None
        if sae.decoder.bias is not None:
            assert sae.decoder.bias.grad is not None

    def test_error_handling_invalid_shapes(self, device, sample_inference_tensors):
        """Test error handling for invalid tensor shapes."""
        hidden_dim = sample_inference_tensors["hidden_dim"]

        # Test flatten_sequence_tensor with wrong dimensions
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            LLaDASAE.flatten_sequence_tensor(torch.randn(10, hidden_dim))

        # Test unflatten_sequence_tensor with wrong dimensions
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            LLaDASAE.unflatten_sequence_tensor(torch.randn(2, 5, hidden_dim), (2, 5))

        # Test unflatten_sequence_tensor with shape mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            LLaDASAE.unflatten_sequence_tensor(
                torch.randn(10, hidden_dim),
                (2, 6),  # 2*6=12 != 10
            )

        # Test forward_flattened with wrong dimensions
        sae = LLaDASAE(d_model=hidden_dim, d_sae=hidden_dim * 4, k_sparse=64).to(device)
        with pytest.raises(ValueError, match="Expected 2D or 3D input tensor"):
            sae.forward_flattened(torch.randn(2, 3, 4, hidden_dim))


if __name__ == "__main__":
    pytest.main([__file__])
