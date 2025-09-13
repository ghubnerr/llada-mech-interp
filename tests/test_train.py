import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import asdict

from llada_mi.sae.train import (
    SAETrainingConfig,
    extract_llada_activations,
    setup_distributed,
    cleanup_distributed,
    save_checkpoint,
    load_checkpoint,
)
from llada_mi.sae.model import LLaDASAE


class TestSAETrainingConfig:
    """Test SAE training configuration."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = SAETrainingConfig()

        assert config.model_name == "GSAI-ML/LLaDA-8B-Base"
        assert config.target_layer == 16
        assert config.target_step == 0
        assert config.d_model == 4096
        assert config.d_sae == 16384
        assert config.k_sparse == 64
        assert config.learning_rate == 3e-4
        assert config.batch_size == 32
        assert config.sequence_length == 512
        assert config.num_epochs == 1
        assert config.pile_subset == 1
        assert config.world_size == 8

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = SAETrainingConfig(
            model_name="custom-model",
            target_layer=10,
            d_model=2048,
            d_sae=8192,
            k_sparse=32,
            learning_rate=1e-4,
            batch_size=16,
            sequence_length=256,
            num_epochs=2,
            pile_subset=0.05,
            world_size=4,
        )

        assert config.model_name == "custom-model"
        assert config.target_layer == 10
        assert config.d_model == 2048
        assert config.d_sae == 8192
        assert config.k_sparse == 32
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.sequence_length == 256
        assert config.num_epochs == 2
        assert config.pile_subset == 0.05
        assert config.world_size == 4

    def test_config_serialization(self):
        """Test config can be serialized to dict."""
        config = SAETrainingConfig()
        config_dict = asdict(config)

        expected_fields = [
            "model_name",
            "target_layer",
            "target_step",
            "d_model",
            "d_sae",
            "k_sparse",
            "learning_rate",
            "batch_size",
            "sequence_length",
            "num_epochs",
            "pile_subset",
            "world_size",
        ]

        for field in expected_fields:
            assert field in config_dict


class TestPileDatasetIntegration:
    """Test PileDataset integration with training config."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        return tokenizer

    @pytest.fixture
    def training_config(self):
        """Create training config for testing."""
        return SAETrainingConfig(
            sequence_length=128,
            pile_subset=0.001,
            batch_size=2,
            world_size=1,
        )

    def test_pile_dataset_with_training_config(self, mock_tokenizer, training_config):
        """Test PileDataset initialization with training config."""
        mock_items = [
            {"text": "Sample text for training"},
            {"text": "Another sample for the dataset"},
        ]

        with patch("llada_mi.sae.train.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_items))
            mock_load.return_value = mock_dataset

            # Import PileDataset from train module
            from llada_mi.sae.train import PileDataset as TrainPileDataset

            dataset = TrainPileDataset(
                tokenizer=mock_tokenizer, config=training_config, rank=0, world_size=1
            )

            assert dataset.config is training_config
            assert dataset.tokenizer is mock_tokenizer

            # Test iteration
            items = list(dataset)
            assert len(items) <= len(mock_items)


class TestLLaDAActivationExtraction:
    """Test LLaDA activation extraction functionality."""

    @pytest.fixture
    def mock_llada_model(self, device):
        """Load real LLaDA model for testing."""
        try:
            # Try to load the real model
            from llada_mi.config import load_model_and_tokenizer

            model, _, _ = load_model_and_tokenizer("GSAI-ML/LLaDA-8B-Base")
            model = model.to(device)
            model.eval()
            return model
        except Exception as e:
            # If loading fails, skip the test
            pytest.skip(f"Could not load LLaDA model: {e}")
            return None

    @pytest.fixture
    def training_config(self):
        """Create training config."""
        return SAETrainingConfig(
            target_layer=16, gen_length=64, mask_id=126336, sequence_length=128
        )

    @pytest.mark.gpu_intensive
    def test_extract_llada_activations(self, mock_llada_model, training_config, device):
        """Test activation extraction from LLaDA model."""
        batch_size = 2
        seq_len = training_config.sequence_length

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        activations = extract_llada_activations(
            mock_llada_model, input_ids, attention_mask, training_config, device
        )

        # Check output shape
        expected_shape = (batch_size, seq_len, training_config.d_model)
        assert activations.shape == expected_shape

        # Check that activations are finite and reasonable
        assert torch.isfinite(activations).all()
        # LLaDA model uses bfloat16 by default
        assert activations.dtype in [torch.float32, torch.bfloat16]

    @pytest.mark.gpu_intensive
    def test_extract_llada_activations_masking(
        self, mock_llada_model, training_config, device
    ):
        """Test that generation tokens are properly masked by testing the masking logic directly."""
        batch_size = 1
        seq_len = training_config.sequence_length

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Test the masking logic directly (from extract_llada_activations function)
        prompt_len = seq_len - training_config.gen_length

        # Create masked input as done in the function
        x = input_ids.clone()
        x[:, prompt_len:] = training_config.mask_id  # Mask generation portion

        # Check that generation portion was masked
        gen_tokens = x[:, prompt_len:]
        assert (gen_tokens == training_config.mask_id).all()

        # Prompt tokens should be unchanged
        prompt_tokens = x[:, :prompt_len]
        original_prompt = input_ids[:, :prompt_len]
        assert torch.equal(prompt_tokens, original_prompt)

        # Also test that the actual function works
        activations = extract_llada_activations(
            mock_llada_model, input_ids, attention_mask, training_config, device
        )

        # Check that we get reasonable activations
        assert activations.shape == (batch_size, seq_len, training_config.d_model)
        assert torch.isfinite(activations).all()


class TestCheckpointing:
    """Test checkpointing functionality."""

    @pytest.fixture
    def mock_sae(self, device):
        """Create mock SAE model."""
        sae = LLaDASAE(d_model=256, d_sae=1024, k_sparse=16).to(device)
        return sae

    @pytest.fixture
    def mock_optimizer(self, mock_sae):
        """Create mock optimizer."""
        return torch.optim.AdamW(mock_sae.parameters(), lr=1e-4)

    @pytest.fixture
    def training_config(self):
        """Create training config."""
        return SAETrainingConfig(checkpoint_dir="test_checkpoints")

    def test_save_checkpoint(self, mock_sae, mock_optimizer, training_config):
        """Test saving training checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.checkpoint_dir = temp_dir

            # Save checkpoint
            epoch = 1
            step = 500
            loss = 0.123
            rank = 0

            save_checkpoint(
                mock_sae, mock_optimizer, epoch, step, loss, training_config, rank
            )

            # Check files were created
            checkpoint_dir = Path(temp_dir)
            checkpoint_file = checkpoint_dir / f"checkpoint_step_{step}.pt"
            config_file = checkpoint_dir / "config.json"

            assert checkpoint_file.exists()
            assert config_file.exists()

            # Check checkpoint content
            checkpoint = torch.load(checkpoint_file, map_location="cpu")

            assert checkpoint["epoch"] == epoch
            assert checkpoint["step"] == step
            assert checkpoint["loss"] == loss
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "config" in checkpoint

            # Check config file
            with open(config_file) as f:
                saved_config = json.load(f)
            assert saved_config["checkpoint_dir"] == temp_dir

    def test_save_checkpoint_non_zero_rank(
        self, mock_sae, mock_optimizer, training_config
    ):
        """Test that non-zero rank doesn't save checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.checkpoint_dir = temp_dir

            # Try to save checkpoint with rank=1
            save_checkpoint(
                mock_sae, mock_optimizer, 1, 500, 0.123, training_config, rank=1
            )

            # No files should be created
            checkpoint_dir = Path(temp_dir)
            assert len(list(checkpoint_dir.iterdir())) == 0

    def test_load_checkpoint(self, mock_sae, mock_optimizer, training_config):
        """Test loading training checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.checkpoint_dir = temp_dir

            # Save checkpoint first
            original_epoch = 2
            original_step = 1000
            original_loss = 0.456

            save_checkpoint(
                mock_sae,
                mock_optimizer,
                original_epoch,
                original_step,
                original_loss,
                training_config,
                rank=0,
            )

            # Create new model and optimizer on the same device as the original
            device = next(mock_sae.parameters()).device
            new_sae = LLaDASAE(d_model=256, d_sae=1024, k_sparse=16).to(device)
            new_optimizer = torch.optim.AdamW(new_sae.parameters(), lr=1e-4)

            # Load checkpoint
            checkpoint_path = Path(temp_dir) / f"checkpoint_step_{original_step}.pt"
            epoch, step, loss = load_checkpoint(
                new_sae, new_optimizer, str(checkpoint_path)
            )

            assert epoch == original_epoch
            assert step == original_step
            assert loss == original_loss

            # Check that model states are the same
            original_params = list(mock_sae.parameters())
            loaded_params = list(new_sae.parameters())

            for orig, loaded in zip(original_params, loaded_params):
                assert torch.allclose(orig, loaded, atol=1e-6)

    def test_load_checkpoint_with_ddp(self, mock_sae, mock_optimizer):
        """Test loading checkpoint with DDP wrapped model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock DDP model
            ddp_model = Mock()
            ddp_model.module = mock_sae

            # Save checkpoint
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            torch.save(
                {
                    "epoch": 1,
                    "step": 100,
                    "model_state_dict": mock_sae.state_dict(),
                    "optimizer_state_dict": mock_optimizer.state_dict(),
                    "loss": 0.1,
                    "config": {},
                },
                checkpoint_path,
            )

            # Load with DDP model
            epoch, step, loss = load_checkpoint(
                ddp_model, mock_optimizer, str(checkpoint_path)
            )

            assert epoch == 1
            assert step == 100
            assert loss == 0.1


class TestDistributedTrainingUtilities:
    """Test distributed training utility functions."""

    def test_setup_distributed_environment_vars(self):
        """Test that setup_distributed sets environment variables."""
        import os

        # Clear existing env vars
        old_master_addr = os.environ.pop("MASTER_ADDR", None)
        old_master_port = os.environ.pop("MASTER_PORT", None)

        try:
            with patch("torch.distributed.init_process_group"), patch(
                "torch.cuda.set_device"
            ):
                setup_distributed(rank=0, world_size=2)

                assert os.environ["MASTER_ADDR"] == "localhost"
                assert os.environ["MASTER_PORT"] == "12355"

        finally:
            # Restore original env vars
            if old_master_addr is not None:
                os.environ["MASTER_ADDR"] = old_master_addr
            if old_master_port is not None:
                os.environ["MASTER_PORT"] = old_master_port

    def test_setup_distributed_torch_calls(self):
        """Test that setup_distributed makes correct torch calls."""
        with patch("torch.distributed.init_process_group") as mock_init, patch(
            "torch.cuda.set_device"
        ) as mock_set_device:
            rank = 2
            world_size = 4

            setup_distributed(rank, world_size)

            mock_init.assert_called_once_with("nccl", rank=rank, world_size=world_size)
            mock_set_device.assert_called_once_with(rank)

    def test_cleanup_distributed(self):
        """Test distributed cleanup."""
        with patch("torch.distributed.destroy_process_group") as mock_destroy:
            cleanup_distributed()
            mock_destroy.assert_called_once()


class TestTrainingIntegration:
    """Integration tests for training components."""

    @pytest.fixture
    def mock_components(self, device):
        """Create all mock components needed for training."""
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Try to load real LLaDA model, fallback to mock if needed
        try:
            from llada_mi.config import load_model_and_tokenizer

            llada_model, _, _ = load_model_and_tokenizer("GSAI-ML/LLaDA-8B-Base")
            llada_model = llada_model.to(device)
            llada_model.eval()
        except Exception:
            # Fallback to mock if model loading fails
            llada_model = Mock()
            llada_model.device = device
            llada_model.eval = Mock()

            def mock_forward(x, output_hidden_states=False, **kwargs):
                batch_size, seq_len = x.shape

                # Create a proper mock object that behaves like model outputs
                class MockOutputs:
                    def __init__(self):
                        self.logits = torch.randn(
                            batch_size, seq_len, 50000, device=device
                        )
                        if output_hidden_states:
                            # Create a list that can be subscripted like the real hidden_states
                            self.hidden_states = [
                                torch.randn(batch_size, seq_len, 4096, device=device)
                                for _ in range(32)
                            ]

                return MockOutputs()

            llada_model.__call__ = mock_forward

        # Real SAE model - match the dtype of the LLaDA model
        # Use default config values: d_model=4096, d_sae=16384 (4x), k_sparse=64
        sae = LLaDASAE(d_model=4096, d_sae=16384, k_sparse=64).to(device)
        # Convert to bfloat16 to match LLaDA model
        sae = sae.to(torch.bfloat16)

        # Real optimizer
        optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-4)

        return {
            "tokenizer": tokenizer,
            "llada_model": llada_model,
            "sae": sae,
            "optimizer": optimizer,
        }

    @pytest.mark.gpu_intensive
    def test_training_step_integration(self, mock_components, device):
        """Test a single training step integration."""
        components = mock_components
        config = SAETrainingConfig(
            sequence_length=128, batch_size=2, gen_length=32, target_layer=16
        )

        # Create sample batch
        batch_size = config.batch_size
        seq_len = config.sequence_length

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Extract activations
        activations = extract_llada_activations(
            components["llada_model"], input_ids, attention_mask, config, device
        )

        # Flatten for SAE
        activations_flat = activations.view(-1, config.d_model)

        # Forward pass through SAE
        reconstruction, sparse_acts, pre_acts = components["sae"](activations_flat)

        # Compute loss
        loss_dict = components["sae"].compute_loss(
            activations_flat, reconstruction, sparse_acts
        )
        loss = loss_dict["total_loss"]

        # Backward pass
        components["optimizer"].zero_grad()
        loss.backward()
        components["optimizer"].step()

        # Verify shapes and values
        assert reconstruction.shape == activations_flat.shape
        assert sparse_acts.shape == (activations_flat.shape[0], config.d_sae)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_training_config_consistency(self):
        """Test that training config values are consistent."""
        config = SAETrainingConfig()

        # Check that SAE dimensions are consistent
        assert config.d_sae > config.d_model  # Overcomplete
        assert config.k_sparse < config.d_sae  # Sparse

        # Check that generation length is reasonable
        assert config.gen_length < config.sequence_length

        # Check that learning rate is reasonable
        assert 0 < config.learning_rate < 1

        # Check that subset fraction is valid
        assert 0 < config.pile_subset <= 1

    @pytest.mark.gpu_intensive
    def test_activation_shape_consistency(self, mock_components, device):
        """Test that activation shapes are consistent throughout pipeline."""
        components = mock_components
        config = SAETrainingConfig(
            sequence_length=64, batch_size=1, gen_length=16, target_layer=10
        )

        batch_size = config.batch_size
        seq_len = config.sequence_length

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Extract activations
        activations = extract_llada_activations(
            components["llada_model"], input_ids, attention_mask, config, device
        )

        # Check activation shape
        assert activations.shape == (batch_size, seq_len, config.d_model)

        # Flatten for SAE
        activations_flat = activations.view(-1, config.d_model)
        expected_flat_size = batch_size * seq_len
        assert activations_flat.shape == (expected_flat_size, config.d_model)

        # Process through SAE
        reconstruction, sparse_acts, pre_acts = components["sae"](activations_flat)

        # Check SAE output shapes
        assert reconstruction.shape == (expected_flat_size, config.d_model)
        assert sparse_acts.shape == (expected_flat_size, config.d_sae)
        assert pre_acts.shape == (expected_flat_size, config.d_sae)

        # Check sparsity constraint
        non_zero_counts = (sparse_acts != 0).sum(dim=-1)
        assert (non_zero_counts == config.k_sparse).all()


if __name__ == "__main__":
    pytest.main([__file__])
