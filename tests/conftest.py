"""Shared pytest fixtures for testing PyTorch models and tensors."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock


@pytest.fixture(scope="session")
def device():
    """Fixture to get the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def random_seed():
    """Fixture to set random seeds for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def sample_prompt(device):
    """Sample 4-token prompt tensor."""
    return torch.tensor([[1234, 5678, 9012, 3456]], device=device)


@pytest.fixture
def sample_inference_tensors(device, generation_params):
    """Sample tensors with realistic shapes for inference pipeline testing."""
    batch_size = 1
    prompt_length = 4
    gen_length = generation_params["gen_length"]
    total_length = prompt_length + gen_length
    steps = generation_params["steps"]
    vocab_size = 50000
    hidden_dim = 4096
    num_layers = 32

    x = torch.full(
        (batch_size, total_length),
        generation_params["mask_id"],
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = torch.tensor([[1234, 5678, 9012, 3456]], device=device)

    logits = torch.randn(batch_size, total_length, vocab_size, device=device)

    hidden_states = [
        torch.randn(batch_size, total_length, hidden_dim, device=device)
        for _ in range(num_layers)
    ]

    all_logits = torch.randn(steps, batch_size, total_length, vocab_size, device=device)

    all_hidden_states = torch.randn(
        steps, num_layers, batch_size, total_length, hidden_dim, device=device
    )

    return {
        "x": x,
        "logits": logits,
        "hidden_states": hidden_states,
        "all_logits": all_logits,
        "all_hidden_states": all_hidden_states,
        "prompt_length": prompt_length,
        "total_length": total_length,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "steps": steps,
    }


@pytest.fixture
def mock_model_with_inference_output(sample_inference_tensors, device):
    """Mock model that returns properly shaped inference outputs."""
    model = Mock()
    model.device = device

    def mock_forward(x, output_hidden_states=False, **kwargs):
        outputs = Mock()
        batch_size, seq_len = x.shape
        vocab_size = sample_inference_tensors["vocab_size"]
        hidden_dim = sample_inference_tensors["hidden_dim"]
        num_layers = sample_inference_tensors["num_layers"]

        outputs.logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        if output_hidden_states:
            outputs.hidden_states = [
                torch.randn(batch_size, seq_len, hidden_dim, device=device)
                for _ in range(num_layers)
            ]

        return outputs

    model.forward = mock_forward
    model.__call__ = mock_forward

    model.model = Mock()
    model.model.transformer = Mock()
    model.model.transformer.ln_f = Mock(side_effect=lambda x: x)
    model.model.transformer.ff_out = Mock(
        return_value=torch.randn(
            1,
            sample_inference_tensors["total_length"],
            sample_inference_tensors["vocab_size"],
            device=device,
        )
    )

    return model


@pytest.fixture
def expected_inference_shapes(generation_params):
    """Expected tensor shapes for inference outputs."""
    batch_size = 1
    prompt_length = 4
    gen_length = generation_params["gen_length"]
    total_length = prompt_length + gen_length
    steps = generation_params["steps"]
    vocab_size = 50000
    hidden_dim = 4096
    num_layers = 32

    return {
        "final_sequence": (batch_size, total_length),
        "all_logits": (steps, batch_size, total_length, vocab_size),
        "all_hidden_states": (steps, num_layers, batch_size, total_length, hidden_dim),
        "single_step_logits": (batch_size, total_length, vocab_size),
        "single_layer_hidden": (batch_size, total_length, hidden_dim),
        "confidence_scores": (batch_size, total_length),
        "transfer_mask": (batch_size, total_length),
    }


@pytest.fixture
def sample_generation_trajectory(device, generation_params):
    """Sample data showing progressive token unmasking over generation steps."""
    batch_size = 1
    prompt_length = 4
    gen_length = generation_params["gen_length"]
    total_length = prompt_length + gen_length
    steps = generation_params["steps"]
    mask_id = generation_params["mask_id"]

    initial_x = torch.full(
        (batch_size, total_length), mask_id, dtype=torch.long, device=device
    )
    initial_x[:, :prompt_length] = torch.tensor(
        [[1234, 5678, 9012, 3456]], device=device
    )

    trajectory = []
    current_x = initial_x.clone()

    for step in range(steps):
        num_unmasked = min(step + 1, gen_length)
        if num_unmasked > 0:
            for i in range(num_unmasked):
                pos = prompt_length + i
                if pos < total_length and current_x[0, pos] == mask_id:
                    current_x[0, pos] = torch.randint(
                        2, 1000, (1,), device=device
                    ).item()

        trajectory.append(current_x.clone())

    return {
        "initial_state": initial_x,
        "trajectory": trajectory,
        "final_state": trajectory[-1],
        "prompt_length": prompt_length,
        "mask_id": mask_id,
    }


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer with basic encode/decode functionality."""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 50000

    def mock_encode(text, **kwargs):
        words = text.split()
        return [hash(word) % 1000 + 2 for word in words]

    def mock_call(text, return_tensors=None, **kwargs):
        input_ids = mock_encode(text)
        result = {"input_ids": input_ids}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([input_ids])
        return result

    def mock_decode(token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return " ".join([f"token_{tid}" for tid in token_ids])

    def mock_batch_decode(token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [mock_decode([seq]) for seq in token_ids]

    def mock_convert_ids_to_tokens(token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [f"token_{tid}" for tid in token_ids]

    tokenizer.__call__ = mock_call
    tokenizer.encode = mock_encode
    tokenizer.decode = mock_decode
    tokenizer.batch_decode = mock_batch_decode
    tokenizer.convert_ids_to_tokens = mock_convert_ids_to_tokens

    return tokenizer


@pytest.fixture
def generation_params():
    """Sample generation parameters for testing."""
    return {
        "steps": 8,
        "gen_length": 16,
        "block_length": 16,
        "temperature": 0.0,
        "remasking": "low_confidence",
        "mask_id": 126336,
    }


@pytest.fixture
def tensor_assertion_helpers():
    """Helper functions for tensor assertions in tests."""

    def assert_tensor_shape(tensor, expected_shape):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {tensor.shape}"
        )

    def assert_tensor_dtype(tensor, expected_dtype):
        """Assert tensor has expected dtype."""
        assert tensor.dtype == expected_dtype, (
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
        )

    def assert_tensor_device(tensor, expected_device):
        """Assert tensor is on expected device."""
        assert str(tensor.device) == str(expected_device), (
            f"Expected device {expected_device}, got {tensor.device}"
        )

    def assert_tensor_finite(tensor):
        """Assert all tensor values are finite (not NaN or inf)."""
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"

    def assert_tensors_close(tensor1, tensor2, rtol=1e-5, atol=1e-8):
        """Assert two tensors are close in value."""
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), (
            "Tensors are not close"
        )

    return {
        "assert_shape": assert_tensor_shape,
        "assert_dtype": assert_tensor_dtype,
        "assert_device": assert_tensor_device,
        "assert_finite": assert_tensor_finite,
        "assert_close": assert_tensors_close,
    }


@pytest.fixture
def mock_huggingface_model():
    """Mock HuggingFace model loading."""
    with pytest.mock.patch(
        "transformers.AutoModel.from_pretrained"
    ) as mock_model, pytest.mock.patch(
        "transformers.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer_cls:
        model_instance = Mock()
        model_instance.eval.return_value = model_instance
        model_instance.to.return_value = model_instance
        model_instance.device = "cpu"
        mock_model.return_value = model_instance

        tokenizer_instance = Mock()
        tokenizer_instance.pad_token_id = 0
        tokenizer_instance.eos_token_id = 1
        mock_tokenizer_cls.return_value = tokenizer_instance

        yield {
            "model": model_instance,
            "tokenizer": tokenizer_instance,
            "mock_model_cls": mock_model,
            "mock_tokenizer_cls": mock_tokenizer_cls,
        }


@pytest.fixture(scope="session", autouse=True)
def configure_torch():
    """Configure PyTorch for testing."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_default_tensor_type(torch.FloatTensor)

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test characteristics."""
    for item in items:
        if "gpu" in item.nodeid.lower() or any(
            "gpu" in mark.name for mark in item.iter_markers()
        ):
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))

        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
