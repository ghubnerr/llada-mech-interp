import torch


def prepare_sae_training_data(
    all_hidden_states: torch.Tensor,
    all_timesteps: torch.Tensor = None,
    target_layers: list = None,
    target_timesteps: list = None,
    target_positions: list = None,
    flatten_batch: bool = True,
) -> torch.Tensor:
    """
    Prepare hidden states for SAE training.

    Args:
        all_hidden_states: Hidden states tensor with shape (num_prompts, num_timesteps, num_layers, seq_len, hidden_dim)
        all_tokens: Optional tokens tensor for filtering
        all_timesteps: Optional timestep indices
        target_layers: List of layer indices to include (None = all layers)
        target_timesteps: List of timestep indices to include (None = all timesteps)
        target_positions: List of position indices to include (None = all positions)
        flatten_batch: Whether to flatten batch dimensions

    Returns:
        Training data tensor with shape (num_samples, hidden_dim) if flatten_batch=True
        or (num_prompts, num_timesteps, num_layers, num_positions, hidden_dim) if False
    """
    if target_layers is not None:
        all_hidden_states = all_hidden_states[:, :, target_layers, :, :]

    if target_timesteps is not None:
        if all_timesteps is not None:
            timestep_mask = torch.zeros(all_hidden_states.shape[1], dtype=torch.bool)
            for ts in target_timesteps:
                timestep_mask |= (all_timesteps == ts).any(dim=0)
            all_hidden_states = all_hidden_states[:, timestep_mask, :, :, :]
        else:
            all_hidden_states = all_hidden_states[:, target_timesteps, :, :, :]

    if target_positions is not None:
        all_hidden_states = all_hidden_states[:, :, :, target_positions, :]

    if flatten_batch:
        return all_hidden_states.reshape(-1, all_hidden_states.shape[-1])
    else:
        return all_hidden_states


def create_sae_training_batches(
    hidden_states: torch.Tensor,
    batch_size: int = 1024,
    shuffle: bool = True,
    device: str = "cuda",
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for SAE training.

    Args:
        hidden_states: Hidden states tensor with shape (num_samples, hidden_dim)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        device: Device to move data to

    Returns:
        DataLoader for training
    """
    dataset = torch.utils.data.TensorDataset(hidden_states.to(device))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )


def sample_activations_for_sae(
    model: torch.nn.Module,
    tokenizer,
    prompts: list,
    steps: int = 64,
    gen_length: int = 64,
    record_timesteps: list = None,
    target_layers: list = None,
    max_prompts: int = 100,
) -> torch.Tensor:
    """
    Convenience function to collect and prepare SAE training data.

    Args:
        model: The model to collect activations from
        tokenizer: Tokenizer for encoding prompts
        prompts: List of text prompts
        steps: Number of diffusion steps
        gen_length: Length of text to generate
        record_timesteps: Which timesteps to record (None = all)
        target_layers: Which layers to include (None = all)
        max_prompts: Maximum number of prompts to process

    Returns:
        Training data tensor with shape (num_samples, hidden_dim)
    """
    from llada_mi.inference import collect_activations_for_sae

    encoded_prompts = []
    for prompt in prompts[:max_prompts]:
        encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        encoded_prompts.append(encoded["input_ids"])

    prompt_batch = torch.cat(encoded_prompts, dim=0)

    _, all_hidden_states, all_timesteps = collect_activations_for_sae(
        model=model,
        prompts=prompt_batch,
        steps=steps,
        gen_length=gen_length,
        record_timesteps=record_timesteps,
        max_prompts=max_prompts,
    )

    training_data = prepare_sae_training_data(
        all_hidden_states=all_hidden_states,
        all_timesteps=all_timesteps,
        target_layers=target_layers,
        target_timesteps=record_timesteps,
        flatten_batch=True,
    )

    return training_data
