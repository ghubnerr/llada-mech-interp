import torch
import torch.nn.functional as F
import numpy as np
from llada_mi.utils import add_gumbel_noise, get_num_transfer_tokens
from typing import Tuple


@torch.no_grad()
def generate_with_logitlens(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 50256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate text with logit lens.

    Args:
        model: The model to use for the generation.
        prompt: The prompt to use for the generation.
        steps: The number of steps to generate.
        gen_length: The length of the generated text.
        block_length: The length of the block.
        temperature: The temperature for the generation.
        remasking: The remasking method to use.
        mask_id: The mask id to use.

    Returns:
        x: The generated text tensor with shape (b, t).
        all_logits: The logits for each step (stacked into a single tensor) with shape
            (s, b, t, c).
        all_hidden_states: The hidden states for each step (stacked into a single tensor) with shape
            (s, l, b, t, c).
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id

    if block_length > gen_length:
        block_length = gen_length

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    all_logits = []
    all_hidden_states = []

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id

            with torch.no_grad():
                outputs = model(x, output_hidden_states=True)
                logits = outputs.logits
                hidden_states = outputs.hidden_states

            all_logits.append(logits.clone().cpu())
            all_hidden_states.append([h.clone().cpu() for h in hidden_states])

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0_p[prompt_index] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return (
        x,
        torch.stack(all_logits),
        torch.stack(
            [torch.stack(hidden_states) for hidden_states in all_hidden_states]
        ),
    )
