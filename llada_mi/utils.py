import torch
import re


def prettify_text(text_list):
    if isinstance(text_list, str):
        pattern = r"[^a-zA-Z0-9\s\.\,\!\?\<\>\|]"
        cleaned = re.sub(pattern, "", text_list)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned if cleaned not in ["", ","] else text_list[:5]
        cleaned = cleaned.replace("Ċ", "_").replace("Ġ", " ")
        return cleaned
    elif isinstance(text_list, (list, tuple)):
        return [prettify_text(text) for text in text_list]
    else:
        raise ValueError("Can't prettify this.")


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def entropy_from_logits(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1).clamp(1e-8, 1)
    return -torch.sum(probs * torch.log(probs), dim=-1).squeeze()


def get_logits_from_hidden_state(model, hidden_state, tokenizer):
    """
    Computes logits and predicted tokens from a hidden state.
    """
    hidden_state = model.model.transformer.ln_f(hidden_state.to(model.device))
    logits = model.model.transformer.ff_out(hidden_state)

    predicted_token_ids = logits.argmax(-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(
        predicted_token_ids[0].cpu(), skip_special_tokens=False
    )
    predicted_tokens = prettify_text(predicted_tokens)

    return logits, predicted_tokens
