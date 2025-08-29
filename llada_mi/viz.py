import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from llada_mi.utils import (
    prettify_text,
    entropy_from_logits,
    get_logits_from_hidden_state,
)
import imageio
import os
import torch
from typing import List, Tuple
from transformers import PreTrainedTokenizer


class HeatmapData:
    def __init__(
        self,
        model: torch.nn.Module,
        hidden_states: torch.Tensor,
        final_x: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.hidden_states = hidden_states
        self.final_x = final_x
        self.tokenizer = tokenizer
        self.tokens = self._get_tokens()

    def _get_tokens(self) -> List[str]:
        raw_tokens = self.tokenizer.convert_ids_to_tokens(
            self.final_x[0], skip_special_tokens=False
        )
        return prettify_text(raw_tokens)


def _compute_layer_data(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[List[str]], List[np.ndarray]]:
    """
    Compute logits and entropies for all layers.

    Args:
        model: The model to use for computing logits
        hidden_states: Hidden states tensor with shape (num_layers, batch_size, sequence_length, hidden_size)
        tokenizer: Tokenizer for converting logits to tokens

    Returns:
        Tuple of (predicted_tokens, entropies)
    """
    predicted_tokens = []
    entropies = []

    for layer_idx in range(hidden_states.shape[0]):
        hidden_state = hidden_states[layer_idx]
        logits, tokens = get_logits_from_hidden_state(model, hidden_state, tokenizer)
        predicted_tokens.append(tokens)
        entropy = entropy_from_logits(logits).float().cpu().detach().numpy()
        entropies.append(entropy)

    return predicted_tokens, entropies


def _create_heatmap(
    entropies: List[np.ndarray],
    predicted_tokens: List[List[str]],
    input_tokens: List[str],
    num_layers: int,
    skip_every_n: int = 2,
    figsize: Tuple[int, int] = (24, 12),
    font_size: int = 14,
    show_colorbar: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a heatmap visualization.

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="white")

    heatmap = sns.heatmap(
        np.stack(entropies)[::skip_every_n],
        annot=prettify_text(predicted_tokens)[::skip_every_n],
        fmt="",
        cmap="YlGnBu",
        xticklabels=input_tokens,
        yticklabels=list(range(num_layers))[::skip_every_n],
        cbar=show_colorbar,
        annot_kws={"size": font_size, "fontweight": "bold"},
        linewidths=0.8,
        ax=ax,
    )
    heatmap.invert_yaxis()

    return fig, ax


def _style_heatmap(
    ax: plt.Axes,
    title: str,
    font_size: int = 16,
    title_size: int = 18,
) -> None:
    """Apply consistent styling to a heatmap."""
    ax.set_title(title, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("Tokens", fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=font_size, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)


def show_heatmap_step(
    data: HeatmapData,
    step: int,
    skip_every_n: int = 2,
) -> None:
    """
    Display a single heatmap for a specific timestep.

    Args:
        data: HeatmapData container with model and data
        step: Timestep to visualize
        skip_every_n: Number of layers to skip
    """
    hidden_states = data.hidden_states[step]
    predicted_tokens, entropies = _compute_layer_data(
        data.model, hidden_states, data.tokenizer
    )

    fig, ax = _create_heatmap(
        entropies=entropies,
        predicted_tokens=predicted_tokens,
        input_tokens=data.tokens,
        num_layers=hidden_states.shape[0],
        skip_every_n=skip_every_n,
    )

    _style_heatmap(
        ax=ax,
        title=f"Logit Lens at Diffusion Timestep {step}",
        title_size=20,
    )

    plt.tight_layout()
    plt.show()


def compare_steps(
    data: HeatmapData,
    step1: int,
    step2: int,
    skip_every_n: int = 2,
) -> None:
    """
    Compare two timesteps side by side.

    Args:
        data: HeatmapData container with model and data
        step1: First timestep to compare
        step2: Second timestep to compare
        skip_every_n: Number of layers to skip
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))
    sns.set_theme(style="white")

    for ax, step, title in [
        (ax1, step1, f"Timestep {step1}"),
        (ax2, step2, f"Timestep {step2}"),
    ]:
        hidden_states = data.hidden_states[step]
        predicted_tokens, entropies = _compute_layer_data(
            data.model, hidden_states, data.tokenizer
        )

        heatmap = sns.heatmap(
            np.stack(entropies)[::skip_every_n],
            annot=prettify_text(predicted_tokens)[::skip_every_n],
            fmt="",
            cmap="YlGnBu",
            xticklabels=data.tokens,
            yticklabels=list(range(hidden_states.shape[0]))[
                ::skip_every_n
            ],  # Use tensor shape instead of len()
            cbar=True,
            annot_kws={"size": 12, "fontweight": "bold"},
            linewidths=0.8,
            ax=ax,
        )
        heatmap.invert_yaxis()

        _style_heatmap(ax=ax, title=title, font_size=14, title_size=16)

    plt.suptitle(
        f"Comparison: Timestep {step1} vs Timestep {step2}",
        fontsize=20,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def save_animation(
    data: HeatmapData,
    output_path: str = "timestep_animation.gif",
    interval_ms: int = 2000,
    skip_every_n: int = 2,
    dpi: int = 100,
) -> str:
    """
    Create an animated GIF showing evolution across timesteps.

    Args:
        data: HeatmapData container with model and data
        output_path: Path to save the GIF
        interval_ms: Interval between frames in milliseconds
        skip_every_n: Number of layers to skip
        dpi: Image resolution

    Returns:
        Path to the saved GIF
    """
    print(f"Creating GIF animation with {data.hidden_states.shape[0]} timesteps...")

    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    for step in range(data.hidden_states.shape[0]):
        print(f"Processing timestep {step}/{data.hidden_states.shape[0] - 1}")

        hidden_states = data.hidden_states[step]
        predicted_tokens, entropies = _compute_layer_data(
            data.model, hidden_states, data.tokenizer
        )

        fig, ax = _create_heatmap(
            entropies=entropies,
            predicted_tokens=predicted_tokens,
            input_tokens=data.tokens,
            num_layers=hidden_states.shape[0],
            skip_every_n=skip_every_n,
            figsize=(20, 10),
            font_size=14,
            show_colorbar=True,
        )

        _style_heatmap(
            ax=ax,
            title=f"Logit Lens at Diffusion Timestep {step}",
            title_size=18,
        )
        plt.tight_layout()

        frame_path = os.path.join(temp_dir, f"frame_{step:04d}.png")
        plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        frame = imageio.imread(frame_path)
        frames.append(frame)

    imageio.mimsave(output_path, frames, duration=interval_ms / 1000.0)

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    return output_path
