import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import json
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import torch.multiprocessing as mp

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random

from llada_mi.config import load_model_and_tokenizer
from llada_mi.sae.model import LLaDASAE


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SAETrainingConfig:
    """Configuration for SAE training."""

    # Model parameters
    model_name: str = "GSAI-ML/LLaDA-8B-Base"
    target_layer: int = 16
    target_step: int = 0

    # SAE architecture
    d_model: int = 4096
    d_sae: int = 16384
    k_sparse: int = 64
    tie_weights: bool = False
    normalize_decoder: bool = True
    bias_decoder: bool = True
    l2_coefficient: float = 1e-6
    normalize_activations: bool = True
    activation_norm_eps: float = 1e-6

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 32
    sequence_length: int = 512
    num_epochs: int = 1
    pile_subset: float = 1

    # Distributed training
    world_size: int = 8

    # Checkpointing
    checkpoint_dir: str = "checkpoints/sae_training"
    save_every: int = 1000
    log_every: int = 100
    max_steps: Optional[int] = None  # Maximum number of training steps

    # Dataset
    dataset_name: str = "monology/pile-uncopyrighted"
    dataset_path: Optional[str] = None  # Path to local dataset (if available)
    dataset_split: str = "train"
    streaming: bool = True

    # Generation parameters for LLaDA
    mask_id: int = 126336
    gen_length: int = 64
    temperature: float = 0.0
    remasking: str = "low_confidence"


class PileDataset(IterableDataset):
    """
    Streaming dataset for the Pile that yields tokenized sequences.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: SAETrainingConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # Add random delay to avoid simultaneous requests from all processes
        if rank > 0:
            time.sleep(random.uniform(1, 5))

        if hasattr(config, "dataset_path") and config.dataset_path:
            logger.info(f"Loading local dataset from: {config.dataset_path}")
            from datasets import load_from_disk

            self.dataset = load_from_disk(config.dataset_path)
            if config.dataset_split != "train":
                self.dataset = self.dataset[config.dataset_split]
        else:
            logger.info(f"Streaming dataset from HuggingFace: {config.dataset_name}")
            self.dataset = load_dataset(
                config.dataset_name, split=config.dataset_split, streaming=True
            )

        if rank > 0:
            self.dataset = self.dataset.skip(rank * 1000)  # Rough sharding

    def __iter__(self):
        """Iterate through the dataset, yielding tokenized batches."""
        count = 0

        max_samples = (
            int(len(self.dataset) * self.config.pile_subset)
            if hasattr(self.dataset, "__len__")
            else float("inf")
        )

        for item in self.dataset:
            if count >= max_samples:
                break

            if count % self.world_size != self.rank:
                count += 1
                continue

            text = item["text"]

            tokens = self.tokenizer(
                text,
                max_length=self.config.sequence_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            yield {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

            count += 1


def extract_llada_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: SAETrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract hidden states from LLaDA at the specified layer and diffusion step.

    Args:
        model: LLaDA model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        config: Training configuration
        device: Device to run on

    Returns:
        Hidden states from target layer and step [batch_size, seq_len, d_model]
    """
    model.eval()

    with torch.no_grad():
        # Create masked input for diffusion (mask out generation tokens)
        batch_size, seq_len = input_ids.shape
        prompt_len = seq_len - config.gen_length

        # Create diffusion input
        x = input_ids.clone()
        x[:, prompt_len:] = config.mask_id  # Mask generation portion

        # Run one diffusion step to get hidden states
        outputs = model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Extract activations from target layer (layer 16)
        # hidden_states is a tuple of (num_layers,) tensors
        target_activations = hidden_states[
            config.target_layer
        ]  # [batch_size, seq_len, d_model]

        return target_activations


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    config: SAETrainingConfig,
    rank: int,
):
    """Save training checkpoint."""
    if rank != 0:
        return

    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config.__dict__,
        },
        checkpoint_path / f"checkpoint_step_{step}.pt",
    )

    # Save config
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    logger.info(f"Checkpoint saved at step {step}")


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str
) -> Tuple[int, int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    loss = checkpoint["loss"]

    logger.info(f"Checkpoint loaded from step {step}")
    return epoch, step, loss


def train_sae(rank: int, world_size: int, config: SAETrainingConfig):
    """
    Main training function for a single GPU process.

    Args:
        rank: GPU rank (0-7)
        world_size: Total number of GPUs (8)
        config: Training configuration
    """
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    logger.info(f"Starting training on rank {rank}/{world_size}")

    logger.info("Loading LLaDA model...")
    model, tokenizer, _ = load_model_and_tokenizer(config.model_name)
    model = model.to(device)

    llada_model = DDP(model, device_ids=[rank])

    logger.info("Initializing SAE...")
    sae = LLaDASAE(
        d_model=config.d_model,
        d_sae=config.d_sae,
        k_sparse=config.k_sparse,
        tie_weights=config.tie_weights,
        normalize_decoder=config.normalize_decoder,
        bias_decoder=config.bias_decoder,
        l2_coefficient=config.l2_coefficient,
        normalize_activations=config.normalize_activations,
        activation_norm_eps=config.activation_norm_eps,
    ).to(device)

    sae = sae.to(dtype=torch.bfloat16)

    sae = DDP(sae, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=config.learning_rate, weight_decay=1e-6
    )

    logger.info("Loading dataset...")
    dataset = PileDataset(tokenizer, config, rank, world_size)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True
    )

    logger.info("Starting training...")
    step = 0
    total_loss = 0.0

    if hasattr(dataset.dataset, "__len__"):
        total_samples = len(dataset.dataset)
        total_samples = int(total_samples * config.pile_subset)
        batches_per_rank = (
            total_samples + world_size - 1
        ) // world_size  # Ceiling division
        estimated_total_batches = (
            batches_per_rank + config.batch_size - 1
        ) // config.batch_size

    start_time = time.time()

    for epoch in range(config.num_epochs):
        sae.train()

        pbar = (
            tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                total=estimated_total_batches,
            )
            if rank == 0
            else dataloader
        )

        for batch in pbar:
            if config.max_steps is not None and step >= config.max_steps:
                logger.info(f"Reached maximum steps limit: {config.max_steps}")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Extract LLaDA activations
            with torch.no_grad():
                activations = extract_llada_activations(
                    llada_model, input_ids, attention_mask, config, device
                )

            # Flatten activations for SAE training
            # Shape: [batch_size, seq_len, d_model] -> [batch_size * seq_len, d_model]
            batch_size, seq_len, d_model = activations.shape
            activations_flat = activations.view(-1, d_model)

            reconstruction, sparse_acts, pre_acts = sae(activations_flat)

            # Compute loss
            loss_dict = sae.module.compute_loss(
                activations_flat, reconstruction, sparse_acts
            )
            loss = loss_dict["total_loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            step += 1

            # Memory monitoring - track GPU memory usage
            if step % 10 == 0 and rank == 0:
                gpu_memory = torch.cuda.memory_allocated(device) / 1e9
                gpu_memory_max = torch.cuda.max_memory_allocated(device) / 1e9
                total_tokens = (
                    step
                    * config.batch_size
                    * config.sequence_length
                    * config.world_size
                )
                print(
                    f"Step {step}: GPU Memory: {gpu_memory:.2f}GB (max: {gpu_memory_max:.2f}GB), Total tokens: {total_tokens:,}"
                )

            if step % config.log_every == 0:
                avg_loss = total_loss / config.log_every
                total_loss = 0.0

                if rank == 0:
                    elapsed_time = time.time() - start_time
                    progress_ratio = (
                        step / estimated_total_batches
                        if estimated_total_batches > 0
                        else 0
                    )
                    if progress_ratio > 0:
                        estimated_total_time = elapsed_time / progress_ratio
                        remaining_time = estimated_total_time - elapsed_time
                        eta_hours = int(remaining_time // 3600)
                        eta_mins = int((remaining_time % 3600) // 60)
                        eta_str = f"ETA: {eta_hours:02d}:{eta_mins:02d}"
                    else:
                        eta_str = "ETA: calculating..."

                    logger.info(
                        f"Step {step}/{estimated_total_batches}: Loss={avg_loss:.6f}, "
                        f"Recon={loss_dict['recon_loss']:.6f}, "
                        f"L0={loss_dict['l0_norm']:.2f}, "
                        f"Sparsity={loss_dict['sparsity_ratio']:.4f}, "
                        f"{eta_str}"
                    )

                    if hasattr(pbar, "set_postfix"):
                        pbar.set_postfix(
                            {
                                "Loss": f"{avg_loss:.6f}",
                                "L0": f"{loss_dict['l0_norm']:.2f}",
                                "ETA": eta_str,
                            }
                        )

            # Checkpointing
            if step % config.save_every == 0:
                save_checkpoint(sae, optimizer, epoch, step, loss.item(), config, rank)
                dist.barrier()

    # Final checkpoint
    if rank == 0:
        save_checkpoint(sae, optimizer, epoch, step, loss.item(), config, rank)
        logger.info("Training completed!")

    cleanup_distributed()


def main():
    """Main entry point for SAE training."""
    config = SAETrainingConfig()

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/disk/onyx-scratch/glucc002/hf_cache"
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = "/disk/onyx-scratch/glucc002/hf_cache"

    logger.info("Starting SAE training on LLaDA activations")
    logger.info(f"Config: {config}")

    mp.spawn(
        train_sae, args=(config.world_size, config), nprocs=config.world_size, join=True
    )


if __name__ == "__main__":
    main()
