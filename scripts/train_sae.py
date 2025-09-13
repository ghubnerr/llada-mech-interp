"""
Simple script to run SAE training on LLaDA activations.

Usage:
    python train_sae.py

Or with custom parameters:
    python train_sae.py --batch-size 16 --learning-rate 1e-4 --d-sae 32768
"""

import argparse
import os
from llada_mi.sae.train import SAETrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SAE on LLaDA activations")

    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="GSAI-ML/LLaDA-8B-Base",
        help="LLaDA model name",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=16,
        help="Target transformer layer (0-indexed)",
    )
    parser.add_argument(
        "--target-step", type=int, default=0, help="Target diffusion step (0-indexed)"
    )

    # SAE architecture
    parser.add_argument(
        "--d-model", type=int, default=4096, help="Model hidden dimension"
    )
    parser.add_argument(
        "--d-sae", type=int, default=16384, help="SAE feature dimension"
    )
    parser.add_argument(
        "--k-sparse", type=int, default=64, help="Number of active features (k-sparse)"
    )
    parser.add_argument(
        "--tie-weights", action="store_true", help="Tie encoder and decoder weights"
    )
    parser.add_argument(
        "--no-normalize-decoder",
        action="store_true",
        help="Disable decoder weight normalization",
    )
    parser.add_argument(
        "--no-bias-decoder", action="store_true", help="Disable decoder bias"
    )
    parser.add_argument(
        "--l2-coefficient",
        type=float,
        default=1e-6,
        help="L2 regularization coefficient",
    )

    # Training parameters
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument(
        "--sequence-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--pile-subset",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.01 = 1%)",
    )

    # Distributed training
    parser.add_argument(
        "--world-size", type=int, default=8, help="Number of GPUs to use"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sae_training",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--log-every", type=int, default=100, help="Log progress every N steps"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum number of training steps"
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="monology/pile-uncopyrighted",
        help="Dataset name on HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset directory (if available)",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="train", help="Dataset split to use"
    )

    # Generation parameters
    parser.add_argument(
        "--mask-id", type=int, default=126336, help="Mask token ID for LLaDA"
    )
    parser.add_argument(
        "--gen-length", type=int, default=64, help="Generation length for diffusion"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="Remasking strategy",
    )

    return parser.parse_args()


def main_with_args():
    """Main function with command line argument parsing."""
    args = parse_args()

    # Create config from arguments
    config = SAETrainingConfig(
        model_name=args.model_name,
        target_layer=args.target_layer,
        target_step=args.target_step,
        d_model=args.d_model,
        d_sae=args.d_sae,
        k_sparse=args.k_sparse,
        tie_weights=args.tie_weights,
        normalize_decoder=not args.no_normalize_decoder,
        bias_decoder=not args.no_bias_decoder,
        l2_coefficient=args.l2_coefficient,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_epochs=args.num_epochs,
        pile_subset=args.pile_subset,
        world_size=args.world_size,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        max_steps=args.max_steps,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        temperature=args.temperature,
        remasking=args.remasking,
    )

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/disk/onyx-scratch/glucc002/hf_cache"
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = "/disk/onyx-scratch/glucc002/hf_cache"

    print("SAE Training Configuration:")
    print("=" * 50)
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    print("=" * 50)

    import torch.multiprocessing as mp
    from llada_mi.sae.train import train_sae

    mp.spawn(
        train_sae, args=(config.world_size, config), nprocs=config.world_size, join=True
    )


if __name__ == "__main__":
    main_with_args()
