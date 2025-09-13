"""
SAE Activation Recording Script for LLaDA Models

This script runs a trained SAE on LLaDA model activations and records the strongest
activations for each SAE feature/neuron, similar to the approach in "Towards Monosemanticity".

The script captures:
1. The strongest activations for each SAE feature across the dataset
2. The tokens that caused these activations (with context)
3. The final generated sentences after diffusion completes
4. Token positions and attention patterns

Usage:
    python -m llada_mi.sae.record --sae-path checkpoints/sae.pt --output-dir results/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
import time
from collections import defaultdict
import pickle

import torch
from tqdm import tqdm

from llada_mi.config import load_model_and_tokenizer
from llada_mi.sae.model import LLaDASAE
from llada_mi.sae.dataset import create_dataset
from llada_mi.inference import generate_with_logitlens

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ActivationRecord:
    """Record of a single activation event."""

    feature_id: int
    activation_value: float
    token_id: int
    token_str: str
    position: int  # Position in sequence
    context_tokens: List[int]  # Surrounding token context
    context_str: str  # Human-readable context
    input_sentence: str  # Original input sentence
    generated_sentence: str  # Final generated sentence after diffusion
    step_id: int  # Diffusion step where activation occurred
    layer_id: int  # Transformer layer
    batch_idx: int  # Batch index for debugging
    sequence_idx: int  # Sequence index within batch


@dataclass
class RecordingConfig:
    """Configuration for activation recording."""

    # Model and SAE paths
    model_name: str = "GSAI-ML/LLaDA-8B-Base"
    sae_checkpoint_path: str = "checkpoints/sae_training/checkpoint_step_1000.pt"

    # SAE parameters (should match training config)
    d_model: int = 4096
    d_sae: int = 16384
    k_sparse: int = 64
    target_layer: int = 16

    # Recording parameters
    top_k_activations: int = 10  # Top K activations to store per feature
    context_window: int = 10  # Tokens of context around activation
    max_samples: int = 1000  # Maximum number of samples to process

    # Dataset parameters
    dataset_type: str = "pile"  # "pile", "custom", "static"
    dataset_name: str = "monology/pile-uncopyrighted"
    dataset_path: Optional[str] = None  # Path to local dataset (if available)
    sequence_length: int = 512
    batch_size: int = 16

    # Generation parameters for full diffusion
    diffusion_steps: int = 12
    gen_length: int = 64
    temperature: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336

    # Output
    output_dir: str = "results/sae_activations"
    save_every: int = 100  # Save intermediate results every N batches

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActivationRecorder:
    """Records strongest activations for each SAE feature."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Storage for top activations per feature
        self.feature_activations: Dict[int, List[ActivationRecord]] = defaultdict(list)

        # Load models
        self._load_models()
        self._load_dataset()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized ActivationRecorder: {config.d_sae} features, "
            f"top-{config.top_k_activations} activations per feature"
        )

    def _load_models(self):
        """Load LLaDA model and trained SAE."""
        logger.info("Loading models...")
        self.model, self.tokenizer, _ = load_model_and_tokenizer(self.config.model_name)
        self.model = self.model.to(self.device).eval()

        # Load SAE checkpoint
        checkpoint = torch.load(
            self.config.sae_checkpoint_path, map_location=self.device
        )

        # Initialize SAE with same architecture as training
        self.sae = LLaDASAE(
            d_model=self.config.d_model,
            d_sae=self.config.d_sae,
            k_sparse=self.config.k_sparse,
        ).to(self.device)

        # Load SAE weights
        if "model_state_dict" in checkpoint:
            self.sae.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.sae.load_state_dict(checkpoint)

        # Convert SAE to same dtype as the model (typically bfloat16)
        self.sae = self.sae.to(dtype=self.model.dtype)
        self.sae.eval()
        logger.info(
            f"Models loaded: LLaDA + SAE from {self.config.sae_checkpoint_path}"
        )

    def _load_dataset(self):
        """Load dataset for activation recording."""
        dataset_kwargs = {
            "sequence_length": self.config.sequence_length,
        }

        if self.config.dataset_type == "pile":
            # Check if we have a local dataset path
            if self.config.dataset_path:
                dataset_kwargs.update(
                    {
                        "dataset_path": self.config.dataset_path,
                        "subset_fraction": min(
                            1.0, self.config.max_samples / 100000
                        ),  # Rough estimate
                        "streaming": False,  # Local datasets don't need streaming
                    }
                )
            else:
                dataset_kwargs.update(
                    {
                        "dataset_name": self.config.dataset_name,
                        "subset_fraction": min(
                            1.0, self.config.max_samples / 100000
                        ),  # Rough estimate
                        "streaming": True,
                    }
                )

        self.dataset = create_dataset(
            self.config.dataset_type, self.tokenizer, **dataset_kwargs
        )

        from torch.utils.data import DataLoader

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=2,
            pin_memory=True,
        )

        logger.info(
            f"Dataset loaded: {self.config.dataset_type}, batch_size={self.config.batch_size}"
        )

    def _extract_activations_and_context(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_idx: int
    ) -> Tuple[List[ActivationRecord], List[str]]:
        """
        Extract SAE activations and generate full sentences with diffusion.

        Returns:
            activations: List of activation records
            generated_sentences: List of fully generated sentences
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Extract hidden states from target layer during first diffusion step
        with torch.no_grad():
            # Create masked input for diffusion
            prompt_len = seq_len - self.config.gen_length
            x = input_ids.clone()
            x[:, prompt_len:] = self.config.mask_id  # Mask generation portion

            # Get hidden states from target layer
            outputs = self.model(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.config.target_layer]  # [B, T, D]

            # Step 2: Run SAE to get feature activations
            # Flatten for SAE processing
            hidden_flat, original_shape = self.sae.flatten_sequence_tensor(
                hidden_states
            )
            _, sparse_acts, _ = self.sae(hidden_flat)  # [B*T, F]

            # Unflatten back to sequence format
            sparse_acts = self.sae.unflatten_sequence_tensor(
                sparse_acts, original_shape
            )  # [B, T, F]

        # Step 3: Generate complete sentences using full diffusion process
        generated_sentences = []
        for i in range(batch_size):
            prompt = input_ids[i : i + 1, :prompt_len]  # Single sequence prompt

            try:
                # Generate with full diffusion
                generated_ids, _, _ = generate_with_logitlens(
                    self.model,
                    prompt,
                    steps=self.config.diffusion_steps,
                    gen_length=self.config.gen_length,
                    temperature=self.config.temperature,
                    remasking=self.config.remasking,
                    mask_id=self.config.mask_id,
                )

                # Decode generated sentence
                generated_text = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                generated_sentences.append(generated_text)

            except Exception as e:
                logger.warning(f"Generation failed for batch {batch_idx}, seq {i}: {e}")
                generated_sentences.append("[GENERATION_FAILED]")

        # Step 4: Process activations and create records
        activation_records = []

        for batch_i in range(batch_size):
            for pos in range(seq_len):
                for feature_id in range(self.config.d_sae):
                    activation_val = sparse_acts[batch_i, pos, feature_id].item()

                    # Only record non-zero activations (due to k-sparse constraint)
                    if activation_val > 0:
                        # Extract token and context
                        token_id = input_ids[batch_i, pos].item()
                        token_str = self.tokenizer.decode([token_id])

                        # Get context window
                        context_start = max(0, pos - self.config.context_window)
                        context_end = min(seq_len, pos + self.config.context_window + 1)
                        context_tokens = input_ids[
                            batch_i, context_start:context_end
                        ].tolist()
                        context_str = self.tokenizer.decode(
                            context_tokens, skip_special_tokens=True
                        )

                        # Get input sentence
                        input_sentence = self.tokenizer.decode(
                            input_ids[batch_i], skip_special_tokens=True
                        )

                        # Create activation record
                        record = ActivationRecord(
                            feature_id=feature_id,
                            activation_value=activation_val,
                            token_id=token_id,
                            token_str=token_str,
                            position=pos,
                            context_tokens=context_tokens,
                            context_str=context_str,
                            input_sentence=input_sentence,
                            generated_sentence=generated_sentences[batch_i],
                            step_id=0,  # First diffusion step
                            layer_id=self.config.target_layer,
                            batch_idx=batch_idx,
                            sequence_idx=batch_i,
                        )

                        activation_records.append(record)

        return activation_records, generated_sentences

    def _update_top_activations(self, new_records: List[ActivationRecord]):
        """Update the top-K activations for each feature."""
        for record in new_records:
            feature_id = record.feature_id

            # Add to feature's activation list
            self.feature_activations[feature_id].append(record)

            # Sort by activation value (descending) and keep top K
            self.feature_activations[feature_id].sort(
                key=lambda x: x.activation_value, reverse=True
            )
            self.feature_activations[feature_id] = self.feature_activations[feature_id][
                : self.config.top_k_activations
            ]

    def record_activations(self):
        """Main function to record activations across the dataset."""
        logger.info("Starting activation recording...")

        total_batches = 0
        total_activations = 0

        start_time = time.time()

        try:
            for batch_idx, batch in enumerate(
                tqdm(self.dataloader, desc="Recording activations")
            ):
                if batch_idx >= self.config.max_samples // self.config.batch_size:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Extract activations and generate sentences
                activation_records, generated_sentences = (
                    self._extract_activations_and_context(
                        input_ids, attention_mask, batch_idx
                    )
                )

                # Update top activations
                self._update_top_activations(activation_records)

                total_batches += 1
                total_activations += len(activation_records)

                # Save intermediate results
                if (batch_idx + 1) % self.config.save_every == 0:
                    self._save_intermediate_results(batch_idx + 1)

                # Log progress less frequently
                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Processed {batch_idx + 1} batches ({batches_per_sec:.1f}/s), "
                        f"{total_activations} activations recorded"
                    )

        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            raise

        # Final save
        self._save_final_results()

        logger.info(f"Recording completed. Processed {total_batches} batches")
        logger.info(f"Total activations recorded: {total_activations}")

        # Print summary statistics
        self._print_summary()

    def _save_intermediate_results(self, batch_num: int):
        """Save intermediate results."""
        output_path = (
            Path(self.config.output_dir) / f"intermediate_results_batch_{batch_num}.pkl"
        )

        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "config": asdict(self.config),
                    "feature_activations": dict(self.feature_activations),
                    "batch_num": batch_num,
                    "timestamp": time.time(),
                },
                f,
            )

        logger.debug(f"Intermediate results saved to {output_path}")

    def _save_final_results(self):
        """Save final results in multiple formats."""
        output_dir = Path(self.config.output_dir)

        # Save as pickle (complete data)
        pickle_path = output_dir / "activation_records.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(
                {
                    "config": asdict(self.config),
                    "feature_activations": dict(self.feature_activations),
                    "timestamp": time.time(),
                },
                f,
            )

        # Save as JSON (human-readable, but may be large)
        json_path = output_dir / "activation_records.json"
        json_data = {
            "config": asdict(self.config),
            "feature_activations": {
                str(feature_id): [asdict(record) for record in records]
                for feature_id, records in self.feature_activations.items()
            },
            "timestamp": time.time(),
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Save summary statistics
        summary_path = output_dir / "summary.json"
        summary = {
            "total_features": len(self.feature_activations),
            "features_with_activations": len(
                [
                    f
                    for f, records in self.feature_activations.items()
                    if len(records) > 0
                ]
            ),
            "total_activation_records": sum(
                len(records) for records in self.feature_activations.values()
            ),
            "config": asdict(self.config),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {output_dir} (pickle, JSON, summary)")

    def _print_summary(self):
        """Print summary statistics."""
        active_features = len(
            [f for f, records in self.feature_activations.items() if len(records) > 0]
        )
        total_records = sum(
            len(records) for records in self.feature_activations.values()
        )

        logger.info("=" * 50)
        logger.info("ACTIVATION RECORDING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total SAE features: {self.config.d_sae}")
        logger.info(f"Features with recorded activations: {active_features}")
        logger.info(f"Total activation records: {total_records}")
        logger.info(
            f"Average records per active feature: {total_records / max(active_features, 1):.2f}"
        )

        # Show top features by max activation
        if self.feature_activations:
            top_features = sorted(
                [
                    (fid, max(record.activation_value for record in records))
                    for fid, records in self.feature_activations.items()
                    if records
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            logger.info("\nTop 10 features by max activation:")
            for i, (feature_id, max_activation) in enumerate(top_features, 1):
                num_records = len(self.feature_activations[feature_id])
                logger.info(
                    f"  {i}. Feature {feature_id}: {max_activation:.4f} ({num_records} records)"
                )

        logger.info("=" * 50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Record SAE activations on LLaDA model"
    )

    # Model and paths
    parser.add_argument(
        "--model-name",
        type=str,
        default="GSAI-ML/LLaDA-8B-Base",
        help="LLaDA model name",
    )
    parser.add_argument(
        "--sae-checkpoint-path",
        type=str,
        required=True,
        help="Path to trained SAE checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sae_activations",
        help="Output directory for results",
    )

    # SAE parameters
    parser.add_argument(
        "--d-model", type=int, default=4096, help="Model hidden dimension"
    )
    parser.add_argument(
        "--d-sae", type=int, default=16384, help="SAE feature dimension"
    )
    parser.add_argument(
        "--k-sparse", type=int, default=64, help="Number of active features"
    )
    parser.add_argument(
        "--target-layer", type=int, default=16, help="Target transformer layer"
    )

    # Recording parameters
    parser.add_argument(
        "--top-k-activations",
        type=int,
        default=10,
        help="Top K activations to store per feature",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=10,
        help="Context window around activating token",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="pile",
        choices=["pile", "custom", "static"],
        help="Dataset type",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="monology/pile-uncopyrighted",
        help="Dataset name for Pile",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset directory (if available)",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=512, help="Sequence length"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    # Generation parameters
    parser.add_argument(
        "--diffusion-steps", type=int, default=12, help="Number of diffusion steps"
    )
    parser.add_argument("--gen-length", type=int, default=64, help="Generation length")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Generation temperature"
    )

    # Other
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save intermediate results every N batches",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create config
    config = RecordingConfig(
        model_name=args.model_name,
        sae_checkpoint_path=args.sae_checkpoint_path,
        d_model=args.d_model,
        d_sae=args.d_sae,
        k_sparse=args.k_sparse,
        target_layer=args.target_layer,
        top_k_activations=args.top_k_activations,
        context_window=args.context_window,
        max_samples=args.max_samples,
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps,
        gen_length=args.gen_length,
        temperature=args.temperature,
        output_dir=args.output_dir,
        save_every=args.save_every,
        device=device,
    )

    logger.info("Starting SAE activation recording")
    logger.info(f"Config: {config}")

    # Create recorder and run
    recorder = ActivationRecorder(config)
    recorder.record_activations()

    logger.info("Recording completed successfully!")


if __name__ == "__main__":
    main()
