import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
import os
import time
import re

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "Google Generative AI not available. Install with: pip install google-generativeai"
    )


class ActivationAnalyzer:
    """Analyzer for SAE activation recordings."""

    def __init__(self, data_path: str):
        """Initialize with path to recorded activation data."""
        self.data_path = Path(data_path)
        self.data = None
        self.feature_activations = None
        self.config = None

        self._load_data()

    def _load_data(self):
        """Load the recorded activation data."""
        logger.info(f"Loading activation data from {self.data_path}")

        if self.data_path.suffix == ".pkl":
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        elif self.data_path.suffix == ".json":
            with open(self.data_path, "r") as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        self.feature_activations = self.data["feature_activations"]
        self.config = self.data.get("config", {})
        self.feature_labels = self.data.get("feature_labels", {})

        logger.info(f"Loaded data for {len(self.feature_activations)} features")
        if self.feature_labels:
            logger.info(f"Found {len(self.feature_labels)} existing feature labels")

    def get_feature_summary(self, feature_id: int) -> Dict:
        """Get summary statistics for a specific feature."""
        if (
            str(feature_id) not in self.feature_activations
            and feature_id not in self.feature_activations
        ):
            return {"error": f"Feature {feature_id} not found"}

        # Handle both string and int keys (JSON vs pickle)
        key = (
            str(feature_id)
            if str(feature_id) in self.feature_activations
            else feature_id
        )
        records = self.feature_activations[key]

        if not records:
            return {"feature_id": feature_id, "num_activations": 0}

        # Convert records to proper format if needed (from JSON)
        if isinstance(records[0], dict):
            from llada_mi.sae.record import ActivationRecord

            records = [ActivationRecord(**record) for record in records]

        # Calculate statistics
        activations = [r.activation_value for r in records]
        tokens = [r.token_str for r in records]
        contexts = [r.context_str for r in records]

        # Token frequency analysis
        token_counts = Counter(tokens)

        # Context analysis - find common words in contexts
        all_context_words = []
        for context in contexts:
            # Simple word extraction (could be improved with proper tokenization)
            words = context.lower().split()
            all_context_words.extend(words)

        context_word_counts = Counter(all_context_words)

        return {
            "feature_id": feature_id,
            "num_activations": len(records),
            "label": self.get_feature_label(feature_id),
            "activation_stats": {
                "min": min(activations),
                "max": max(activations),
                "mean": sum(activations) / len(activations),
                "median": sorted(activations)[len(activations) // 2],
            },
            "top_tokens": token_counts.most_common(10),
            "top_context_words": context_word_counts.most_common(20),
            "sample_activations": [
                {
                    "activation": r.activation_value,
                    "token": r.token_str,
                    "context": r.context_str[:200] + "..."
                    if len(r.context_str) > 200
                    else r.context_str,
                }
                for r in records[:5]  # Top 5 activations
            ],
        }

    def find_features_by_token(
        self, token: str, top_k: int = 10
    ) -> List[Tuple[int, float, int]]:
        """Find features that activate most strongly for a given token."""
        feature_scores = []

        for feature_id, records in self.feature_activations.items():
            if not records:
                continue

            # Convert records if needed
            if isinstance(records[0], dict):
                from llada_mi.sae.record import ActivationRecord

                records = [ActivationRecord(**record) for record in records]

            # Find activations for this token
            token_activations = [
                r.activation_value for r in records if r.token_str == token
            ]

            if token_activations:
                max_activation = max(token_activations)
                count = len(token_activations)
                feature_scores.append((int(feature_id), max_activation, count))

        # Sort by max activation value
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return feature_scores[:top_k]

    def find_features_by_context(
        self, context_word: str, top_k: int = 10
    ) -> List[Tuple[int, float, int]]:
        """Find features that activate in contexts containing a specific word."""
        feature_scores = []

        for feature_id, records in self.feature_activations.items():
            if not records:
                continue

            # Convert records if needed
            if isinstance(records[0], dict):
                from llada_mi.sae.record import ActivationRecord

                records = [ActivationRecord(**record) for record in records]

            # Find activations in contexts containing the word
            matching_activations = [
                r.activation_value
                for r in records
                if context_word.lower() in r.context_str.lower()
            ]

            if matching_activations:
                max_activation = max(matching_activations)
                count = len(matching_activations)
                feature_scores.append((int(feature_id), max_activation, count))

        # Sort by max activation value
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return feature_scores[:top_k]

    def get_top_features_by_activation(
        self, top_k: int = 20
    ) -> List[Tuple[int, float, int]]:
        """Get features with highest maximum activations."""
        feature_scores = []

        for feature_id, records in self.feature_activations.items():
            if not records:
                continue

            # Convert records if needed
            if isinstance(records[0], dict):
                from llada_mi.sae.record import ActivationRecord

                records = [ActivationRecord(**record) for record in records]

            activations = [r.activation_value for r in records]
            max_activation = max(activations)
            count = len(activations)
            feature_scores.append((int(feature_id), max_activation, count))

        # Sort by max activation value
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return feature_scores[:top_k]

    def analyze_feature_diversity(self) -> Dict:
        """Analyze the diversity of tokens and contexts for each feature."""
        diversity_stats = {}

        for feature_id, records in self.feature_activations.items():
            if not records:
                continue

            # Convert records if needed
            if isinstance(records[0], dict):
                from llada_mi.sae.record import ActivationRecord

                records = [ActivationRecord(**record) for record in records]

            tokens = [r.token_str for r in records]
            unique_tokens = set(tokens)

            # Simple context diversity: count unique words across all contexts
            all_context_words = set()
            for r in records:
                words = r.context_str.lower().split()
                all_context_words.update(words)

            diversity_stats[int(feature_id)] = {
                "num_activations": len(records),
                "unique_tokens": len(unique_tokens),
                "token_diversity_ratio": len(unique_tokens) / len(tokens)
                if tokens
                else 0,
                "unique_context_words": len(all_context_words),
                "most_common_token": Counter(tokens).most_common(1)[0]
                if tokens
                else None,
            }

        return diversity_stats

    def export_feature_summaries(self, output_dir: str, top_features: int = 100):
        """Export detailed summaries for top features."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get top features by activation
        top_feature_list = self.get_top_features_by_activation(top_features)

        logger.info(f"Exporting summaries for top {len(top_feature_list)} features...")

        # Export individual feature summaries
        for feature_id, max_activation, count in top_feature_list:
            summary = self.get_feature_summary(feature_id)

            with open(output_path / f"feature_{feature_id}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Export overall analysis
        overall_analysis = {
            "top_features": [
                {
                    "feature_id": fid,
                    "max_activation": max_act,
                    "num_records": count,
                    "label": self.get_feature_label(fid),
                }
                for fid, max_act, count in top_feature_list
            ],
            "diversity_stats": self.analyze_feature_diversity(),
            "config": self.config,
            "total_labeled_features": len(self.feature_labels),
        }

        with open(output_path / "overall_analysis.json", "w") as f:
            json.dump(overall_analysis, f, indent=2)

        logger.info(f"Exported summaries to {output_path}")

    def create_feature_report(self, feature_id: int) -> str:
        """Create a human-readable report for a specific feature."""
        summary = self.get_feature_summary(feature_id)

        if "error" in summary:
            return summary["error"]

        # Get label if available
        label = self.get_feature_label(feature_id)
        label_str = f" - {label}" if label else ""

        report = f"""
FEATURE {feature_id} ANALYSIS REPORT{label_str}
{"=" * 50}

Basic Statistics:
- Number of activations recorded: {summary["num_activations"]}
- Activation value range: {summary["activation_stats"]["min"]:.4f} - {summary["activation_stats"]["max"]:.4f}
- Mean activation: {summary["activation_stats"]["mean"]:.4f}
- Median activation: {summary["activation_stats"]["median"]:.4f}

Top Activating Tokens:
"""

        for token, count in summary["top_tokens"]:
            report += f"- '{token}': {count} occurrences\n"

        report += "\nMost Common Context Words:\n"
        for word, count in summary["top_context_words"][:10]:
            report += f"- '{word}': {count} occurrences\n"

        report += "\nTop 5 Activation Examples:\n"
        for i, example in enumerate(summary["sample_activations"], 1):
            report += f"\n{i}. Activation: {example['activation']:.4f}\n"
            report += f"   Token: '{example['token']}'\n"
            report += f"   Context: {example['context']}\n"

        return report

    def search_features(
        self, query: str, search_type: str = "token"
    ) -> List[Tuple[int, float, int]]:
        """Search for features based on tokens or context."""
        if search_type == "token":
            return self.find_features_by_token(query)
        elif search_type == "context":
            return self.find_features_by_context(query)
        else:
            raise ValueError(
                f"Invalid search_type: {search_type}. Use 'token' or 'context'"
            )

    def _create_labeling_prompt(self, feature_summary: Dict) -> str:
        """Create a prompt for the language model to label a feature."""
        feature_id = feature_summary["feature_id"]
        num_activations = feature_summary["num_activations"]

        # Get top tokens and contexts
        top_tokens = feature_summary["top_tokens"][:5]  # Top 5 tokens
        top_contexts = feature_summary["top_context_words"][:10]  # Top 10 context words
        sample_activations = feature_summary["sample_activations"][:3]  # Top 3 examples

        prompt = f"""You are analyzing a feature (neuron) from a Sparse Autoencoder trained on a language model. Your task is to provide a concise, interpretable label that describes what linguistic pattern or concept this feature detects.

Feature #{feature_id} Analysis:
- Total activations recorded: {num_activations}
- Activation strength range: {feature_summary["activation_stats"]["min"]:.3f} to {feature_summary["activation_stats"]["max"]:.3f}

Top tokens that activate this feature:
"""

        for token, count in top_tokens:
            prompt += f"- '{token}' (appears {count} times)\n"

        prompt += "\nMost common words in activation contexts:\n"
        for word, count in top_contexts:
            prompt += f"- '{word}' ({count} occurrences)\n"

        prompt += "\nTop activation examples with context:\n"
        for i, example in enumerate(sample_activations, 1):
            prompt += f"{i}. Activation: {example['activation']:.3f}\n"
            prompt += f"   Token: '{example['token']}'\n"
            prompt += f'   Context: "{example["context"]}"\n\n'

        prompt += """Based on this evidence, provide a concise label (2-10 words) that describes what this feature detects. Consider:
- What linguistic pattern, grammatical role, or semantic concept does it capture?
- What makes the contexts and tokens similar?
- Is it detecting specific words, grammatical structures, semantic concepts, or syntactic patterns?

Examples of good labels:
- "definite articles in formal text"
- "past tense verbs"
- "scientific terminology"
- "sentence beginnings"
- "mathematical expressions"
- "proper nouns"
- "conjunctions in lists"
- "unknown", in case the feature is not clear.

Respond with just the label, nothing else."""

        return prompt

    def _query_gemini(
        self, prompt: str, api_key: Optional[str] = None
    ) -> Optional[str]:
        """Query Gemini API for feature labeling."""
        if not GEMINI_AVAILABLE:
            logger.error(
                "Google Generative AI not available. Cannot use Gemini labeling."
            )
            return None

        # Get API key from parameter or environment
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            logger.error(
                "No Gemini API key provided. Set GEMINI_API_KEY environment variable or pass --gemini-api-key"
            )
            return None

        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent labeling
                    max_output_tokens=50,  # Short labels only
                ),
            )

            # Extract and clean the label
            label = response.text.strip()

            # Remove quotes if present
            label = re.sub(r'^["\']|["\']$', "", label)

            # Ensure it's not too long
            if len(label) > 100:
                label = label[:100] + "..."

            return label

        except Exception as e:
            logger.error(f"Error querying Gemini API: {e}")
            return None

    def label_features_with_gemini(
        self,
        api_key: Optional[str] = None,
        max_features: Optional[int] = None,
        skip_existing: bool = True,
        dead_threshold: float = 0.001,
    ) -> Dict[str, str]:
        """
        Use Gemini API to automatically label features.

        Args:
            api_key: Gemini API key (if not provided, uses GEMINI_API_KEY env var)
            max_features: Maximum number of features to label (None for all)
            skip_existing: Whether to skip features that already have labels
            dead_threshold: Maximum activation threshold below which neurons are labeled as "dead"

        Returns:
            Dictionary mapping feature IDs to their labels
        """
        if not GEMINI_AVAILABLE:
            logger.error(
                "Google Generative AI not available. Install with: pip install google-generativeai"
            )
            return {}

        logger.info("Starting automated feature labeling with Gemini...")

        # Get features to label
        features_to_label = []
        for feature_id, records in self.feature_activations.items():
            if not records:  # Skip empty features
                continue

            feature_id_str = str(feature_id)

            # Skip if already labeled and skip_existing is True
            if skip_existing and feature_id_str in self.feature_labels:
                continue

            features_to_label.append(int(feature_id))

        # Sort by activation strength (highest first)
        top_features = self.get_top_features_by_activation(len(features_to_label))
        feature_order = [fid for fid, _, _ in top_features]

        # Limit number of features if specified
        if max_features:
            feature_order = feature_order[:max_features]

        logger.info(f"Labeling {len(feature_order)} features...")

        new_labels = {}
        failed_count = 0
        dead_count = 0

        for i, feature_id in enumerate(feature_order, 1):
            logger.info(f"Labeling feature {feature_id} ({i}/{len(feature_order)})...")

            try:
                # Get feature summary
                summary = self.get_feature_summary(feature_id)
                if "error" in summary:
                    logger.warning(f"Skipping feature {feature_id}: {summary['error']}")
                    continue

                # Check if neuron is "dead" (max activation below threshold)
                max_activation = summary["activation_stats"]["max"]
                if max_activation < dead_threshold:
                    new_labels[str(feature_id)] = "dead"
                    dead_count += 1
                    logger.info(
                        f"Feature {feature_id}: 'dead' (max activation: {max_activation:.6f})"
                    )
                    continue

                # Create prompt
                prompt = self._create_labeling_prompt(summary)

                # Query Gemini
                label = self._query_gemini(prompt, api_key)

                if label:
                    new_labels[str(feature_id)] = label
                    logger.info(f"Feature {feature_id}: '{label}'")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to label feature {feature_id}")

                # Rate limiting - wait between requests
                if i < len(feature_order):  # Don't wait after the last request
                    time.sleep(1)  # 1 second between requests

            except Exception as e:
                failed_count += 1
                logger.error(f"Error labeling feature {feature_id}: {e}")

        # Update stored labels
        self.feature_labels.update(new_labels)

        logger.info(
            f"Labeling completed: {len(new_labels)} new labels ({dead_count} dead neurons), {failed_count} failures"
        )
        return new_labels

    def save_labels_to_file(self, output_path: Optional[str] = None):
        """
        Save the updated data with feature labels back to file.

        Args:
            output_path: Path to save to (if None, updates the original file)
        """
        if output_path is None:
            output_path = self.data_path

        # Update the data structure
        self.data["feature_labels"] = self.feature_labels

        output_path = Path(output_path)

        if output_path.suffix == ".pkl":
            with open(output_path, "wb") as f:
                pickle.dump(self.data, f)
        elif output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(self.data, f, indent=2)
        else:
            # Save as both formats
            pkl_path = output_path.with_suffix(".pkl")
            json_path = output_path.with_suffix(".json")

            with open(pkl_path, "wb") as f:
                pickle.dump(self.data, f)

            with open(json_path, "w") as f:
                json.dump(self.data, f, indent=2)

        logger.info(
            f"Saved updated data with {len(self.feature_labels)} labels to {output_path}"
        )

    def get_feature_label(self, feature_id: int) -> Optional[str]:
        """Get the label for a feature if it exists."""
        return self.feature_labels.get(str(feature_id))

    def get_labeled_features(self) -> Dict[int, str]:
        """Get all features that have labels."""
        return {int(fid): label for fid, label in self.feature_labels.items()}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze SAE activation recordings")

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to activation records file (.pkl or .json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--feature-id", type=int, default=None, help="Analyze specific feature ID"
    )
    parser.add_argument(
        "--search-token",
        type=str,
        default=None,
        help="Search for features that activate on specific token",
    )
    parser.add_argument(
        "--search-context",
        type=str,
        default=None,
        help="Search for features that activate in contexts with specific word",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=100,
        help="Number of top features to analyze",
    )
    parser.add_argument(
        "--export-summaries",
        action="store_true",
        help="Export detailed summaries for top features",
    )
    parser.add_argument(
        "--label-features",
        action="store_true",
        help="Use Gemini API to automatically label features",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--max-label-features",
        type=int,
        default=None,
        help="Maximum number of features to label (default: all)",
    )
    parser.add_argument(
        "--skip-existing-labels",
        action="store_true",
        default=True,
        help="Skip features that already have labels",
    )
    parser.add_argument(
        "--dead-threshold",
        type=float,
        default=0.001,
        help="Maximum activation threshold below which neurons are labeled as 'dead' (default: 0.001)",
    )

    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()

    # Load analyzer
    analyzer = ActivationAnalyzer(args.data_path)

    # Feature labeling with Gemini
    if args.label_features:
        new_labels = analyzer.label_features_with_gemini(
            api_key=args.gemini_api_key,
            max_features=args.max_label_features,
            skip_existing=args.skip_existing_labels,
            dead_threshold=args.dead_threshold,
        )

        if new_labels:
            analyzer.save_labels_to_file()
            print(f"\n✓ Successfully labeled {len(new_labels)} features")

            # Show some examples
            print("\nExample labels:")
            for feature_id, label in list(new_labels.items())[:5]:
                print(f"  Feature {feature_id}: {label}")
        else:
            print("❌ No features were labeled")
        return

    # Specific feature analysis
    if args.feature_id is not None:
        print(analyzer.create_feature_report(args.feature_id))
        return

    # Token search
    if args.search_token:
        results = analyzer.search_features(args.search_token, "token")
        print(f"\nTop features for token '{args.search_token}':")
        print("Feature ID | Max Activation | Count")
        print("-" * 40)
        for feature_id, max_act, count in results:
            print(f"{feature_id:9d} | {max_act:13.4f} | {count:5d}")
        return

    # Context search
    if args.search_context:
        results = analyzer.search_features(args.search_context, "context")
        print(f"\nTop features for context word '{args.search_context}':")
        print("Feature ID | Max Activation | Count")
        print("-" * 40)
        for feature_id, max_act, count in results:
            print(f"{feature_id:9d} | {max_act:13.4f} | {count:5d}")
        return

    # General analysis
    print("SAE ACTIVATION ANALYSIS")
    print("=" * 50)

    # Top features by activation
    top_features = analyzer.get_top_features_by_activation(20)
    print("\nTop 20 Features by Maximum Activation:")
    print("Feature ID | Max Activation | Records | Label")
    print("-" * 70)
    for feature_id, max_act, count in top_features:
        label = analyzer.get_feature_label(feature_id)
        label_str = label[:30] + "..." if label and len(label) > 30 else (label or "")
        print(f"{feature_id:9d} | {max_act:13.4f} | {count:7d} | {label_str}")

    # Export detailed summaries if requested
    if args.export_summaries:
        analyzer.export_feature_summaries(args.output_dir, args.top_features)
        print(f"\nDetailed summaries exported to {args.output_dir}")

    # Diversity analysis
    diversity_stats = analyzer.analyze_feature_diversity()
    active_features = [
        fid for fid, stats in diversity_stats.items() if stats["num_activations"] > 0
    ]

    if active_features:
        avg_diversity = sum(
            diversity_stats[fid]["token_diversity_ratio"] for fid in active_features
        ) / len(active_features)
        print("\nDiversity Statistics:")
        print(f"- Active features: {len(active_features)}")
        print(f"- Average token diversity ratio: {avg_diversity:.3f}")

        # Most and least diverse features
        most_diverse = max(
            active_features,
            key=lambda fid: diversity_stats[fid]["token_diversity_ratio"],
        )
        least_diverse = min(
            active_features,
            key=lambda fid: diversity_stats[fid]["token_diversity_ratio"],
        )

        print(
            f"- Most diverse feature: {most_diverse} (ratio: {diversity_stats[most_diverse]['token_diversity_ratio']:.3f})"
        )
        print(
            f"- Least diverse feature: {least_diverse} (ratio: {diversity_stats[least_diverse]['token_diversity_ratio']:.3f})"
        )


if __name__ == "__main__":
    main()
