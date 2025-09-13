import os
from pathlib import Path
from datasets import load_dataset
import argparse


def download_pile_subset(
    output_dir: str = "/disk/onyx-scratch/glucc002/pile_data",
    num_files: int = 5,
    subset_fraction: float = 0.01,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Pile dataset to: {output_path}")
    print(f"Target: {num_files} files (~{subset_fraction * 100:.1f}% of dataset)")

    try:
        print("Loading dataset in streaming mode...")
        dataset = load_dataset(
            "monology/pile-uncopyrighted", split="train", streaming=True
        )

        print(f"Taking first {num_files} files worth of data...")

        samples_per_file = int(4_100_000 / num_files)
        total_samples = samples_per_file * num_files

        print(f"Estimated {samples_per_file:,} samples per file")
        print(f"Total target: {total_samples:,} samples")

        dataset_subset = dataset.take(total_samples)

        print("Downloading samples with progress tracking...")
        from tqdm import tqdm
        from datasets import Dataset

        dataset_list = []
        progress_bar = tqdm(total=total_samples, desc="Downloading samples")

        for i, item in enumerate(dataset_subset):
            dataset_list.append(item)
            progress_bar.update(1)

            if (i + 1) % 10_000 == 0:
                progress_bar.set_description(f"Downloaded {i + 1:,}/{total_samples:,}")

        progress_bar.close()
        print(f"Successfully downloaded {len(dataset_list):,} samples")

        print("Saving dataset to disk...")
        dataset_dict = Dataset.from_list(dataset_list)
        dataset_dict.save_to_disk(str(output_path / "pile_subset"))

        print(f"Successfully saved {len(dataset_list):,} samples")
        print(f"Dataset saved to: {output_path / 'pile_subset'}")

        return str(output_path / "pile_subset")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("You may need to:")
        print("1. Check your internet connection")
        print("2. Verify HuggingFace credentials")
        print("3. Try again later if rate limited")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download Pile dataset locally")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/disk/onyx-scratch/glucc002/pile_data",
        help="Directory to save dataset",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=5,
        help="Number of Pile files to download (out of ~30)",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.01,
        help="Fraction of total dataset (0.01 = 1%)",
    )

    args = parser.parse_args()

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/disk/onyx-scratch/glucc002/hf_cache"
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = "/disk/onyx-scratch/glucc002/hf_cache"

    print("Pile Dataset Downloader")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of files: {args.num_files}")
    print(f"Subset fraction: {args.subset_fraction}")
    print("=" * 50)

    dataset_path = download_pile_subset(
        args.output_dir, args.num_files, args.subset_fraction
    )

    print("\n✅ Download complete!")
    print(f"Dataset path: {dataset_path}")
    print("\nTo use in training, update your config with:")
    print(f"  dataset_path: '{dataset_path}'")
    print("  streaming: False")


if __name__ == "__main__":
    main()
