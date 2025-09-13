import logging
from typing import Dict, Iterator, List, Optional, Union
from pathlib import Path
import json

import torch
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PileDataset(IterableDataset):
    """
    Streaming dataset for The Pile that yields tokenized sequences.

    This dataset streams data from The Pile dataset, tokenizes text sequences,
    and yields them in batches suitable for SAE training on LLaDA activations.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sequence_length: int = 512,
        subset_fraction: float = 1.0,
        dataset_name: str = "monology/pile-uncopyrighted",
        dataset_path: Optional[str] = None,
        dataset_split: str = "train",
        streaming: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize The Pile dataset.

        Args:
            tokenizer: Tokenizer to use for text processing
            sequence_length: Maximum sequence length for tokenization
            subset_fraction: Fraction of dataset to use (0.0 to 1.0)
            dataset_name: HuggingFace dataset name
            dataset_path: Path to local dataset directory (if available)
            dataset_split: Dataset split to use
            streaming: Whether to use streaming mode
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Random seed for reproducibility
            cache_dir: Directory for caching dataset
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.subset_fraction = subset_fraction
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.streaming = streaming
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.cache_dir = cache_dir

        # Validate parameters
        if not 0.0 < subset_fraction <= 1.0:
            raise ValueError(
                f"subset_fraction must be in (0.0, 1.0], got {subset_fraction}"
            )

        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the dataset from HuggingFace or local path."""
        try:
            if self.dataset_path:
                # Load from local dataset
                logger.info(f"Loading local dataset from: {self.dataset_path}")
                from datasets import load_from_disk

                self.dataset = load_from_disk(self.dataset_path)
                if self.dataset_split != "train":
                    self.dataset = self.dataset[self.dataset_split]
            else:
                # Load from HuggingFace
                logger.info(
                    f"Loading dataset {self.dataset_name} (split: {self.dataset_split})"
                )

                load_kwargs = {
                    "path": self.dataset_name,
                    "split": self.dataset_split,
                    "streaming": self.streaming,
                }

                if self.cache_dir:
                    load_kwargs["cache_dir"] = self.cache_dir

                self.dataset = load_dataset(**load_kwargs)

            # Apply distributed sharding if needed
            if self.streaming and self.world_size > 1:
                # For streaming datasets, we'll handle sharding in __iter__
                pass
            elif not self.streaming and self.world_size > 1:
                # For non-streaming datasets, shard the dataset
                total_size = len(self.dataset)
                shard_size = total_size // self.world_size
                start_idx = self.rank * shard_size
                end_idx = (
                    (self.rank + 1) * shard_size
                    if self.rank < self.world_size - 1
                    else total_size
                )

                self.dataset = self.dataset.select(range(start_idx, end_idx))

            logger.info(f"Dataset loaded successfully on rank {self.rank}")

        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset, yielding tokenized sequences."""
        count = 0

        # Calculate maximum samples based on subset fraction
        if hasattr(self.dataset, "__len__") and not self.streaming:
            max_samples = int(len(self.dataset) * self.subset_fraction)
        else:
            # For streaming datasets, we'll approximate based on known Pile size
            estimated_pile_size = 825000000  # Approximate number of documents in Pile
            max_samples = int(
                estimated_pile_size * self.subset_fraction / self.world_size
            )

        logger.info(f"Rank {self.rank}: Processing up to {max_samples} samples")

        try:
            for item in self.dataset:
                if count >= max_samples:
                    break

                # Skip samples for distributed training
                if self.streaming and count % self.world_size != self.rank:
                    count += 1
                    continue

                # Extract text content
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                elif isinstance(item, str):
                    text = item
                else:
                    logger.warning(f"Unexpected item format: {type(item)}")
                    count += 1
                    continue

                # Skip empty or very short texts
                if not text or len(text.strip()) < 10:
                    count += 1
                    continue

                # Tokenize the text
                try:
                    tokens = self.tokenizer(
                        text,
                        max_length=self.sequence_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    yield {
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                    }

                except Exception as e:
                    logger.warning(f"Tokenization failed for sample {count}: {e}")
                    count += 1
                    continue

                count += 1

        except Exception as e:
            logger.error(f"Error during dataset iteration: {e}")
            raise

        logger.info(f"Rank {self.rank}: Processed {count} samples")

    def get_sample_text(self, num_samples: int = 5) -> List[str]:
        """
        Get sample texts from the dataset for inspection.

        Args:
            num_samples: Number of sample texts to return

        Returns:
            List of sample text strings
        """
        samples = []
        count = 0

        try:
            for item in self.dataset:
                if count >= num_samples:
                    break

                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                elif isinstance(item, str):
                    text = item
                else:
                    continue

                if text and len(text.strip()) > 10:
                    samples.append(text[:500] + "..." if len(text) > 500 else text)
                    count += 1

        except Exception as e:
            logger.warning(f"Error getting sample texts: {e}")

        return samples


class StaticTextDataset(Dataset):
    """
    Static dataset for pre-loaded text data.

    This dataset is useful for testing and when you have a fixed set of texts
    that you want to process multiple times.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        sequence_length: int = 512,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize static text dataset.

        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use for text processing
            sequence_length: Maximum sequence length for tokenization
            shuffle: Whether to shuffle the texts
            seed: Random seed for shuffling
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            torch.manual_seed(seed)
            indices = torch.randperm(len(texts))
            self.texts = [texts[i] for i in indices]

        # Pre-tokenize all texts
        self._tokenize_texts()

    def _tokenize_texts(self):
        """Pre-tokenize all texts in the dataset."""
        logger.info(f"Pre-tokenizing {len(self.texts)} texts...")

        self.tokenized_data = []

        for i, text in enumerate(self.texts):
            if i % 1000 == 0:
                logger.info(f"Tokenized {i}/{len(self.texts)} texts")

            try:
                tokens = self.tokenizer(
                    text,
                    max_length=self.sequence_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                self.tokenized_data.append(
                    {
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                    }
                )

            except Exception as e:
                logger.warning(f"Tokenization failed for text {i}: {e}")
                continue

        logger.info(f"Successfully tokenized {len(self.tokenized_data)} texts")

    def __len__(self) -> int:
        """Return the number of tokenized samples."""
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized sample by index."""
        return self.tokenized_data[idx]


class CustomTextDataset(IterableDataset):
    """
    Custom dataset for loading text from files or directories.

    This dataset can load text from various sources and formats.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        sequence_length: int = 512,
        file_extensions: List[str] = [".txt", ".json"],
        text_field: Optional[str] = None,  # For JSON files
        max_files: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize custom text dataset.

        Args:
            data_path: Path to file or directory containing text data
            tokenizer: Tokenizer to use for text processing
            sequence_length: Maximum sequence length for tokenization
            file_extensions: List of file extensions to process
            text_field: Field name for text in JSON files
            max_files: Maximum number of files to process
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.file_extensions = file_extensions
        self.text_field = text_field
        self.max_files = max_files
        self.rank = rank
        self.world_size = world_size

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        # Get list of files to process
        self._get_file_list()

    def _get_file_list(self):
        """Get list of files to process."""
        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            self.files = []
            for ext in self.file_extensions:
                self.files.extend(self.data_path.glob(f"*{ext}"))
                self.files.extend(self.data_path.glob(f"**/*{ext}"))

        # Remove duplicates and sort
        self.files = sorted(list(set(self.files)))

        # Apply max_files limit
        if self.max_files:
            self.files = self.files[: self.max_files]

        # Shard files for distributed processing
        if self.world_size > 1:
            self.files = [
                f for i, f in enumerate(self.files) if i % self.world_size == self.rank
            ]

        logger.info(f"Rank {self.rank}: Processing {len(self.files)} files")

    def _read_file(self, file_path: Path) -> Iterator[str]:
        """Read text content from a file."""
        try:
            if file_path.suffix == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        yield content

            elif file_path.suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if self.text_field and self.text_field in data:
                                text = data[self.text_field]
                            elif "text" in data:
                                text = data["text"]
                            else:
                                continue

                            if text and text.strip():
                                yield text.strip()

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through files and yield tokenized sequences."""
        for file_path in self.files:
            logger.debug(f"Processing file: {file_path}")

            for text in self._read_file(file_path):
                # Skip empty or very short texts
                if not text or len(text.strip()) < 10:
                    continue

                try:
                    tokens = self.tokenizer(
                        text,
                        max_length=self.sequence_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    yield {
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                    }

                except Exception as e:
                    logger.warning(f"Tokenization failed: {e}")
                    continue


def create_dataset(
    dataset_type: str, tokenizer: PreTrainedTokenizer, **kwargs
) -> Union[PileDataset, StaticTextDataset, CustomTextDataset]:
    """
    Factory function to create datasets.

    Args:
        dataset_type: Type of dataset ("pile", "static", "custom")
        tokenizer: Tokenizer to use
        **kwargs: Additional arguments for dataset initialization

    Returns:
        Dataset instance
    """
    if dataset_type == "pile":
        return PileDataset(tokenizer, **kwargs)
    elif dataset_type == "static":
        return StaticTextDataset(tokenizer=tokenizer, **kwargs)
    elif dataset_type == "custom":
        return CustomTextDataset(tokenizer=tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_pile_sample_data(
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 100,
    sequence_length: int = 512,
    cache_dir: Optional[str] = None,
) -> StaticTextDataset:
    """
    Get a small sample of The Pile for testing purposes.

    Args:
        tokenizer: Tokenizer to use
        num_samples: Number of samples to extract
        sequence_length: Maximum sequence length
        cache_dir: Cache directory for dataset

    Returns:
        StaticTextDataset with sample data
    """
    logger.info(f"Loading {num_samples} samples from The Pile for testing")

    # Load a small subset of the Pile
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    # Extract sample texts
    texts = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break

        if "text" in item and item["text"]:
            texts.append(item["text"])

    logger.info(f"Extracted {len(texts)} sample texts")

    return StaticTextDataset(
        texts=texts,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        shuffle=True,
    )
