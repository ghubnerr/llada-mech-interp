import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from llada_mi.sae.dataset import (
    PileDataset,
    StaticTextDataset,
    CustomTextDataset,
    create_dataset,
    get_pile_sample_data,
)


class TestPileDataset:
    """Test The Pile dataset functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        return tokenizer

    @pytest.fixture
    def mock_dataset_items(self):
        """Mock dataset items for testing."""
        return [
            {"text": "This is the first sample text for testing."},
            {"text": "Here is another sample with different content."},
            {"text": "Third sample with some more interesting content."},
            {"text": "Fourth sample for comprehensive testing."},
            {"text": "Final sample to complete the test dataset."},
        ]

    def test_pile_dataset_initialization(self, mock_tokenizer):
        """Test PileDataset initialization with valid parameters."""
        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_load.return_value = Mock()

            dataset = PileDataset(
                tokenizer=mock_tokenizer,
                sequence_length=512,
                subset_fraction=0.1,
                rank=0,
                world_size=1,
            )

            assert dataset.tokenizer is mock_tokenizer
            assert dataset.sequence_length == 512
            assert dataset.subset_fraction == 0.1
            assert dataset.rank == 0
            assert dataset.world_size == 1

            mock_load.assert_called_once()

    def test_pile_dataset_invalid_parameters(self, mock_tokenizer):
        """Test PileDataset initialization with invalid parameters."""
        with patch("llada_mi.sae.dataset.load_dataset"):
            # Invalid subset_fraction
            with pytest.raises(ValueError, match="subset_fraction must be in"):
                PileDataset(mock_tokenizer, subset_fraction=0.0)

            with pytest.raises(ValueError, match="subset_fraction must be in"):
                PileDataset(mock_tokenizer, subset_fraction=1.5)

            # Invalid sequence_length
            with pytest.raises(ValueError, match="sequence_length must be positive"):
                PileDataset(mock_tokenizer, sequence_length=0)

            with pytest.raises(ValueError, match="sequence_length must be positive"):
                PileDataset(mock_tokenizer, sequence_length=-10)

    def test_pile_dataset_iteration(self, mock_tokenizer, mock_dataset_items):
        """Test PileDataset iteration and tokenization."""
        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_dataset_items))
            mock_load.return_value = mock_dataset

            dataset = PileDataset(
                tokenizer=mock_tokenizer,
                sequence_length=512,
                subset_fraction=1.0,
                rank=0,
                world_size=1,
            )

            # Collect all items
            items = list(dataset)

            # Should have processed all items
            assert len(items) == len(mock_dataset_items)

            # Check tokenizer was called for each item
            assert mock_tokenizer.call_count == len(mock_dataset_items)

            # Check output format
            for item in items:
                assert "input_ids" in item
                assert "attention_mask" in item
                assert isinstance(item["input_ids"], torch.Tensor)
                assert isinstance(item["attention_mask"], torch.Tensor)

    def test_pile_dataset_distributed_sharding(
        self, mock_tokenizer, mock_dataset_items
    ):
        """Test distributed sharding in PileDataset."""
        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_dataset_items))
            mock_load.return_value = mock_dataset

            # Test rank 0 of 2 processes
            dataset_rank0 = PileDataset(
                tokenizer=mock_tokenizer,
                sequence_length=512,
                subset_fraction=1.0,
                rank=0,
                world_size=2,
            )

            items_rank0 = list(dataset_rank0)

            # Test rank 1 of 2 processes
            mock_dataset.__iter__ = Mock(return_value=iter(mock_dataset_items))
            dataset_rank1 = PileDataset(
                tokenizer=mock_tokenizer,
                sequence_length=512,
                subset_fraction=1.0,
                rank=1,
                world_size=2,
            )

            items_rank1 = list(dataset_rank1)

            # Each rank should process different subsets
            total_items = len(items_rank0) + len(items_rank1)
            assert total_items <= len(mock_dataset_items)

    def test_pile_dataset_get_sample_text(self, mock_tokenizer, mock_dataset_items):
        """Test getting sample texts from dataset."""
        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_dataset_items))
            mock_load.return_value = mock_dataset

            dataset = PileDataset(
                tokenizer=mock_tokenizer, sequence_length=512, subset_fraction=1.0
            )

            samples = dataset.get_sample_text(num_samples=3)

            assert len(samples) <= 3
            assert all(isinstance(sample, str) for sample in samples)

    def test_pile_dataset_error_handling(self, mock_tokenizer):
        """Test error handling in PileDataset."""
        # Test dataset loading failure
        with patch(
            "llada_mi.sae.dataset.load_dataset", side_effect=Exception("Load failed")
        ):
            with pytest.raises(Exception, match="Load failed"):
                PileDataset(mock_tokenizer)

    def test_pile_dataset_tokenization_error_handling(
        self, mock_tokenizer, mock_dataset_items
    ):
        """Test handling of tokenization errors."""
        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_dataset_items))
            mock_load.return_value = mock_dataset

            # Make tokenizer fail on some inputs
            def failing_tokenizer(text, **kwargs):
                if "first" in text:
                    raise Exception("Tokenization failed")
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                }

            mock_tokenizer.side_effect = failing_tokenizer

            dataset = PileDataset(
                tokenizer=mock_tokenizer, sequence_length=512, subset_fraction=1.0
            )

            # Should handle errors gracefully and continue
            items = list(dataset)

            # Should have fewer items due to tokenization failures
            assert len(items) < len(mock_dataset_items)


class TestStaticTextDataset:
    """Test static text dataset functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        return tokenizer

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "This is the first sample text.",
            "Here is the second sample text.",
            "Third sample with different content.",
            "Fourth text for testing purposes.",
            "Final sample text for the dataset.",
        ]

    def test_static_dataset_initialization(self, mock_tokenizer, sample_texts):
        """Test StaticTextDataset initialization."""
        dataset = StaticTextDataset(
            texts=sample_texts,
            tokenizer=mock_tokenizer,
            sequence_length=512,
            shuffle=False,
        )

        assert len(dataset) == len(sample_texts)
        assert dataset.sequence_length == 512
        assert not dataset.shuffle

    def test_static_dataset_shuffling(self, mock_tokenizer, sample_texts):
        """Test text shuffling in StaticTextDataset."""
        # Create dataset with shuffling
        dataset = StaticTextDataset(
            texts=sample_texts,
            tokenizer=mock_tokenizer,
            sequence_length=512,
            shuffle=True,
            seed=42,
        )

        assert len(dataset.texts) == len(sample_texts)
        assert set(dataset.texts) == set(sample_texts)

    def test_static_dataset_getitem(self, mock_tokenizer, sample_texts):
        """Test getting items from StaticTextDataset."""
        dataset = StaticTextDataset(
            texts=sample_texts,
            tokenizer=mock_tokenizer,
            sequence_length=512,
            shuffle=False,
        )

        item0 = dataset[0]
        assert "input_ids" in item0
        assert "attention_mask" in item0
        assert isinstance(item0["input_ids"], torch.Tensor)
        assert isinstance(item0["attention_mask"], torch.Tensor)

        last_item = dataset[len(dataset) - 1]
        assert "input_ids" in last_item
        assert "attention_mask" in last_item

    def test_static_dataset_iteration(self, mock_tokenizer, sample_texts):
        """Test iterating through StaticTextDataset."""
        dataset = StaticTextDataset(
            texts=sample_texts,
            tokenizer=mock_tokenizer,
            sequence_length=512,
            shuffle=False,
        )

        items = list(dataset)
        assert len(items) == len(sample_texts)

        for item in items:
            assert "input_ids" in item
            assert "attention_mask" in item

    def test_static_dataset_tokenization_failure(self, sample_texts):
        """Test handling of tokenization failures in StaticTextDataset."""
        failing_tokenizer = Mock()
        failing_tokenizer.side_effect = Exception("Tokenization failed")

        with patch("llada_mi.sae.dataset.logger"):
            dataset = StaticTextDataset(
                texts=sample_texts,
                tokenizer=failing_tokenizer,
                sequence_length=512,
                shuffle=False,
            )

            assert len(dataset.tokenized_data) == 0


class TestCustomTextDataset:
    """Test custom text dataset functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        return tokenizer

    @pytest.fixture
    def temp_text_files(self):
        """Create temporary text files for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create text files
        (temp_dir / "file1.txt").write_text("This is content from file 1.")
        (temp_dir / "file2.txt").write_text("Content from the second file.")

        # Create JSON file
        json_data = [{"text": "JSON content line 1"}, {"text": "JSON content line 2"}]
        with open(temp_dir / "data.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")

        yield temp_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    def test_custom_dataset_initialization(self, mock_tokenizer, temp_text_files):
        """Test CustomTextDataset initialization."""
        dataset = CustomTextDataset(
            data_path=temp_text_files,
            tokenizer=mock_tokenizer,
            sequence_length=512,
            file_extensions=[".txt"],
            rank=0,
            world_size=1,
        )

        assert dataset.data_path == temp_text_files
        assert dataset.tokenizer is mock_tokenizer
        assert dataset.sequence_length == 512
        assert ".txt" in dataset.file_extensions
        assert len(dataset.files) >= 2  # Should find text files

    def test_custom_dataset_file_not_found(self, mock_tokenizer):
        """Test CustomTextDataset with non-existent path."""
        with pytest.raises(FileNotFoundError):
            CustomTextDataset(data_path="/non/existent/path", tokenizer=mock_tokenizer)

    def test_custom_dataset_single_file(self, mock_tokenizer, temp_text_files):
        """Test CustomTextDataset with single file."""
        single_file = temp_text_files / "file1.txt"

        dataset = CustomTextDataset(
            data_path=single_file, tokenizer=mock_tokenizer, sequence_length=512
        )

        assert len(dataset.files) == 1
        assert dataset.files[0] == single_file

    def test_custom_dataset_iteration_txt(self, mock_tokenizer, temp_text_files):
        """Test CustomTextDataset iteration with text files."""
        dataset = CustomTextDataset(
            data_path=temp_text_files,
            tokenizer=mock_tokenizer,
            sequence_length=512,
            file_extensions=[".txt"],
        )

        items = list(dataset)

        # Should have items from text files
        assert len(items) >= 2

        for item in items:
            assert "input_ids" in item
            assert "attention_mask" in item

    def test_custom_dataset_iteration_json(self, mock_tokenizer, temp_text_files):
        """Test CustomTextDataset iteration with JSON files."""
        dataset = CustomTextDataset(
            data_path=temp_text_files,
            tokenizer=mock_tokenizer,
            sequence_length=512,
            file_extensions=[".json"],
            text_field="text",
        )

        items = list(dataset)

        # Should have items from JSON file
        assert len(items) >= 2

        for item in items:
            assert "input_ids" in item
            assert "attention_mask" in item

    def test_custom_dataset_distributed_sharding(self, mock_tokenizer, temp_text_files):
        """Test distributed file sharding in CustomTextDataset."""
        # Create more files for better sharding test
        for i in range(10):
            (temp_text_files / f"extra_{i}.txt").write_text(f"Extra content {i}")

        dataset_rank0 = CustomTextDataset(
            data_path=temp_text_files, tokenizer=mock_tokenizer, rank=0, world_size=2
        )

        dataset_rank1 = CustomTextDataset(
            data_path=temp_text_files, tokenizer=mock_tokenizer, rank=1, world_size=2
        )

        # Different ranks should process different files
        files_rank0 = set(dataset_rank0.files)
        files_rank1 = set(dataset_rank1.files)

        # No overlap between ranks
        assert len(files_rank0.intersection(files_rank1)) == 0


class TestDatasetFactory:
    """Test dataset factory functions."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        return Mock()

    def test_create_dataset_pile(self, mock_tokenizer):
        """Test creating PileDataset through factory."""
        with patch("llada_mi.sae.dataset.load_dataset"):
            dataset = create_dataset(
                dataset_type="pile", tokenizer=mock_tokenizer, sequence_length=512
            )

            assert isinstance(dataset, PileDataset)
            assert dataset.tokenizer is mock_tokenizer

    def test_create_dataset_static(self, mock_tokenizer):
        """Test creating StaticTextDataset through factory."""
        texts = ["Sample text 1", "Sample text 2"]

        dataset = create_dataset(
            dataset_type="static",
            tokenizer=mock_tokenizer,
            texts=texts,
            sequence_length=512,
        )

        assert isinstance(dataset, StaticTextDataset)
        assert dataset.tokenizer is mock_tokenizer

    def test_create_dataset_custom(self, mock_tokenizer):
        """Test creating CustomTextDataset through factory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "test.txt").write_text("Test content")

            dataset = create_dataset(
                dataset_type="custom",
                tokenizer=mock_tokenizer,
                data_path=temp_dir,
                sequence_length=512,
            )

            assert isinstance(dataset, CustomTextDataset)
            assert dataset.tokenizer is mock_tokenizer

    def test_create_dataset_invalid_type(self, mock_tokenizer):
        """Test creating dataset with invalid type."""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_dataset(dataset_type="invalid_type", tokenizer=mock_tokenizer)

    def test_get_pile_sample_data(self, mock_tokenizer):
        """Test getting sample data from The Pile."""
        # Mock the dataset loading
        mock_items = [
            {"text": "Sample text 1"},
            {"text": "Sample text 2"},
            {"text": "Sample text 3"},
        ]

        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_items))
            mock_load.return_value = mock_dataset

            # Mock the tokenizer for StaticTextDataset
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

            dataset = get_pile_sample_data(
                tokenizer=mock_tokenizer, num_samples=2, sequence_length=512
            )

            assert isinstance(dataset, StaticTextDataset)
            # Should limit to num_samples
            assert len(dataset) <= 2


class TestDatasetEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        return tokenizer

    def test_pile_dataset_empty_texts(self, mock_tokenizer):
        """Test PileDataset with empty or invalid texts."""
        mock_items = [
            {"text": ""},  # Empty text
            {"text": "   "},  # Whitespace only
            {"text": "ok"},  # Too short
            {"text": "This is a valid text sample for testing."},  # Valid
        ]

        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_items))
            mock_load.return_value = mock_dataset

            dataset = PileDataset(
                tokenizer=mock_tokenizer, sequence_length=512, subset_fraction=1.0
            )

            items = list(dataset)

            # Should only process the valid text
            assert len(items) == 1

    def test_pile_dataset_non_dict_items(self, mock_tokenizer):
        """Test PileDataset with non-dictionary items."""
        mock_items = [
            "string item",  # String instead of dict
            {"no_text_field": "value"},  # Dict without text field
            {"text": "Valid text sample for testing purposes"},  # Valid item
        ]

        with patch("llada_mi.sae.dataset.load_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_items))
            mock_load.return_value = mock_dataset

            dataset = PileDataset(
                tokenizer=mock_tokenizer, sequence_length=512, subset_fraction=1.0
            )

            items = list(dataset)

            # Should process the string item and the valid dict item (2 total)
            # The dict without text field should be skipped
            assert len(items) == 2

    def test_static_dataset_empty_texts(self, mock_tokenizer):
        """Test StaticTextDataset with empty text list."""
        dataset = StaticTextDataset(
            texts=[], tokenizer=mock_tokenizer, sequence_length=512
        )

        assert len(dataset) == 0

    def test_custom_dataset_corrupted_json(self, mock_tokenizer):
        """Test CustomTextDataset with corrupted JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create corrupted JSON file
            (temp_path / "corrupted.json").write_text("not valid json\n{invalid}")

            dataset = CustomTextDataset(
                data_path=temp_path, tokenizer=mock_tokenizer, file_extensions=[".json"]
            )

            # Should handle corrupted JSON gracefully
            items = list(dataset)
            assert len(items) == 0


if __name__ == "__main__":
    pytest.main([__file__])
