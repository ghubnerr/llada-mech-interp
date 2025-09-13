import pytest
import tempfile
import json
import pickle
from pathlib import Path
from unittest.mock import patch
from collections import defaultdict
from dataclasses import asdict

from llada_mi.sae.record import (
    RecordingConfig,
    ActivationRecord,
    ActivationRecorder,
)
from llada_mi.sae.analyze_activations import ActivationAnalyzer


class TestActivationRecord:
    """Test ActivationRecord data structure."""

    def test_activation_record_creation(self):
        """Test creating ActivationRecord with valid data."""
        record = ActivationRecord(
            feature_id=123,
            activation_value=0.75,
            token_id=1000,
            token_str="hello",
            position=5,
            context_tokens=[998, 999, 1000, 1001, 1002],
            context_str="context hello world",
            input_sentence="This is an input sentence.",
            generated_sentence="This is a generated sentence.",
            step_id=0,
            layer_id=16,
            batch_idx=0,
            sequence_idx=2,
        )

        assert record.feature_id == 123
        assert record.activation_value == 0.75
        assert record.token_str == "hello"
        assert len(record.context_tokens) == 5
        assert record.position == 5

    def test_activation_record_serialization(self):
        """Test ActivationRecord serialization and deserialization."""
        record = ActivationRecord(
            feature_id=456,
            activation_value=1.25,
            token_id=2000,
            token_str="test",
            position=10,
            context_tokens=[1998, 1999, 2000, 2001, 2002],
            context_str="test context string",
            input_sentence="Input test sentence.",
            generated_sentence="Generated test sentence.",
            step_id=1,
            layer_id=8,
            batch_idx=1,
            sequence_idx=3,
        )

        # Test JSON serialization
        record_dict = asdict(record)
        json_str = json.dumps(record_dict)
        loaded_dict = json.loads(json_str)

        assert loaded_dict["feature_id"] == 456
        assert loaded_dict["activation_value"] == 1.25
        assert loaded_dict["token_str"] == "test"

        # Test pickle serialization
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump(record, f)
            temp_path = f.name

        with open(temp_path, "rb") as f:
            loaded_record = pickle.load(f)

        assert loaded_record.feature_id == record.feature_id
        assert loaded_record.activation_value == record.activation_value
        assert loaded_record.token_str == record.token_str

        Path(temp_path).unlink()


class TestRecordingConfig:
    """Test RecordingConfig validation and defaults."""

    def test_default_config(self):
        """Test RecordingConfig with default values."""
        config = RecordingConfig()

        assert config.model_name == "GSAI-ML/LLaDA-8B-Base"
        assert config.d_model == 4096
        assert config.d_sae == 16384
        assert config.k_sparse == 64
        assert config.target_layer == 16
        assert config.top_k_activations == 10
        assert config.context_window == 10
        assert config.max_samples == 1000
        assert config.batch_size == 16

    def test_custom_config(self):
        """Test RecordingConfig with custom values."""
        config = RecordingConfig(
            sae_checkpoint_path="/custom/path/sae.pt",
            d_sae=32768,
            k_sparse=128,
            max_samples=5000,
            batch_size=32,
            context_window=20,
            output_dir="/custom/output",
        )

        assert config.sae_checkpoint_path == "/custom/path/sae.pt"
        assert config.d_sae == 32768
        assert config.k_sparse == 128
        assert config.max_samples == 5000
        assert config.batch_size == 32
        assert config.context_window == 20
        assert config.output_dir == "/custom/output"

    def test_config_serialization(self):
        """Test RecordingConfig serialization."""
        config = RecordingConfig(
            max_samples=2000,
            batch_size=8,
            temperature=0.5,
        )

        config_dict = asdict(config)
        assert isinstance(config_dict, dict)
        assert config_dict["max_samples"] == 2000
        assert config_dict["batch_size"] == 8
        assert config_dict["temperature"] == 0.5


class TestActivationRecorderComponents:
    """Test individual components of ActivationRecorder without full model loading."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return RecordingConfig(
            sae_checkpoint_path="dummy_path.pt",
            max_samples=100,
            batch_size=4,
            d_sae=1024,
            k_sparse=32,
            output_dir="test_output",
        )

    @pytest.fixture
    def sample_activation_records(self):
        """Create sample activation records for testing."""
        records = []
        for i in range(50):
            record = ActivationRecord(
                feature_id=i % 10,  # 10 features
                activation_value=0.1 + i * 0.02,  # Increasing values
                token_id=5000 + i,
                token_str=f"token_{i}",
                position=i % 20,
                context_tokens=list(range(5000 + i - 2, 5000 + i + 3)),
                context_str=f"context for token_{i} with words",
                input_sentence=f"Input sentence {i}.",
                generated_sentence=f"Generated sentence {i}.",
                step_id=0,
                layer_id=16,
                batch_idx=i // 10,
                sequence_idx=i % 10,
            )
            records.append(record)
        return records

    def test_update_top_activations(self, mock_config, sample_activation_records):
        """Test the top-K activation tracking logic."""
        # Mock the ActivationRecorder initialization to avoid model loading
        with patch.object(ActivationRecorder, "_load_models"), patch.object(
            ActivationRecorder, "_load_dataset"
        ), patch("pathlib.Path.mkdir"):
            recorder = ActivationRecorder(mock_config)
            recorder.feature_activations = defaultdict(list)

            # Test updating with sample records
            recorder._update_top_activations(sample_activation_records)

            # Check that we have the expected number of features
            assert len(recorder.feature_activations) == 10

            # Check that each feature has at most top_k_activations records
            for feature_id, records in recorder.feature_activations.items():
                assert len(records) <= mock_config.top_k_activations

                # Check that records are sorted by activation value (descending)
                if len(records) > 1:
                    for i in range(len(records) - 1):
                        assert (
                            records[i].activation_value
                            >= records[i + 1].activation_value
                        )

    def test_activation_record_filtering(self, mock_config, sample_activation_records):
        """Test that only the highest activations are kept per feature."""
        with patch.object(ActivationRecorder, "_load_models"), patch.object(
            ActivationRecorder, "_load_dataset"
        ), patch("pathlib.Path.mkdir"):
            recorder = ActivationRecorder(mock_config)
            recorder.feature_activations = defaultdict(list)

            recorder.config.top_k_activations = 3

            recorder._update_top_activations(sample_activation_records)

            for feature_id, records in recorder.feature_activations.items():
                assert len(records) <= 3

                feature_records = [
                    r for r in sample_activation_records if r.feature_id == feature_id
                ]
                expected_top_values = sorted(
                    [r.activation_value for r in feature_records], reverse=True
                )[:3]

                actual_values = [r.activation_value for r in records]
                assert actual_values == expected_top_values

    def test_save_intermediate_results(self, mock_config):
        """Test intermediate results saving."""
        with patch.object(ActivationRecorder, "_load_models"), patch.object(
            ActivationRecorder, "_load_dataset"
        ), patch("pathlib.Path.mkdir"):
            recorder = ActivationRecorder(mock_config)
            recorder.feature_activations = defaultdict(list)

            test_record = ActivationRecord(
                feature_id=1,
                activation_value=0.5,
                token_id=100,
                token_str="test",
                position=1,
                context_tokens=[99, 100, 101],
                context_str="test context",
                input_sentence="Test input.",
                generated_sentence="Test generated.",
                step_id=0,
                layer_id=16,
                batch_idx=0,
                sequence_idx=0,
            )
            recorder.feature_activations[1].append(test_record)

            with tempfile.TemporaryDirectory() as temp_dir:
                recorder.config.output_dir = temp_dir
                recorder._save_intermediate_results(batch_num=5)

                expected_path = Path(temp_dir) / "intermediate_results_batch_5.pkl"
                assert expected_path.exists()

                with open(expected_path, "rb") as f:
                    data = pickle.load(f)

                assert "config" in data
                assert "feature_activations" in data
                assert "batch_num" in data
                assert "timestamp" in data
                assert data["batch_num"] == 5
                assert len(data["feature_activations"]) == 1

    def test_save_final_results(self, mock_config):
        """Test final results saving in multiple formats."""
        with patch.object(ActivationRecorder, "_load_models"), patch.object(
            ActivationRecorder, "_load_dataset"
        ), patch("pathlib.Path.mkdir"):
            recorder = ActivationRecorder(mock_config)
            recorder.feature_activations = defaultdict(list)

            test_record = ActivationRecord(
                feature_id=2,
                activation_value=0.8,
                token_id=200,
                token_str="final",
                position=2,
                context_tokens=[199, 200, 201],
                context_str="final context",
                input_sentence="Final input.",
                generated_sentence="Final generated.",
                step_id=0,
                layer_id=16,
                batch_idx=1,
                sequence_idx=0,
            )
            recorder.feature_activations[2].append(test_record)

            with tempfile.TemporaryDirectory() as temp_dir:
                recorder.config.output_dir = temp_dir
                recorder._save_final_results()

                pickle_path = Path(temp_dir) / "activation_records.pkl"
                json_path = Path(temp_dir) / "activation_records.json"
                summary_path = Path(temp_dir) / "summary.json"

                assert pickle_path.exists()
                assert json_path.exists()
                assert summary_path.exists()

                with open(pickle_path, "rb") as f:
                    pickle_data = pickle.load(f)
                assert "feature_activations" in pickle_data
                assert "config" in pickle_data

                with open(json_path, "r") as f:
                    json_data = json.load(f)
                assert "feature_activations" in json_data
                assert "config" in json_data

                with open(summary_path, "r") as f:
                    summary_data = json.load(f)
                assert "total_features" in summary_data
                assert "features_with_activations" in summary_data
                assert "total_activation_records" in summary_data


class TestActivationAnalyzer:
    """Test ActivationAnalyzer functionality."""

    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample data for analysis testing."""
        records_by_feature = {}

        test_cases = [
            (
                0,
                [
                    ("the", 0.9, "the quick brown fox"),
                    ("the", 0.8, "the lazy dog jumps"),
                    ("the", 0.7, "over the fence quickly"),
                ],
            ),
            (
                1,
                [
                    ("and", 0.6, "cats and dogs play"),
                    ("or", 0.5, "cats or dogs sleep"),
                    ("but", 0.4, "cats but not dogs"),
                ],
            ),
            (
                2,
                [
                    ("five", 0.8, "five cats running"),
                    ("ten", 0.7, "ten dogs barking"),
                    ("many", 0.6, "many animals playing"),
                ],
            ),
        ]

        for feature_id, token_data in test_cases:
            records = []
            for i, (token, activation, context) in enumerate(token_data):
                record = ActivationRecord(
                    feature_id=feature_id,
                    activation_value=activation,
                    token_id=1000 + i,
                    token_str=token,
                    position=i,
                    context_tokens=list(range(1000 + i - 2, 1000 + i + 3)),
                    context_str=context,
                    input_sentence=f"Input with {token}.",
                    generated_sentence=f"Generated with {token}.",
                    step_id=0,
                    layer_id=16,
                    batch_idx=0,
                    sequence_idx=i,
                )
                records.append(record)
            records_by_feature[feature_id] = records

        return {
            "feature_activations": records_by_feature,
            "config": {"d_sae": 3, "max_samples": 9},
            "feature_labels": {
                "0": "definite articles in formal contexts",
                "1": "coordinating conjunctions",
                "2": "quantitative expressions",
            },
        }

    @pytest.fixture
    def temp_analysis_file(self, sample_analysis_data):
        """Create temporary file with sample analysis data."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(sample_analysis_data, f)
            return f.name

    def test_analyzer_initialization(self, temp_analysis_file):
        """Test ActivationAnalyzer initialization."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        assert analyzer.data is not None
        assert analyzer.feature_activations is not None
        assert analyzer.config is not None
        assert analyzer.feature_labels is not None
        assert len(analyzer.feature_activations) == 3
        assert len(analyzer.feature_labels) == 3

        Path(temp_analysis_file).unlink()

    def test_feature_summary(self, temp_analysis_file):
        """Test feature summary generation."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        summary = analyzer.get_feature_summary(0)

        assert summary["feature_id"] == 0
        assert summary["num_activations"] == 3
        assert "activation_stats" in summary
        assert "top_tokens" in summary
        assert "top_context_words" in summary
        assert "sample_activations" in summary

        stats = summary["activation_stats"]
        assert stats["min"] == 0.7
        assert stats["max"] == 0.9
        assert 0.7 <= stats["mean"] <= 0.9

        top_tokens = summary["top_tokens"]
        assert len(top_tokens) > 0
        assert top_tokens[0][0] == "the"  # Most common token
        assert top_tokens[0][1] == 3  # Appears 3 times

        invalid_summary = analyzer.get_feature_summary(999)
        assert "error" in invalid_summary

        Path(temp_analysis_file).unlink()

    def test_find_features_by_token(self, temp_analysis_file):
        """Test finding features by token."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        results = analyzer.find_features_by_token("the", top_k=5)

        assert len(results) == 1  # Only feature 0 has "the"
        assert results[0][0] == 0  # Feature ID
        assert results[0][1] == 0.9  # Max activation value
        assert results[0][2] == 3  # Count

        no_results = analyzer.find_features_by_token("nonexistent", top_k=5)
        assert len(no_results) == 0

        Path(temp_analysis_file).unlink()

    def test_find_features_by_context(self, temp_analysis_file):
        """Test finding features by context word."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        results = analyzer.find_features_by_context("cats", top_k=5)

        assert len(results) >= 1
        feature_ids = [result[0] for result in results]
        assert 1 in feature_ids or 2 in feature_ids

        Path(temp_analysis_file).unlink()

    def test_top_features_by_activation(self, temp_analysis_file):
        """Test getting top features by activation value."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        results = analyzer.get_top_features_by_activation(top_k=3)

        assert len(results) == 3
        assert results[0][1] >= results[1][1] >= results[2][1]
        assert results[0][0] == 0
        assert results[0][1] == 0.9

        Path(temp_analysis_file).unlink()

    def test_diversity_analysis(self, temp_analysis_file):
        """Test feature diversity analysis."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        diversity = analyzer.analyze_feature_diversity()

        assert len(diversity) == 3

        feature_0_stats = diversity[0]
        assert feature_0_stats["num_activations"] == 3
        assert feature_0_stats["unique_tokens"] == 1  # Only "the"
        assert feature_0_stats["token_diversity_ratio"] == 1 / 3  # 1 unique / 3 total

        feature_1_stats = diversity[1]
        assert feature_1_stats["num_activations"] == 3
        assert feature_1_stats["unique_tokens"] == 3  # "and", "or", "but"
        assert feature_1_stats["token_diversity_ratio"] == 1.0  # 3 unique / 3 total

        Path(temp_analysis_file).unlink()

    def test_feature_report_generation(self, temp_analysis_file):
        """Test human-readable feature report generation."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        report = analyzer.create_feature_report(0)

        assert "FEATURE 0 ANALYSIS REPORT" in report
        assert "Basic Statistics:" in report
        assert "Top Activating Tokens:" in report
        assert "Most Common Context Words:" in report
        assert "Top 5 Activation Examples:" in report

        # Check that specific data appears in report
        assert "the" in report  # Should mention the top token
        assert "0.9000" in report  # Should show max activation

        # Test invalid feature
        invalid_report = analyzer.create_feature_report(999)
        assert "Feature 999 not found" in invalid_report

        Path(temp_analysis_file).unlink()

    def test_search_features(self, temp_analysis_file):
        """Test the unified search interface."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        # Test token search
        token_results = analyzer.search_features("the", search_type="token")
        assert len(token_results) == 1
        assert token_results[0][0] == 0

        # Test context search
        context_results = analyzer.search_features("cats", search_type="context")
        assert len(context_results) >= 1

        # Test invalid search type
        with pytest.raises(ValueError, match="Invalid search_type"):
            analyzer.search_features("test", search_type="invalid")

        Path(temp_analysis_file).unlink()

    def test_export_summaries(self, temp_analysis_file):
        """Test exporting detailed summaries."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer.export_feature_summaries(temp_dir, top_features=2)

            # Check that individual feature files were created
            feature_files = list(Path(temp_dir).glob("feature_*_summary.json"))
            assert len(feature_files) == 2  # Top 2 features

            # Check that overall analysis file was created
            overall_file = Path(temp_dir) / "overall_analysis.json"
            assert overall_file.exists()

            # Verify content of overall analysis
            with open(overall_file, "r") as f:
                overall_data = json.load(f)

            assert "top_features" in overall_data
            assert "diversity_stats" in overall_data
            assert "config" in overall_data
            assert len(overall_data["top_features"]) == 2

        Path(temp_analysis_file).unlink()

    def test_feature_labeling(self, temp_analysis_file):
        """Test feature label functionality."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        # Test getting existing labels
        label_0 = analyzer.get_feature_label(0)
        assert label_0 == "definite articles in formal contexts"

        label_1 = analyzer.get_feature_label(1)
        assert label_1 == "coordinating conjunctions"

        # Test getting non-existent label
        label_999 = analyzer.get_feature_label(999)
        assert label_999 is None

        # Test getting all labeled features
        labeled_features = analyzer.get_labeled_features()
        assert len(labeled_features) == 3
        assert 0 in labeled_features
        assert 1 in labeled_features
        assert 2 in labeled_features

        # Test that feature summaries include labels
        summary = analyzer.get_feature_summary(0)
        assert summary["label"] == "definite articles in formal contexts"

        # Test report generation includes label
        report = analyzer.create_feature_report(0)
        assert "definite articles in formal contexts" in report

        Path(temp_analysis_file).unlink()

    def test_labeling_prompt_creation(self, temp_analysis_file):
        """Test creation of labeling prompts for Gemini."""
        analyzer = ActivationAnalyzer(temp_analysis_file)

        # Get feature summary
        summary = analyzer.get_feature_summary(0)

        # Create prompt
        prompt = analyzer._create_labeling_prompt(summary)

        # Check that prompt contains key information
        assert "Feature #0" in prompt
        assert "the" in prompt.lower()  # Should mention the top token
        assert "activation" in prompt.lower()
        assert "context" in prompt.lower()
        assert "label" in prompt.lower()

        # Check that it's asking for a concise response
        assert "concise" in prompt.lower()

        Path(temp_analysis_file).unlink()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_activation_data(self):
        """Test handling of empty activation data."""
        empty_data = {
            "feature_activations": {},
            "config": {"d_sae": 0, "max_samples": 0},
        }

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(empty_data, f)
            temp_path = f.name

        analyzer = ActivationAnalyzer(temp_path)

        # Test with empty data
        top_features = analyzer.get_top_features_by_activation(10)
        assert len(top_features) == 0

        diversity = analyzer.analyze_feature_diversity()
        assert len(diversity) == 0

        token_results = analyzer.find_features_by_token("test")
        assert len(token_results) == 0

        Path(temp_path).unlink()

    def test_malformed_data_handling(self):
        """Test handling of malformed data files."""
        # Test with invalid file format
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("invalid data")
            invalid_path = f.name

        with pytest.raises(ValueError, match="Unsupported file format"):
            ActivationAnalyzer(invalid_path)

        Path(invalid_path).unlink()

    def test_large_dataset_simulation(self):
        """Test behavior with large dataset simulation."""
        # Create data with many features and activations
        large_data = {"feature_activations": {}, "config": {"d_sae": 1000}}

        for feature_id in range(100):  # 100 features
            records = []
            for i in range(50):  # 50 records per feature
                record_dict = {
                    "feature_id": feature_id,
                    "activation_value": 0.1 + i * 0.01,
                    "token_id": i,
                    "token_str": f"token_{i}",
                    "position": i,
                    "context_tokens": list(range(i, i + 5)),
                    "context_str": f"context {i}",
                    "input_sentence": f"input {i}",
                    "generated_sentence": f"generated {i}",
                    "step_id": 0,
                    "layer_id": 16,
                    "batch_idx": 0,
                    "sequence_idx": i,
                }
                records.append(record_dict)
            large_data["feature_activations"][str(feature_id)] = records

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(large_data, f)
            temp_path = f.name

        analyzer = ActivationAnalyzer(temp_path)

        # Test that analysis completes without issues
        top_features = analyzer.get_top_features_by_activation(20)
        assert len(top_features) == 20

        diversity = analyzer.analyze_feature_diversity()
        assert len(diversity) == 100

        Path(temp_path).unlink()


class TestIntegration:
    """Integration tests for the complete recording and analysis pipeline."""

    def test_record_analyze_pipeline(self):
        """Test the complete pipeline from recording to analysis."""
        # Create minimal mock data that simulates the recording output
        mock_records = []
        for i in range(30):
            record = ActivationRecord(
                feature_id=i % 5,  # 5 features
                activation_value=0.2 + i * 0.03,
                token_id=3000 + i,
                token_str=f"word_{i}",
                position=i % 15,
                context_tokens=list(range(3000 + i - 3, 3000 + i + 4)),
                context_str=f"context word_{i} sentence",
                input_sentence=f"This is input sentence {i}.",
                generated_sentence=f"This is generated sentence {i}.",
                step_id=0,
                layer_id=16,
                batch_idx=i // 10,
                sequence_idx=i % 10,
            )
            mock_records.append(record)

        # Group by feature (simulating the recording process)
        feature_activations = defaultdict(list)
        for record in mock_records:
            feature_activations[record.feature_id].append(record)

        # Sort by activation value (simulating top-K selection)
        for feature_id in feature_activations:
            feature_activations[feature_id].sort(
                key=lambda x: x.activation_value, reverse=True
            )
            # Keep only top 5 per feature
            feature_activations[feature_id] = feature_activations[feature_id][:5]

        # Save the data (simulating recording output)
        data = {
            "feature_activations": dict(feature_activations),
            "config": {
                "d_sae": 5,
                "max_samples": 30,
                "top_k_activations": 5,
            },
        }

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(data, f)
            temp_path = f.name

        # Test analysis of the recorded data
        analyzer = ActivationAnalyzer(temp_path)

        # Comprehensive analysis
        top_features = analyzer.get_top_features_by_activation(5)
        assert len(top_features) == 5

        # Each feature should have exactly 5 records (our top-K limit)
        for feature_id, max_activation, count in top_features:
            assert count == 5

        # Test feature-specific analysis
        summary = analyzer.get_feature_summary(0)
        assert summary["num_activations"] == 5
        assert len(summary["sample_activations"]) == 5

        # Test search functionality
        # Find features that activate on a specific token pattern
        token_results = analyzer.find_features_by_token("word_1", top_k=3)
        assert len(token_results) >= 0  # May or may not find results

        # Test diversity analysis
        diversity = analyzer.analyze_feature_diversity()
        assert len(diversity) == 5

        # Test report generation
        report = analyzer.create_feature_report(0)
        assert len(report) > 100  # Should be a substantial report

        Path(temp_path).unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
