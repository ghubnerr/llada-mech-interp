import sys
from pathlib import Path
import tempfile
import json
import pytest

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_labeling_functionality():
    try:
        from llada_mi.sae.analyze_activations import ActivationAnalyzer

        mock_data = {
            "feature_activations": {
                "0": [
                    {
                        "feature_id": 0,
                        "activation_value": 0.9,
                        "token_id": 262,
                        "token_str": " the",
                        "position": 5,
                        "context_tokens": [1, 2, 262, 4, 5],
                        "context_str": "context the example",
                        "input_sentence": "This is the example.",
                        "generated_sentence": "This is the example sentence.",
                        "step_id": 0,
                        "layer_id": 16,
                        "batch_idx": 0,
                        "sequence_idx": 0,
                    }
                ]
            },
            "config": {"d_sae": 1},
            "feature_labels": {"0": "definite articles"},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(mock_data, f)
            temp_path = f.name

        analyzer = ActivationAnalyzer(temp_path)

        assert analyzer.get_feature_label(0) == "definite articles"
        assert analyzer.get_feature_label(999) is None

        labeled_features = analyzer.get_labeled_features()
        assert len(labeled_features) == 1
        assert labeled_features[0] == "definite articles"

        summary = analyzer.get_feature_summary(0)
        assert summary["label"] == "definite articles"

        report = analyzer.create_feature_report(0)
        assert "definite articles" in report

        prompt = analyzer._create_labeling_prompt(summary)
        assert "Feature #0" in prompt
        assert "the" in prompt
        assert "label" in prompt.lower()

        analyzer.feature_labels["1"] = "test label"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_save_path = f.name

        analyzer.save_labels_to_file(temp_save_path)

        with open(temp_save_path, "r") as f:
            saved_data = json.load(f)

        assert "feature_labels" in saved_data
        assert saved_data["feature_labels"]["0"] == "definite articles"
        assert saved_data["feature_labels"]["1"] == "test label"

        Path(temp_path).unlink()
        Path(temp_save_path).unlink()

        assert True, "Labeling functionality is working"
    except Exception as e:
        assert False, f"Labeling functionality is not working: {e}"


def test_gemini_availability():
    try:
        import google.generativeai as genai

        assert genai.__version__ is not None, (
            "Google Generative AI package is available"
        )

    except ImportError:
        assert False, (
            "Google Generative AI package not installed. Install with: pip install google-generativeai"
        )


if __name__ == "__main__":
    pytest.main([__file__])
