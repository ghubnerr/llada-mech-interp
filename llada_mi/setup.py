import torch
from typing import Literal, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel


def load_model_and_tokenizer(
    model_name: str = "GSAI-ML/LLaDA-8B-Base",
) -> Tuple[AutoModel, AutoTokenizer, str]:
    """
    Loads a pre-trained diffusion language model and its tokenizer.

    This function specifically handles LLaDA (Large Language Diffusion Architecture) models,
    which are diffusion-based language models that generate text through iterative denoising
    processes. The model is automatically placed on GPU if available and set to evaluation mode.

    Args:
        model_name (str): The name of the LLaDA model to load from Hugging Face Hub.
                         Defaults to "GSAI-ML/LLaDA-8B-Base", which is an 8B parameter
                         diffusion language model.

    Returns:
        Tuple[AutoModel, AutoTokenizer, str]: A tuple containing:
            - model: The loaded LLaDA diffusion model in evaluation mode
            - tokenizer: The corresponding tokenizer for text processing
            - device: The device the model is loaded on ("cuda" or "cpu")

    Example:
        >>> model, tokenizer, device = load_model_and_tokenizer()
        >>> print(f"Model loaded on {device}")
        Model loaded on cuda
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: AutoModel = (
        AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    return model, tokenizer, device


def get_generation_parameters(
    model_name: str = "GSAI-ML/LLaDA-8B-Base",
    steps: int = 12,
    gen_length: int = 8,
    block_length: int = 8,
    temperature: float = 0.0,
    remasking: Literal["low_confidence", "random"] = "low_confidence",
    mask_id: int = 126336,
) -> Dict[str, Any]:
    """
    Returns default generation parameters for LLaDA diffusion language models.

    LLaDA models use a diffusion process where text is generated through iterative
    denoising steps. This function provides the optimal parameters for the diffusion
    generation process, including the number of denoising steps, generation length,
    and remasking strategy.

    Args:
        model_name (str): The name of the LLaDA model. Currently supports
                         "GSAI-ML/LLaDA-8B-Base" with optimized parameters.
        steps (int): Number of denoising steps in the diffusion process.
                    More steps generally produce higher quality but slower generation.
                    Default: 12
        gen_length (int): Length of the generated text sequence.
                         Default: 8
        block_length (int): Length of text blocks processed in each step.
                           Default: 8
        temperature (float): Controls randomness in the generation process.
                           Lower values (0.0) produce more deterministic output.
                           Default: 0.0
        remasking (Literal["low_confidence", "random"]): Strategy for remasking
            during the diffusion process:
            - "low_confidence": Remask tokens with low confidence scores
            - "random": Apply random remasking
            Default: "low_confidence"
        mask_id (int): Token ID used for masking in the diffusion process.
                      Default: 126336 (specific to LLaDA models)

    Returns:
        Dict[str, Any]: Dictionary containing generation parameters optimized
                       for the specified LLaDA model. Returns empty dict for
                       unsupported models.

    Example:
        >>> params = get_generation_parameters(steps=20, temperature=0.1)
        >>> print(params)
        {'steps': 20, 'gen_length': 8, 'block_length': 8, 'temperature': 0.1, ...}
    """
    if model_name == "GSAI-ML/LLaDA-8B-Base":
        return {
            "steps": steps,
            "gen_length": gen_length,
            "block_length": block_length,
            "temperature": temperature,
            "remasking": remasking,
            "mask_id": mask_id,
        }
    else:
        return {}
