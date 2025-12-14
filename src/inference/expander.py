"""Main expansion API for shorthand to full text."""

from pathlib import Path
from typing import List, Dict, Optional

from .model_loader import load_model
from .beam_search import generate_top_k


_model_instance = None
_tokenizer_instance = None


def expand(
	shorthand: str,
	left_context: str = "",
	right_context: str = "",
	k: int = 5,
	model_path: Optional[str | Path] = None,
	quantize: bool = False,
	device: str = "cpu",
	task_prefix: str = "expand: ",
	max_length: int = 128,
) -> List[Dict[str, any]]:
	"""Expand shorthand to full text.

	Args:
		shorthand: Input shorthand string
		left_context: Left context (reserved for future use)
		right_context: Right context (reserved for future use)
		k: Number of top candidates to return
		model_path: Path to model checkpoint (if None, uses cached model)
		quantize: Whether to use quantized model
		device: Device to run on ("cpu" or "cuda")
		task_prefix: Task prefix for input formatting
		max_length: Maximum generation length

	Returns:
		List of dicts with "text" and "score" keys, sorted by score (descending)
	"""
	global _model_instance, _tokenizer_instance
	
	# Load model if needed
	if model_path is not None or _model_instance is None:
		if model_path is None:
			raise ValueError("model_path must be provided on first call")
		_model_instance, _tokenizer_instance = load_model(
			model_path,
			quantize=quantize,
			device=device,
		)
	
	# Format input: "expand: {shorthand}"
	# Context fields reserved for future use
	input_text = f"{task_prefix}{shorthand}"
	
	# Generate top-k candidates
	results = generate_top_k(
		_model_instance,
		_tokenizer_instance,
		input_text,
		k=k,
		max_length=max_length,
		device=device,
	)
	
	return results
