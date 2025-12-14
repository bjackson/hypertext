"""Model loading utilities for ByT5."""

from pathlib import Path
from typing import Optional, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .quantize import quantize_model


_model_cache = {}
_tokenizer_cache = {}


def load_model(
	model_path: str | Path,
	quantize: bool = False,
	device: str = None,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
	"""Load ByT5 model and tokenizer from checkpoint.

	Args:
		model_path: Path to model checkpoint or HuggingFace model name
		quantize: Whether to apply dynamic quantization
		device: Device to load model on ("cpu", "cuda", "mps", or None for auto-detect)

	Returns:
		(model, tokenizer) tuple
	"""
	# Auto-detect device if not specified
	if device is None:
		if torch.cuda.is_available():
			device = 'cuda'
		elif torch.backends.mps.is_available():
			device = 'mps'
		else:
			device = 'cpu'

	# Check cache
	cache_key = f'{model_path}_{quantize}_{device}'
	if cache_key in _model_cache:
		return _model_cache[cache_key], _tokenizer_cache[cache_key]

	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_path)

	# Load model
	model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
	model.eval()

	# Move to device
	model = model.to(device)
	print(f'Model loaded on device: {device}')

	# Apply quantization if requested (only for CPU)
	if quantize and device == 'cpu':
		model = quantize_model(model)
	elif quantize and device != 'cpu':
		print(f'Warning: Quantization only supported on CPU, ignoring quantize flag for device {device}')

	# Cache
	_model_cache[cache_key] = model
	_tokenizer_cache[cache_key] = tokenizer

	return model, tokenizer
