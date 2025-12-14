"""CPU quantization utilities for faster inference."""

import torch
from transformers import AutoModelForSeq2SeqLM


def quantize_model(model: AutoModelForSeq2SeqLM) -> AutoModelForSeq2SeqLM:
	"""Apply dynamic quantization to model for CPU inference.

	Args:
		model: ByT5 model

	Returns:
		Quantized model
	"""
	# Dynamic quantization for Linear layers
	# Use torch.ao.quantization for newer PyTorch versions
	try:
		from torch.ao.quantization import quantize_dynamic
	except ImportError:
		from torch.quantization import quantize_dynamic

	quantized_model = quantize_dynamic(
		model,
		{torch.nn.Linear},
		dtype=torch.qint8,
	)

	return quantized_model
