"""Beam search wrapper for generation."""

from typing import List, Dict, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def generate_top_k(
	model: AutoModelForSeq2SeqLM,
	tokenizer: AutoTokenizer,
	input_text: str,
	k: int = 5,
	max_length: int = 128,
	early_stopping: bool = True,
	no_repeat_ngram_size: int = 2,
	device: str = 'cpu',
) -> List[Dict[str, any]]:
	"""Generate top-k candidates using beam search.

	Args:
		model: ByT5 model
		tokenizer: ByT5 tokenizer
		input_text: Input shorthand text
		k: Number of candidates to return
		max_length: Maximum generation length
		early_stopping: Whether to stop early
		no_repeat_ngram_size: Size of n-grams to avoid repeating
		device: Device to run on

	Returns:
		List of dicts with "text" and "score" keys, sorted by score (descending)
	"""
	# Tokenize input
	inputs = tokenizer(
		input_text,
		return_tensors='pt',
		padding=True,
		truncation=True,
		max_length=max_length,
	).to(device)

	# Generate with beam search
	with torch.no_grad():
		# Get decoder_start_token_id from model config if available
		decoder_start_token_id = getattr(model.config, 'decoder_start_token_id', None)

		outputs = model.generate(
			**inputs,
			num_beams=k,
			num_return_sequences=k,
			max_length=max_length,
			early_stopping=early_stopping,
			no_repeat_ngram_size=no_repeat_ngram_size,
			return_dict_in_generate=True,
			output_scores=True,
			# Optimize for speed
			do_sample=False,  # Deterministic beam search
			length_penalty=1.0,  # No length penalty for speed
			decoder_start_token_id=decoder_start_token_id,  # Use proper decoder start token
		)

	# Decode sequences
	sequences = outputs.sequences
	scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None

	# Decode to text
	results = []
	for i, seq in enumerate(sequences):
		text = tokenizer.decode(seq, skip_special_tokens=True)

		# Clean up output: remove task prefix if model includes it
		# The model sometimes generates "expand:" or "xpand:" in output
		text = text.strip()

		# Remove common prefix variations (case-insensitive)
		prefixes_to_remove = [
			'xpand:',
			'expand:',
			'expansi',
			'expansion',
			'expans',
		]
		for prefix in prefixes_to_remove:
			if text.lower().startswith(prefix.lower()):
				# Find where the prefix ends
				prefix_len = len(prefix)
				text = text[prefix_len:].strip()
				# Remove any leading colon, space, or newline
				while text and (text[0] in [':', ' ', '\n', '\t']):
					text = text[1:].strip()
				break

		# Also remove if it starts with just ":" or other common patterns
		if text.startswith(':'):
			text = text[1:].strip()

		# Remove any leading/trailing whitespace
		text = text.strip()

		score = float(scores[i]) if scores is not None else 0.0
		results.append(
			{
				'text': text,
				'score': score,
			}
		)

	# Sort by score (descending, higher is better for log-prob)
	results.sort(key=lambda x: x['score'], reverse=True)

	return results
