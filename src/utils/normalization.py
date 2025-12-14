"""Text normalization utilities for evaluation."""

import re


def normalize_text(text: str) -> str:
	"""Normalize text for evaluation (lowercase, remove punctuation, collapse spaces).

	Args:
		text: Input text

	Returns:
		Normalized text
	"""
	# Lowercase
	text = text.lower()

	# Remove punctuation (keep alphanumeric + spaces)
	text = re.sub(r'[^a-z0-9 ]', '', text)

	# Collapse multiple spaces to single space
	text = re.sub(r'\s+', ' ', text)

	# Trim whitespace
	text = text.strip()

	return text
