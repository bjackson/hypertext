"""Lexicon management for phrase-to-abbreviation mappings."""

import json
from pathlib import Path
from typing import Dict


def load_lexicon(lexicon_path: str | Path) -> Dict[str, str]:
	"""Load lexicon from JSON file.

	Args:
		lexicon_path: Path to lexicon.json file

	Returns:
		Dictionary mapping full phrases to abbreviations
	"""
	with open(lexicon_path, 'r', encoding='utf-8') as f:
		return json.load(f)


def apply_lexicon_replacements(text: str, lexicon: Dict[str, str]) -> str:
	"""Apply lexicon replacements to text (longest matches first).

	Args:
		text: Input text
		lexicon: Dictionary of phrase -> abbreviation mappings

	Returns:
		Text with lexicon replacements applied
	"""
	# Sort by length (longest first) to match longer phrases first
	sorted_items = sorted(lexicon.items(), key=lambda x: len(x[0]), reverse=True)
	
	result = text
	for phrase, abbrev in sorted_items:
		# Case-insensitive replacement
		import re
		pattern = re.compile(re.escape(phrase), re.IGNORECASE)
		result = pattern.sub(abbrev, result)
	
	return result
