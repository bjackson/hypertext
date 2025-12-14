"""Tests for shorthand generator."""

import pytest
from pathlib import Path

from src.data.generator import ShorthandGenerator
from src.data.lexicon import load_lexicon, apply_lexicon_replacements


def test_generator_basic():
	"""Test basic generator functionality."""
	generator = ShorthandGenerator(seed=42)
	
	# Test that it generates something
	result = generator.generate("I don't know about that")
	assert result is not None
	assert len(result) > 0
	assert len(result) < len("I don't know about that")


def test_generator_with_lexicon():
	"""Test generator with lexicon."""
	lexicon_path = Path('data/lexicon.json')
	if not lexicon_path.exists():
		pytest.skip("lexicon.json not found")
	
	generator = ShorthandGenerator(lexicon_path=str(lexicon_path), seed=42)
	
	# Test with phrase that should be replaced
	result = generator.generate("right now")
	assert result is not None
	# Should contain "rn" from lexicon
	assert 'rn' in result.lower()


def test_generator_validation():
	"""Test generator validation filters."""
	generator = ShorthandGenerator(seed=42)
	
	# Very short input should return None
	result = generator.generate("hi")
	assert result is None or len(result) >= 2
	
	# Very long input might be filtered
	long_text = " ".join(["word"] * 100)
	result = generator.generate(long_text)
	# Should either be None or valid
	if result:
		assert 2 <= len(result) <= 50


def test_lexicon_loading():
	"""Test lexicon loading."""
	lexicon_path = Path('data/lexicon.json')
	if not lexicon_path.exists():
		pytest.skip("lexicon.json not found")
	
	lexicon = load_lexicon(lexicon_path)
	assert isinstance(lexicon, dict)
	assert len(lexicon) > 0
	assert "right now" in lexicon
	assert lexicon["right now"] == "rn"


def test_lexicon_replacements():
	"""Test lexicon replacements."""
	lexicon = {
		"right now": "rn",
		"about": "abt",
		"I don't know": "idk",
	}
	
	text = "I don't know about that right now"
	result = apply_lexicon_replacements(text, lexicon)
	
	# Should have replacements
	assert "idk" in result.lower() or "abt" in result.lower() or "rn" in result.lower()
