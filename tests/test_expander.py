"""Tests for expander API (requires trained model)."""

import pytest
from pathlib import Path

from src.inference.expander import expand


@pytest.mark.skip(reason="Requires trained model")
def test_expand_basic():
	"""Test basic expansion (requires trained model)."""
	model_path = Path('models/byt5-shorthand-v1')
	if not model_path.exists():
		pytest.skip("Model not found")
	
	results = expand(
		shorthand="idk",
		model_path=str(model_path),
		k=5,
	)
	
	assert len(results) > 0
	assert all('text' in r and 'score' in r for r in results)
	assert results[0]['score'] >= results[-1]['score']  # Sorted by score


@pytest.mark.skip(reason="Requires trained model")
def test_expand_known_examples():
	"""Test expansion on known examples (requires trained model)."""
	model_path = Path('models/byt5-shorthand-v1')
	if not model_path.exists():
		pytest.skip("Model not found")
	
	test_cases = [
		"idk",
		"ruavailrn",
		"whtaddr",
	]
	
	for shorthand in test_cases:
		results = expand(
			shorthand=shorthand,
			model_path=str(model_path),
			k=5,
		)
		assert len(results) > 0
		assert all(isinstance(r['text'], str) for r in results)
		assert all(isinstance(r['score'], (int, float)) for r in results)
