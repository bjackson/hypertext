"""Tests for evaluation metrics."""

import pytest

from src.eval.metrics import (
	levenshtein_distance,
	character_error_rate,
	normalized_edit_distance,
	exact_match,
	compute_metrics,
)
from src.utils.normalization import normalize_text


def test_levenshtein_distance():
	"""Test Levenshtein distance calculation."""
	assert levenshtein_distance("kitten", "sitting") == 3
	assert levenshtein_distance("", "abc") == 3
	assert levenshtein_distance("abc", "") == 3
	assert levenshtein_distance("abc", "abc") == 0
	assert levenshtein_distance("abc", "def") == 3


def test_character_error_rate():
	"""Test CER calculation."""
	# Perfect match
	assert character_error_rate("hello", "hello") == 0.0
	
	# One substitution
	assert character_error_rate("hello", "hallo") == 0.2  # 1/5
	
	# Empty gold
	assert character_error_rate("abc", "") == 1.0
	assert character_error_rate("", "") == 0.0


def test_normalized_edit_distance():
	"""Test normalized edit distance."""
	assert normalized_edit_distance("abc", "abc") == 0.0
	assert normalized_edit_distance("abc", "def") == 1.0  # 3/3
	assert normalized_edit_distance("kitten", "sitting") == 3 / 7  # 3/7


def test_exact_match():
	"""Test exact match with normalization."""
	assert exact_match("Hello World", "hello world") == True
	assert exact_match("Hello, World!", "hello world") == True
	assert exact_match("Hello World", "hello  world") == True  # spaces collapsed
	assert exact_match("Hello World", "goodbye world") == False


def test_compute_metrics():
	"""Test metric computation."""
	predictions = [
		"hello world",
		"hello word",
		"goodbye world",
	]
	gold = "hello world"
	
	metrics = compute_metrics(predictions, gold, k=3)
	
	assert metrics['top1_exact_match'] == 1.0
	assert metrics['topk_exact_match'] == 1.0
	assert metrics['cer'] == 0.0
	assert metrics['levenshtein_distance'] == 0.0
	
	# Test with wrong prediction
	predictions2 = ["goodbye", "hello word"]
	gold2 = "hello world"
	metrics2 = compute_metrics(predictions2, gold2, k=2)
	
	assert metrics2['top1_exact_match'] == 0.0
	assert metrics2['topk_exact_match'] == 0.0
	assert metrics2['cer'] > 0.0


def test_normalize_text():
	"""Test text normalization."""
	assert normalize_text("Hello, World!") == "hello world"
	assert normalize_text("  Hello   World  ") == "hello world"
	assert normalize_text("HELLO") == "hello"
	assert normalize_text("test123") == "test123"
