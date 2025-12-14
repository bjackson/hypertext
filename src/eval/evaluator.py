"""Evaluator to run metrics on test set."""

import json
from pathlib import Path
from typing import Dict, List, Any

from tqdm import tqdm

from ..inference.expander import expand
from .metrics import compute_metrics
from .test_loader import load_test_set


def evaluate(
	model_path: str | Path,
	test_path: str | Path,
	k: int = 5,
	quantize: bool = False,
	device: str = "cpu",
) -> Dict[str, Any]:
	"""Run evaluation on test set.

	Args:
		model_path: Path to model checkpoint
		test_path: Path to test.jsonl file
		k: Number of top candidates to generate
		quantize: Whether to use quantized model
		device: Device to run on

	Returns:
		Dictionary with aggregated metrics and per-example results
	"""
	# Load test set
	test_examples = load_test_set(test_path)
	
	# Initialize metrics accumulators
	all_metrics = []
	per_example = []
	
	# Evaluate each example
	for example in tqdm(test_examples, desc="Evaluating"):
		shorthand = example['shorthand']
		gold = example['full']
		
		# Generate predictions
		try:
			results = expand(
				shorthand,
				k=k,
				model_path=model_path,
				quantize=quantize,
				device=device,
			)
			predictions = [r['text'] for r in results]
		except Exception as e:
			print(f"Error expanding '{shorthand}': {e}")
			predictions = []
		
		# Compute metrics
		metrics = compute_metrics(predictions, gold, k=k)
		all_metrics.append(metrics)
		
		# Store per-example result
		per_example.append({
			'shorthand': shorthand,
			'gold': gold,
			'predictions': predictions,
			**metrics,
		})
	
	# Aggregate metrics
	aggregated = {
		'top1_exact_match': sum(m['top1_exact_match'] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
		'topk_exact_match': sum(m['topk_exact_match'] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
		'avg_cer': sum(m['cer'] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
		'avg_levenshtein': sum(m['levenshtein_distance'] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
		'avg_normalized_edit_distance': sum(m['normalized_edit_distance'] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
		'num_examples': len(all_metrics),
	}
	
	return {
		'aggregated': aggregated,
		'per_example': per_example,
	}
