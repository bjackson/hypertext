"""Evaluation metrics for shorthand expansion."""

from typing import List, Dict

from ..utils.normalization import normalize_text


def levenshtein_distance(s1: str, s2: str) -> int:
	"""Compute Levenshtein (edit) distance between two strings.

	Args:
		s1: First string
		s2: Second string

	Returns:
		Edit distance
	"""
	if len(s1) < len(s2):
		return levenshtein_distance(s2, s1)
	
	if len(s2) == 0:
		return len(s1)
	
	previous_row = list(range(len(s2) + 1))
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1
			deletions = current_row[j] + 1
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row
	
	return previous_row[-1]


def character_error_rate(pred: str, gold: str) -> float:
	"""Compute Character Error Rate (CER).

	CER = (substitutions + insertions + deletions) / len(gold)

	Args:
		pred: Predicted text
		gold: Gold standard text

	Returns:
		CER (0.0 = perfect match)
	"""
	if len(gold) == 0:
		return 1.0 if len(pred) > 0 else 0.0
	
	edit_distance = levenshtein_distance(pred, gold)
	return edit_distance / len(gold)


def normalized_edit_distance(pred: str, gold: str) -> float:
	"""Compute normalized edit distance.

	Normalized by max(len(pred), len(gold))

	Args:
		pred: Predicted text
		gold: Gold standard text

	Returns:
		Normalized edit distance (0.0 = perfect match, 1.0 = completely different)
	"""
	max_len = max(len(pred), len(gold))
	if max_len == 0:
		return 0.0
	
	edit_distance = levenshtein_distance(pred, gold)
	return edit_distance / max_len


def exact_match(pred: str, gold: str, normalize: bool = True) -> bool:
	"""Check if prediction exactly matches gold (after normalization).

	Args:
		pred: Predicted text
		gold: Gold standard text
		normalize: Whether to normalize before comparison

	Returns:
		True if exact match
	"""
	if normalize:
		pred = normalize_text(pred)
		gold = normalize_text(gold)
	
	return pred == gold


def compute_metrics(
	predictions: List[str],
	gold: str,
	k: int = 5,
) -> Dict[str, float]:
	"""Compute all metrics for a single example.

	Args:
		predictions: List of top-k predictions (sorted by score, best first)
		gold: Gold standard text
		k: Number of predictions provided

	Returns:
		Dictionary of metric names to values
	"""
	# Normalize gold
	gold_norm = normalize_text(gold)
	
	# Normalize predictions
	preds_norm = [normalize_text(p) for p in predictions]
	
	# Top-1 exact match
	top1_exact = exact_match(preds_norm[0] if preds_norm else "", gold_norm, normalize=False)
	
	# Top-k exact match (is gold in top-k?)
	topk_exact = any(p == gold_norm for p in preds_norm)
	
	# Character Error Rate (on top-1)
	cer = character_error_rate(preds_norm[0] if preds_norm else "", gold_norm)
	
	# Levenshtein distance (on top-1)
	lev_dist = levenshtein_distance(preds_norm[0] if preds_norm else "", gold_norm)
	
	# Normalized edit distance (on top-1)
	norm_edit_dist = normalized_edit_distance(preds_norm[0] if preds_norm else "", gold_norm)
	
	return {
		'top1_exact_match': 1.0 if top1_exact else 0.0,
		'topk_exact_match': 1.0 if topk_exact else 0.0,
		'cer': cer,
		'levenshtein_distance': float(lev_dist),
		'normalized_edit_distance': norm_edit_dist,
	}
