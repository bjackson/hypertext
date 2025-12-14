"""Test set loading utilities."""

import json
from pathlib import Path
from typing import List, Dict


def load_test_set(test_path: str | Path) -> List[Dict[str, str]]:
	"""Load test set from JSONL file.

	Args:
		test_path: Path to test.jsonl file

	Returns:
		List of test examples
	"""
	examples = []
	with open(test_path, 'r', encoding='utf-8') as f:
		for line in f:
			if line.strip():
				examples.append(json.loads(line))
	
	return examples
