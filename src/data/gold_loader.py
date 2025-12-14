"""Load curated gold dataset from specs.txt or seed_pairs.tsv."""

import json
from pathlib import Path
from typing import List, Dict


def load_from_specs(specs_path: str | Path) -> List[Dict[str, str]]:
	"""Load gold pairs from specs.txt (format: shorthand, full text).

	Args:
		specs_path: Path to specs.txt file

	Returns:
		List of dicts with keys: shorthand, full, left, right
	"""
	pairs = []
	with open(specs_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line or ',' not in line:
				continue
			
			# Split on comma, but handle commas in the full text
			parts = line.split(',', 1)
			if len(parts) == 2:
				shorthand = parts[0].strip()
				full = parts[1].strip()
				if shorthand and full:
					pairs.append({
						'shorthand': shorthand,
						'full': full,
						'left': '',
						'right': '',
					})
	
	return pairs


def load_from_tsv(tsv_path: str | Path) -> List[Dict[str, str]]:
	"""Load gold pairs from TSV file (format: shorthand\tfull text).

	Args:
		tsv_path: Path to TSV file

	Returns:
		List of dicts with keys: shorthand, full, left, right
	"""
	pairs = []
	with open(tsv_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line or '\t' not in line:
				continue
			
			parts = line.split('\t', 1)
			if len(parts) == 2:
				shorthand = parts[0].strip()
				full = parts[1].strip()
				if shorthand and full:
					pairs.append({
						'shorthand': shorthand,
						'full': full,
						'left': '',
						'right': '',
					})
	
	return pairs
