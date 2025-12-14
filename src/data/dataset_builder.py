"""Build training datasets from corpus and gold data."""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm
from datasets import load_dataset

from .generator import ShorthandGenerator
from .gold_loader import load_from_specs, load_from_tsv


def build_synthetic_dataset_from_hf_single(
	dataset_name: str,
	column_name: str,
	output_path: str | Path,
	n_examples: int,
	generator: ShorthandGenerator,
	split: str = 'train',
	streaming: bool = False,
	seed: int = 42,
) -> int:
	"""Generate synthetic shorthand pairs from a single Hugging Face dataset.

	Args:
		dataset_name: Hugging Face dataset name (e.g., "binhgiangnguyendanh/reddit_casual_conversation_for_alpaca_lora")
		column_name: Column name to use (e.g., "output")
		output_path: Path to output JSONL file (will append if exists)
		n_examples: Target number of examples to generate from this dataset
		generator: ShorthandGenerator instance
		split: Dataset split to use (default: "train")
		streaming: Whether to use streaming mode for large datasets
		seed: Random seed

	Returns:
		Number of examples actually generated
	"""
	random.seed(seed)

	# Load dataset
	print(f'Loading dataset: {dataset_name} (split: {split}, column: {column_name})')
	dataset = load_dataset(dataset_name, streaming=streaming)

	# Get the split (same for both streaming and non-streaming)
	split_data = dataset[split]

	generated = 0
	with open(output_path, 'w', encoding='utf-8') as out_f:
		for record in tqdm(split_data, desc=f'Generating from {dataset_name}'):
			if generated >= n_examples:
				break

			# Get text from specified column
			text = record.get(column_name, '')
			if not text or not isinstance(text, str):
				continue

			# Clean and filter by length (3-60 tokens roughly)
			text = text.strip()
			words = text.split()
			if len(words) < 3 or len(words) > 60:
				continue

			# Generate shorthand
			shorthand = generator.generate(text)
			if shorthand:
				example = {
					'shorthand': shorthand,
					'full': text,
					'left': '',
					'right': '',
				}
				out_f.write(json.dumps(example) + '\n')
				generated += 1

	return generated


def build_synthetic_dataset_from_hf_multiple(
	datasets: List[Dict[str, any]],
	output_path: str | Path,
	n_examples: int,
	generator: ShorthandGenerator,
	seed: int = 42,
) -> int:
	"""Generate synthetic shorthand pairs from multiple Hugging Face datasets.

	Args:
		datasets: List of dataset configs, each with 'name', 'column', 'split', 'streaming', 'weight'
		output_path: Path to output JSONL file
		n_examples: Target total number of examples to generate
		generator: ShorthandGenerator instance
		seed: Random seed

	Returns:
		Total number of examples generated
	"""
	random.seed(seed)

	# Calculate examples per dataset based on weights
	total_weight = sum(d.get('weight', 1.0) for d in datasets)
	examples_per_dataset = []
	for dataset_config in datasets:
		weight = dataset_config.get('weight', 1.0)
		n = int(n_examples * weight / total_weight)
		examples_per_dataset.append(n)

	# Generate from each dataset
	total_generated = 0
	for i, dataset_config in enumerate(datasets):
		dataset_name = dataset_config['name']
		column_name = dataset_config['column']
		split = dataset_config.get('split', 'train')
		streaming = dataset_config.get('streaming', True)
		n = examples_per_dataset[i]

		print(f'Generating {n} examples from {dataset_name}...')
		generated = build_synthetic_dataset_from_hf_single(
			dataset_name=dataset_name,
			column_name=column_name,
			output_path=output_path,
			n_examples=n,
			generator=generator,
			split=split,
			streaming=streaming,
			seed=seed + i,  # Different seed for each dataset
		)
		total_generated += generated

	return total_generated


def build_synthetic_dataset(
	corpus_path: Optional[str | Path] = None,
	output_path: Optional[str | Path] = None,
	n_examples: Optional[int] = None,
	generator: Optional[ShorthandGenerator] = None,
	seed: int = 42,
	# HF dataset options (single dataset - backward compatibility)
	dataset_name: Optional[str] = None,
	column_name: Optional[str] = None,
	split: str = 'train',
	streaming: bool = False,
	# Multiple datasets option
	datasets: Optional[List[Dict[str, any]]] = None,
) -> int:
	"""Generate synthetic shorthand pairs from corpus or Hugging Face dataset.

	Args:
		corpus_path: Path to raw_corpus.txt (if using file-based source)
		output_path: Path to output JSONL file
		n_examples: Target number of examples to generate
		generator: ShorthandGenerator instance
		seed: Random seed
		dataset_name: Hugging Face dataset name (if using single HF source, backward compatibility)
		column_name: Column name to use from HF dataset
		split: Dataset split to use (default: "train")
		streaming: Whether to use streaming mode for large datasets
		datasets: List of dataset configs for multiple datasets (each with 'name', 'column', 'split', 'streaming', 'weight')

	Returns:
		Number of examples actually generated
	"""
	# Use multiple HF datasets if provided
	if datasets:
		if not output_path or n_examples is None or generator is None:
			raise ValueError('output_path, n_examples, and generator must be provided')
		return build_synthetic_dataset_from_hf_multiple(
			datasets=datasets,
			output_path=output_path,
			n_examples=n_examples,
			generator=generator,
			seed=seed,
		)

	# Use single HF dataset if provided (backward compatibility)
	if dataset_name and column_name:
		if not output_path or n_examples is None or generator is None:
			raise ValueError('output_path, n_examples, and generator must be provided')
		return build_synthetic_dataset_from_hf_single(
			dataset_name=dataset_name,
			column_name=column_name,
			output_path=output_path,
			n_examples=n_examples,
			generator=generator,
			split=split,
			streaming=streaming,
			seed=seed,
		)

	# Otherwise use file-based corpus
	if not corpus_path or not output_path or n_examples is None or generator is None:
		raise ValueError(
			'Either corpus_path or (dataset_name, column_name) must be provided, and output_path, n_examples, and generator must be provided'
		)

	random.seed(seed)

	# Read corpus lines
	with open(corpus_path, 'r', encoding='utf-8') as f:
		lines = [line.strip() for line in f if line.strip()]

	# Shuffle and sample
	random.shuffle(lines)

	generated = 0
	with open(output_path, 'w', encoding='utf-8') as out_f:
		for line in tqdm(lines, desc='Generating synthetic pairs'):
			if generated >= n_examples:
				break

			# Filter by length (3-60 tokens roughly)
			words = line.split()
			if len(words) < 3 or len(words) > 60:
				continue

			# Generate shorthand
			shorthand = generator.generate(line)
			if shorthand:
				example = {
					'shorthand': shorthand,
					'full': line,
					'left': '',
					'right': '',
				}
				out_f.write(json.dumps(example) + '\n')
				generated += 1

	return generated


def load_gold_dataset(
	specs_path: Optional[str | Path] = None, tsv_path: Optional[str | Path] = None
) -> List[Dict[str, str]]:
	"""Load gold dataset from specs.txt or seed_pairs.tsv.

	Args:
		specs_path: Path to specs.txt (optional)
		tsv_path: Path to seed_pairs.tsv (optional)

	Returns:
		List of gold examples
	"""
	if tsv_path and Path(tsv_path).exists():
		return load_from_tsv(tsv_path)
	elif specs_path and Path(specs_path).exists():
		return load_from_specs(specs_path)
	else:
		return []


def split_dataset(
	data: List[Dict[str, str]],
	train_ratio: float = 0.8,
	val_ratio: float = 0.1,
	test_ratio: float = 0.1,
	seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
	"""Split dataset into train/val/test.

	Args:
		data: List of examples
		train_ratio: Proportion for training
		val_ratio: Proportion for validation
		test_ratio: Proportion for test

	Returns:
		(train, val, test) lists
	"""
	random.seed(seed)
	shuffled = data.copy()
	random.shuffle(shuffled)

	n = len(shuffled)
	n_train = int(n * train_ratio)
	n_val = int(n * val_ratio)

	train = shuffled[:n_train]
	val = shuffled[n_train : n_train + n_val]
	test = shuffled[n_train + n_val :]

	return train, val, test


def merge_and_shuffle(
	synthetic: List[Dict[str, str]],
	gold: List[Dict[str, str]],
	synthetic_ratio: float = 0.9,
	seed: int = 42,
) -> List[Dict[str, str]]:
	"""Merge synthetic and gold data with specified ratio.

	Args:
		synthetic: Synthetic examples
		gold: Gold examples
		synthetic_ratio: Proportion of synthetic data in output

	Returns:
		Merged and shuffled list
	"""
	random.seed(seed)

	# Calculate how many gold examples to include
	if not gold:
		return synthetic

	if not synthetic:
		return gold

	# Oversample gold to reach desired ratio
	target_gold = int(len(synthetic) * (1 - synthetic_ratio) / synthetic_ratio)
	if target_gold > len(gold):
		# Repeat gold examples if needed
		repeated_gold = (gold * ((target_gold // len(gold)) + 1))[:target_gold]
	else:
		repeated_gold = random.sample(gold, target_gold)

	merged = synthetic + repeated_gold
	random.shuffle(merged)
	return merged


def write_jsonl(data: List[Dict[str, str]], output_path: str | Path):
	"""Write data to JSONL file.

	Args:
		data: List of examples
		output_path: Output file path
	"""
	with open(output_path, 'w', encoding='utf-8') as f:
		for example in data:
			f.write(json.dumps(example) + '\n')


def validate_jsonl(path: str | Path) -> bool:
	"""Validate JSONL file format.

	Args:
		path: Path to JSONL file

	Returns:
		True if valid, False otherwise
	"""
	required_keys = {'shorthand', 'full', 'left', 'right'}

	try:
		with open(path, 'r', encoding='utf-8') as f:
			for i, line in enumerate(f):
				if not line.strip():
					continue
				example = json.loads(line)
				if not all(key in example for key in required_keys):
					print(f'Missing keys in line {i+1}: {required_keys - set(example.keys())}')
					return False
		return True
	except Exception as e:
		print(f'Error validating JSONL: {e}')
		return False
