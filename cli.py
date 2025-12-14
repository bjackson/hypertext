"""CLI interface for shorthand expansion model."""

import argparse
import json
import sys
from pathlib import Path

from src.data.dataset_builder import (
	build_synthetic_dataset,
	load_gold_dataset,
	split_dataset,
	merge_and_shuffle,
	write_jsonl,
	validate_jsonl,
)
from src.data.generator import ShorthandGenerator
from src.data.lexicon import load_lexicon
from src.train.trainer import train
from src.eval.evaluator import evaluate
from src.inference.expander import expand
import yaml


def cmd_generate_data(args):
	"""Generate synthetic training data."""
	# Load data config
	if args.config:
		with open(args.config, 'r') as f:
			config = yaml.safe_load(f)
		data_config = config.get('dataset', {})
		gen_config = config.get('generator', {})
	else:
		data_config = {}
		gen_config = {}

	# Get paths and options
	lexicon_path = args.lexicon or data_config.get('lexicon_path', 'data/lexicon.json')
	output_path = args.output or 'data/train.jsonl'
	n_examples = args.n or data_config.get('n_synthetic', 500000)
	seed = args.seed or data_config.get('seed', 42)

	# Check if using HF dataset or file-based corpus
	dataset_name = args.dataset_name or data_config.get('dataset_name')
	column_name = args.column_name or data_config.get('column_name', 'output')
	split = args.split or data_config.get('split', 'train')
	streaming = args.streaming if args.streaming is not None else data_config.get('streaming', True)
	corpus_path = args.corpus or data_config.get('corpus_path')

	# Create generator
	generator = ShorthandGenerator(
		lexicon_path=lexicon_path,
		p_vowel_drop=gen_config.get('p_vowel_drop', 0.4),
		p_space_drop=gen_config.get('p_space_drop', 0.5),
		p_punct_drop=gen_config.get('p_punct_drop', 0.8),
		use_consonant_skeleton=gen_config.get('use_consonant_skeleton', False),
		seed=seed,
	)

	# Generate synthetic dataset
	if dataset_name:
		print(f'Generating {n_examples} synthetic examples from HF dataset: {dataset_name}')
	else:
		print(f'Generating {n_examples} synthetic examples from corpus: {corpus_path}')

	generated = build_synthetic_dataset(
		corpus_path=corpus_path,
		output_path=output_path,
		n_examples=n_examples,
		generator=generator,
		seed=seed,
		dataset_name=dataset_name,
		column_name=column_name,
		split=split,
		streaming=streaming,
	)

	print(f'Generated {generated} examples to {output_path}')

	# Validate
	if validate_jsonl(output_path):
		print('✓ JSONL validation passed')
	else:
		print('✗ JSONL validation failed')
		sys.exit(1)


def cmd_build_dataset(args):
	"""Build complete dataset with train/val/test splits."""
	# Load data config
	if args.config:
		with open(args.config, 'r') as f:
			config = yaml.safe_load(f)
		data_config = config.get('dataset', {})
		gen_config = config.get('generator', {})
	else:
		data_config = {}
		gen_config = {}

	# Get paths
	corpus_path = data_config.get('corpus_path', 'data/raw_corpus.txt')
	lexicon_path = data_config.get('lexicon_path', 'data/lexicon.json')
	specs_path = data_config.get('specs_path', 'notes/specs.txt')
	tsv_path = data_config.get('tsv_path', 'data/seed_pairs.tsv')
	n_synthetic = data_config.get('n_synthetic', 500000)
	train_ratio = data_config.get('train_ratio', 0.8)
	val_ratio = data_config.get('val_ratio', 0.1)
	test_ratio = data_config.get('test_ratio', 0.1)
	synthetic_train_ratio = data_config.get('synthetic_train_ratio', 0.9)
	seed = data_config.get('seed', 42)

	# Create generator
	generator = ShorthandGenerator(
		lexicon_path=lexicon_path,
		p_vowel_drop=gen_config.get('p_vowel_drop', 0.4),
		p_space_drop=gen_config.get('p_space_drop', 0.5),
		p_punct_drop=gen_config.get('p_punct_drop', 0.8),
		use_consonant_skeleton=gen_config.get('use_consonant_skeleton', False),
		seed=seed,
	)

	# Generate synthetic data
	print('Generating synthetic data...')
	synthetic_output = Path('data') / 'synthetic_temp.jsonl'

	# Check if using HF datasets (multiple or single)
	datasets = data_config.get('datasets')  # New multi-dataset format
	dataset_name = data_config.get('dataset_name')  # Legacy single dataset format
	column_name = data_config.get('column_name', 'output')
	split = data_config.get('split', 'train')
	streaming = data_config.get('streaming', True)

	if datasets:
		# Multiple datasets
		build_synthetic_dataset(
			datasets=datasets,
			output_path=synthetic_output,
			n_examples=n_synthetic,
			generator=generator,
			seed=seed,
		)
	elif dataset_name:
		# Single dataset (backward compatibility)
		build_synthetic_dataset(
			dataset_name=dataset_name,
			column_name=column_name,
			output_path=synthetic_output,
			n_examples=n_synthetic,
			generator=generator,
			split=split,
			streaming=streaming,
			seed=seed,
		)
	else:
		build_synthetic_dataset(
			corpus_path=corpus_path,
			output_path=synthetic_output,
			n_examples=n_synthetic,
			generator=generator,
			seed=seed,
		)

	# Load synthetic
	with open(synthetic_output, 'r') as f:
		synthetic = [json.loads(line) for line in f if line.strip()]

	# Load gold
	print('Loading gold data...')
	gold = load_gold_dataset(specs_path=specs_path, tsv_path=tsv_path)

	# Split gold
	print('Splitting gold data...')
	gold_train, gold_val, gold_test = split_dataset(
		gold,
		train_ratio=train_ratio,
		val_ratio=val_ratio,
		test_ratio=test_ratio,
		seed=seed,
	)

	# Split synthetic (for test set)
	synthetic_train, synthetic_val, synthetic_test = split_dataset(
		synthetic,
		train_ratio=train_ratio,
		val_ratio=val_ratio,
		test_ratio=test_ratio,
		seed=seed + 1,  # Different seed for synthetic split
	)

	# Merge train and val
	train_data = merge_and_shuffle(synthetic_train, gold_train, synthetic_train_ratio, seed=seed)
	val_data = merge_and_shuffle(synthetic_val, gold_val, 0.5, seed=seed)  # 50/50 for val
	test_data = gold_test  # 100% gold for test

	# Write splits
	print('Writing splits...')
	write_jsonl(train_data, 'data/train.jsonl')
	write_jsonl(val_data, 'data/val.jsonl')
	write_jsonl(test_data, 'data/test.jsonl')

	# Clean up temp file
	synthetic_output.unlink()

	print(f'Train: {len(train_data)} examples')
	print(f'Val: {len(val_data)} examples')
	print(f'Test: {len(test_data)} examples')


def cmd_train(args):
	"""Train the model."""
	config_path = args.config or 'configs/train_config.yaml'

	# Auto-detect device if not specified
	device = args.device
	if not device:
		import torch

		if torch.cuda.is_available():
			device = 'cuda'
		elif torch.backends.mps.is_available():
			device = 'mps'
		else:
			device = 'cpu'

	print(f'Training with config: {config_path}')
	print(f'Using device: {device}')
	train(config_path, device=device)


def cmd_eval(args):
	"""Evaluate the model."""
	model_path = args.model
	test_path = args.test or 'data/test.jsonl'
	k = args.k or 5
	quantize = args.quantize

	print(f'Evaluating model: {model_path}')
	print(f'Test set: {test_path}')

	# Auto-detect device if not specified
	device = args.device
	if not device:
		import torch

		if torch.cuda.is_available():
			device = 'cuda'
		elif torch.backends.mps.is_available():
			device = 'mps'
		else:
			device = 'cpu'

	print(f'Using device: {device}')

	results = evaluate(
		model_path=model_path,
		test_path=test_path,
		k=k,
		quantize=quantize,
		device=device,
	)

	# Print summary
	agg = results['aggregated']
	print('\n=== Evaluation Results ===')
	print(f"Top-1 Exact Match: {agg['top1_exact_match']:.4f}")
	print(f"Top-{k} Exact Match: {agg['topk_exact_match']:.4f}")
	print(f"Avg CER: {agg['avg_cer']:.4f}")
	print(f"Avg Levenshtein Distance: {agg['avg_levenshtein']:.2f}")
	print(f"Avg Normalized Edit Distance: {agg['avg_normalized_edit_distance']:.4f}")
	print(f"Number of examples: {agg['num_examples']}")

	# Save detailed results if requested
	if args.output:
		with open(args.output, 'w') as f:
			json.dump(results, f, indent=2)
		print(f'\nDetailed results saved to {args.output}')


def cmd_expand(args):
	"""Expand shorthand interactively."""
	model_path = args.model
	shorthand = args.shorthand
	k = args.k or 5
	quantize = args.quantize
	# Auto-detect device if not specified
	device = args.device
	if not device:
		import torch

		if torch.cuda.is_available():
			device = 'cuda'
		elif torch.backends.mps.is_available():
			device = 'mps'
		else:
			device = 'cpu'

	print(f'Expanding: {shorthand} (device: {device})')

	results = expand(
		shorthand=shorthand,
		k=k,
		model_path=model_path,
		quantize=quantize,
		device=device,
	)

	print(f'\nTop-{k} expansions:')
	for i, result in enumerate(results, 1):
		print(f"{i}. {result['text']} (score: {result['score']:.4f})")


def cmd_expand_batch(args):
	"""Expand shorthand from file."""
	model_path = args.model
	input_path = args.input
	output_path = args.output
	k = args.k or 5
	quantize = args.quantize
	# Auto-detect device if not specified
	device = args.device
	if not device:
		import torch

		if torch.cuda.is_available():
			device = 'cuda'
		elif torch.backends.mps.is_available():
			device = 'mps'
		else:
			device = 'cpu'

	print(f'Using device: {device}')

	# Read input
	with open(input_path, 'r') as f:
		inputs = [line.strip() for line in f if line.strip()]

	print(f'Loading model and processing {len(inputs)} examples...')

	# Load model once (will be cached)
	if inputs:
		expand(
			shorthand=inputs[0],
			k=k,
			model_path=model_path,
			quantize=quantize,
			device=device,
		)

	# Expand each with progress bar and write incrementally
	from tqdm import tqdm

	# Open output file for incremental writing
	with open(output_path, 'w') as f:
		for shorthand in tqdm(inputs, desc='Expanding'):
			try:
				expansions = expand(
					shorthand=shorthand,
					k=k,
					model_path=None,  # Use cached model
					quantize=quantize,
					device=device,
				)
				result = {
					'shorthand': shorthand,
					'expansions': expansions,
				}
			except Exception as e:
				tqdm.write(f"Error expanding '{shorthand}': {e}")
				result = {
					'shorthand': shorthand,
					'expansions': [],
					'error': str(e),
				}

			# Write immediately to file
			f.write(json.dumps(result) + '\n')
			f.flush()  # Ensure it's written to disk immediately

	print(f'\nProcessed {len(inputs)} examples, saved to {output_path}')


def main():
	parser = argparse.ArgumentParser(description='Shorthand Expansion Model CLI')
	subparsers = parser.add_subparsers(dest='command', help='Command to run')

	# generate-data
	gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic training data')
	gen_parser.add_argument('--corpus', type=str, help='Path to raw corpus (if not using HF dataset)')
	gen_parser.add_argument('--dataset-name', type=str, help='Hugging Face dataset name')
	gen_parser.add_argument('--column-name', type=str, help='Column name in HF dataset (default: output)')
	gen_parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
	gen_parser.add_argument('--streaming', action='store_true', help='Use streaming mode for large datasets')
	gen_parser.add_argument('--lexicon', type=str, help='Path to lexicon.json')
	gen_parser.add_argument('--output', type=str, help='Output JSONL path')
	gen_parser.add_argument('--n', type=int, help='Number of examples to generate')
	gen_parser.add_argument('--config', type=str, help='Data config YAML')
	gen_parser.add_argument('--seed', type=int, help='Random seed')

	# build-dataset
	build_parser = subparsers.add_parser('build-dataset', help='Build complete dataset with splits')
	build_parser.add_argument('--config', type=str, help='Data config YAML')

	# train
	train_parser = subparsers.add_parser('train', help='Train the model')
	train_parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Training config YAML')
	train_parser.add_argument(
		'--device', type=str, default=None, help='Device (cpu/cuda/mps, or auto-detect if not specified)'
	)

	# eval
	eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
	eval_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
	eval_parser.add_argument('--test', type=str, help='Path to test.jsonl')
	eval_parser.add_argument('--k', type=int, default=5, help='Number of candidates')
	eval_parser.add_argument('--quantize', action='store_true', help='Use quantized model')
	eval_parser.add_argument(
		'--device', type=str, default=None, help='Device (cpu/cuda/mps, or auto-detect if not specified)'
	)
	eval_parser.add_argument('--output', type=str, help='Output JSON path for detailed results')

	# expand
	expand_parser = subparsers.add_parser('expand', help='Expand shorthand interactively')
	expand_parser.add_argument('shorthand', type=str, help='Shorthand to expand')
	expand_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
	expand_parser.add_argument('--k', type=int, default=5, help='Number of candidates')
	expand_parser.add_argument('--quantize', action='store_true', help='Use quantized model')
	expand_parser.add_argument(
		'--device', type=str, default=None, help='Device (cpu/cuda/mps, or auto-detect if not specified)'
	)

	# expand-batch
	batch_parser = subparsers.add_parser('expand-batch', help='Expand shorthand from file')
	batch_parser.add_argument('--input', type=str, required=True, help='Input file with one shorthand per line')
	batch_parser.add_argument('--output', type=str, required=True, help='Output JSONL path')
	batch_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
	batch_parser.add_argument('--k', type=int, default=5, help='Number of candidates')
	batch_parser.add_argument('--quantize', action='store_true', help='Use quantized model')
	batch_parser.add_argument(
		'--device', type=str, default=None, help='Device (cpu/cuda/mps, or auto-detect if not specified)'
	)

	args = parser.parse_args()

	if args.command == 'generate-data':
		cmd_generate_data(args)
	elif args.command == 'build-dataset':
		cmd_build_dataset(args)
	elif args.command == 'train':
		cmd_train(args)
	elif args.command == 'eval':
		cmd_eval(args)
	elif args.command == 'expand':
		cmd_expand(args)
	elif args.command == 'expand-batch':
		cmd_expand_batch(args)
	else:
		parser.print_help()


if __name__ == '__main__':
	main()
