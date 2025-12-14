"""Training script using HuggingFace Trainer with ByT5."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
	AutoTokenizer,
	AutoModelForSeq2SeqLM,
	Seq2SeqTrainingArguments,
	Seq2SeqTrainer,
	DataCollatorForSeq2Seq,
)

from .config import load_config
from .callbacks import LoggingCallback


def set_seed(seed: int):
	"""Set random seeds for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


class ShorthandDataset:
	"""Dataset class for loading JSONL shorthand pairs."""

	def __init__(
		self,
		jsonl_path: str | Path,
		tokenizer: AutoTokenizer,
		max_input_length: int,
		max_output_length: int,
		task_prefix: str = 'expand: ',
	):
		self.tokenizer = tokenizer
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
		self.task_prefix = task_prefix

		# Load examples
		self.examples = []
		with open(jsonl_path, 'r', encoding='utf-8') as f:
			for line in f:
				if line.strip():
					self.examples.append(json.loads(line))

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		example = self.examples[idx]

		# Format input: "expand: {shorthand}"
		input_text = f'{self.task_prefix}{example["shorthand"]}'
		target_text = example['full']

		# Tokenize - return lists, not tensors (data collator will handle tensor conversion efficiently)
		inputs = self.tokenizer(
			input_text,
			max_length=self.max_input_length,
			padding='max_length',
			truncation=True,
			return_tensors=None,  # Return lists, not tensors
		)

		targets = self.tokenizer(
			target_text,
			max_length=self.max_output_length,
			padding='max_length',
			truncation=True,
			return_tensors=None,  # Return lists, not tensors
		)

		return {
			'input_ids': inputs['input_ids'],
			'attention_mask': inputs['attention_mask'],
			'labels': targets['input_ids'],
		}


def compute_metrics(eval_pred):
	"""Compute evaluation metrics (placeholder for now)."""
	# This will be called during evaluation
	# For now, return empty dict (loss is computed automatically)
	return {}


def train(config_path: str | Path, device: str | None = None):
	"""Main training function.

	Args:
		config_path: Path to training config YAML
		device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detection)
	"""
	# Load config
	config = load_config(config_path)
	model_config = config['model']
	train_config = config['training']
	data_config = config['data']
	checkpoint_config = config['checkpointing']
	logging_config = config['logging']
	seed = config.get('seed', 42)

	# Set seeds
	set_seed(seed)

	# Load model and tokenizer
	model_name = model_config['name']
	print(f'Loading model: {model_name}')
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

	# Use provided device or auto-detect
	if device:
		# Validate device
		if device == 'cuda' and not torch.cuda.is_available():
			raise ValueError('CUDA requested but not available')
		if device == 'mps' and not torch.backends.mps.is_available():
			raise ValueError('MPS requested but not available')
		print(f'Using specified device: {device}')
	else:
		# Auto-detect and use best available device
		if torch.cuda.is_available():
			device = 'cuda'
			print('Using CUDA device (auto-detected)')
		elif torch.backends.mps.is_available():
			device = 'mps'
			print('Using MPS device (Apple Silicon GPU, auto-detected)')
		else:
			device = 'cpu'
			print('Using CPU device (auto-detected)')

	# Move model to device
	model = model.to(device)

	# Load datasets
	task_prefix = 'expand: '
	train_dataset = ShorthandDataset(
		data_config['train_path'],
		tokenizer,
		model_config['max_input_length'],
		model_config['max_output_length'],
		task_prefix,
	)

	val_dataset = ShorthandDataset(
		data_config['val_path'],
		tokenizer,
		model_config['max_input_length'],
		model_config['max_output_length'],
		task_prefix,
	)

	print(f'Train examples: {len(train_dataset)}')
	print(f'Val examples: {len(val_dataset)}')

	# Data collator
	data_collator = DataCollatorForSeq2Seq(
		tokenizer=tokenizer,
		model=model,
		padding=True,
	)

	# Training arguments
	training_args = Seq2SeqTrainingArguments(
		output_dir=checkpoint_config['output_dir'],
		num_train_epochs=int(train_config['num_epochs']),
		per_device_train_batch_size=int(train_config['batch_size']),
		per_device_eval_batch_size=int(train_config['batch_size']),
		gradient_accumulation_steps=int(train_config['gradient_accumulation_steps']),
		learning_rate=float(train_config['learning_rate']),
		warmup_steps=int(train_config['warmup_steps']),
		weight_decay=float(train_config['weight_decay']),
		max_grad_norm=float(train_config['max_grad_norm']),
		logging_steps=int(logging_config['logging_steps']),
		eval_steps=int(logging_config['eval_steps']),
		save_steps=int(logging_config['save_steps']),
		eval_strategy=checkpoint_config['save_strategy'],  # Match save and eval strategy for load_best_model_at_end
		save_strategy=checkpoint_config['save_strategy'],
		save_total_limit=checkpoint_config['save_total_limit'],
		load_best_model_at_end=True,
		metric_for_best_model='eval_loss',
		greater_is_better=False,
		predict_with_generate=True,
		fp16=False,  # Use fp32 for CPU compatibility
		# Set num_workers to 0 on MPS to avoid multiprocessing serialization issues
		dataloader_num_workers=0
		if (torch.backends.mps.is_available() or torch.cuda.is_available())
		else int(data_config['num_workers']),
		report_to='none',  # Disable wandb/tensorboard for now
	)

	# Trainer
	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		data_collator=data_collator,
		processing_class=tokenizer,  # Use processing_class instead of deprecated tokenizer
		compute_metrics=compute_metrics,
		callbacks=[LoggingCallback()],
	)

	# Train
	print('Starting training...')
	trainer.train()

	# Save final model
	print(f'Saving final model to {checkpoint_config["output_dir"]}')
	trainer.save_model()
	tokenizer.save_pretrained(checkpoint_config['output_dir'])

	print('Training complete!')
