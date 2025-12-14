"""Synthetic shorthand generator with stochastic transformations."""

import random
import re
import string
from typing import Dict, Optional

from .lexicon import apply_lexicon_replacements, load_lexicon


class ShorthandGenerator:
	"""Generate shorthand from full text using stochastic transformations."""

	def __init__(
		self,
		lexicon_path: Optional[str] = None,
		p_vowel_drop: float = 0.4,
		p_space_drop: float = 0.5,
		p_punct_drop: float = 0.8,
		use_consonant_skeleton: bool = False,
		min_compression_ratio: float = 0.3,
		max_compression_ratio: float = 0.7,
		seed: Optional[int] = None,
	):
		"""Initialize generator.

		Args:
			lexicon_path: Path to lexicon.json (optional)
			p_vowel_drop: Probability of dropping a vowel (0-1)
			p_space_drop: Probability of dropping a space (0-1)
			p_punct_drop: Probability of dropping punctuation (0-1)
			use_consonant_skeleton: Whether to use aggressive consonant skeleton compression
			min_compression_ratio: Minimum shorthand/original length ratio
			max_compression_ratio: Maximum shorthand/original length ratio
			seed: Random seed for reproducibility
		"""
		self.p_vowel_drop = p_vowel_drop
		self.p_space_drop = p_space_drop
		self.p_punct_drop = p_punct_drop
		self.use_consonant_skeleton = use_consonant_skeleton
		self.min_compression_ratio = min_compression_ratio
		self.max_compression_ratio = max_compression_ratio
		
		self.lexicon = {}
		if lexicon_path:
			self.lexicon = load_lexicon(lexicon_path)
		
		if seed is not None:
			random.seed(seed)
	
	def _is_vowel(self, char: str) -> bool:
		"""Check if character is a vowel."""
		return char.lower() in 'aeiouy'
	
	def _drop_vowels(self, text: str) -> str:
		"""Drop vowels with probability p_vowel_drop, preserving first char of words."""
		words = text.split()
		result_words = []
		
		for word in words:
			if len(word) <= 1:
				result_words.append(word)
				continue
			
			result = word[0]  # Always keep first character
			for i, char in enumerate(word[1:], 1):
				if self._is_vowel(char):
					if random.random() > self.p_vowel_drop:
						result += char
				else:
					result += char
			
			result_words.append(result)
		
		return ' '.join(result_words)
	
	def _drop_spaces(self, text: str) -> str:
		"""Drop spaces with probability p_space_drop."""
		result = []
		for char in text:
			if char == ' ':
				if random.random() > self.p_space_drop:
					result.append(char)
			else:
				result.append(char)
		return ''.join(result)
	
	def _drop_punctuation(self, text: str) -> str:
		"""Drop punctuation with probability p_punct_drop."""
		result = []
		for char in text:
			if char in string.punctuation:
				if random.random() > self.p_punct_drop:
					result.append(char)
			else:
				result.append(char)
		return ''.join(result)
	
	def _consonant_skeleton(self, text: str) -> str:
		"""Apply aggressive consonant skeleton compression."""
		words = text.split()
		result_words = []
		
		for word in words:
			if len(word) <= 2:
				result_words.append(word)
				continue
			
			# Keep first char, then consonants only
			result = word[0]
			for char in word[1:]:
				if not self._is_vowel(char):
					result += char
			
			result_words.append(result)
		
		return ' '.join(result_words)
	
	def _lowercase(self, text: str) -> str:
		"""Convert to lowercase."""
		return text.lower()
	
	def _filter_alphanumeric(self, text: str) -> str:
		"""Keep only alphanumeric characters and spaces."""
		return re.sub(r'[^a-z0-9 ]', '', text)
	
	def _validate_output(self, shorthand: str, full_text: str) -> bool:
		"""Validate that shorthand meets quality criteria.

		Args:
			shorthand: Generated shorthand
			full_text: Original full text

		Returns:
			True if valid, False otherwise
		"""
		# Check length ratios
		compression_ratio = len(shorthand) / len(full_text) if len(full_text) > 0 else 1.0
		if compression_ratio < self.min_compression_ratio or compression_ratio > self.max_compression_ratio:
			return False
		
		# Reject if identical
		if shorthand == full_text:
			return False
		
		# Reject if too short or too long
		if len(shorthand) < 2 or len(shorthand) > 50:
			return False
		
		# Reject if only vowels or only consonants (unrealistic)
		has_vowel = any(self._is_vowel(c) for c in shorthand)
		has_consonant = any(not self._is_vowel(c) and c.isalpha() for c in shorthand)
		if not (has_vowel and has_consonant) and len(shorthand) > 3:
			return False
		
		return True
	
	def generate(self, full_text: str, max_attempts: int = 10) -> Optional[str]:
		"""Generate shorthand from full text.

		Args:
			full_text: Input full text
			max_attempts: Maximum attempts to generate valid shorthand

		Returns:
			Generated shorthand or None if validation fails
		"""
		# Normalize input
		full_text = full_text.strip()
		if len(full_text) < 3:
			return None
		
		for _ in range(max_attempts):
			# Apply transformations in order
			result = full_text
			
			# 1. Apply lexicon replacements first
			if self.lexicon:
				result = apply_lexicon_replacements(result, self.lexicon)
			
			# 2. Lowercase
			result = self._lowercase(result)
			
			# 3. Consonant skeleton (if enabled)
			if self.use_consonant_skeleton:
				result = self._consonant_skeleton(result)
			else:
				# 4. Drop vowels
				result = self._drop_vowels(result)
			
			# 5. Drop punctuation
			result = self._drop_punctuation(result)
			
			# 6. Drop spaces
			result = self._drop_spaces(result)
			
			# 7. Filter to alphanumeric only
			result = self._filter_alphanumeric(result)
			
			# 8. Clean up multiple spaces
			result = re.sub(r'\s+', '', result)
			
			# Validate
			if self._validate_output(result, full_text):
				return result
		
		return None
