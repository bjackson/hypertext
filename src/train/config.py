"""Training configuration loading."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str | Path) -> Dict[str, Any]:
	"""Load YAML configuration file.

	Args:
		config_path: Path to YAML config file

	Returns:
		Configuration dictionary
	"""
	with open(config_path, 'r', encoding='utf-8') as f:
		return yaml.safe_load(f)
