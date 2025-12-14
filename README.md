# Hypertype: Shorthand Expansion Model

An offline prototype model that expands shorthand strings into full text using ByT5-small fine-tuned on synthetic and gold data.

## Features

- **Offline Inference**: No external APIs required at inference time
- **Top-K Expansions**: Returns top-5 (configurable) expansion candidates with scores
- **Synthetic Data Generation**: Generates training data from multiple Hugging Face datasets or local corpus
- **Multi-Device Support**: Auto-detects and uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU
- **CPU-Friendly**: Optional quantization for faster CPU inference
- **Comprehensive Evaluation**: Top-1/Top-K accuracy, CER, Levenshtein distance metrics

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### 1. Generate Training Data

The default configuration uses multiple Hugging Face datasets for more diverse training data:

```bash
# Generate synthetic data (uses configs/data_config.yaml by default)
python cli.py build-dataset --config configs/data_config.yaml

# Or generate data manually
python cli.py generate-data \
  --dataset-name binhgiangnguyendanh/reddit_casual_conversation_for_alpaca_lora \
  --column-name output \
  --n 500000 \
  --output data/train.jsonl \
  --streaming
```

The default config uses:

- `binhgiangnguyendanh/reddit_casual_conversation_for_alpaca_lora` (output column)
- `clockwork7/reddit_news_articles_comments` (response column)

You can configure multiple datasets in `configs/data_config.yaml` with different weights.

### 2. Train the Model

```bash
# Auto-detect best device (MPS > CUDA > CPU)
python cli.py train --config configs/train_config.yaml

# Explicitly specify device (mps, cuda, or cpu)
python cli.py train --config configs/train_config.yaml --device mps
```

This will:

- Load ByT5-small from Hugging Face (downloads once, then cached)
- Auto-detect and use the best available device (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU)
- Train on the generated dataset
- Save checkpoints to `models/byt5-shorthand-v1/`

### 3. Evaluate

```bash
# Auto-detect device
python cli.py eval \
  --model models/byt5-shorthand-v1 \
  --test data/test.jsonl \
  --k 5

# Specify device explicitly
python cli.py eval \
  --model models/byt5-shorthand-v1 \
  --test data/test.jsonl \
  --k 5 \
  --device mps
```

### 4. Expand Shorthand

```bash
# Interactive expansion (auto-detects device)
python cli.py expand "ruavailrn" \
  --model models/byt5-shorthand-v1 \
  --k 5

# Batch expansion (auto-detects device)
python cli.py expand-batch \
  --input test_inputs.txt \
  --output predictions.jsonl \
  --model models/byt5-shorthand-v1 \
  --k 5

# Use quantized model for faster CPU inference
python cli.py expand-batch \
  --input test_inputs.txt \
  --output predictions.jsonl \
  --model models/byt5-shorthand-v1 \
  --k 5 \
  --quantize \
  --device cpu
```

## Project Structure

```
hypertype/
├── src/
│   ├── data/           # Data generation and loading
│   │   ├── generator.py          # Synthetic shorthand generator
│   │   ├── lexicon.py            # Phrase replacement mappings
│   │   ├── dataset_builder.py   # Dataset construction
│   │   └── gold_loader.py        # Gold data loading
│   ├── train/         # Training scripts
│   │   ├── trainer.py            # HuggingFace training
│   │   ├── config.py             # Config loading
│   │   └── callbacks.py         # Training callbacks
│   ├── eval/          # Evaluation
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── evaluator.py         # Evaluation runner
│   │   └── test_loader.py       # Test set loading
│   ├── inference/     # Inference API
│   │   ├── model_loader.py      # Model loading
│   │   ├── expander.py          # Main expand() API
│   │   ├── beam_search.py       # Beam search generation
│   │   └── quantize.py          # CPU quantization
│   └── utils/         # Utilities
│       └── normalization.py     # Text normalization
├── configs/          # Configuration files
│   ├── train_config.yaml
│   ├── data_config.yaml
│   └── model_config.yaml
├── data/             # Data files
│   ├── lexicon.json            # Phrase mappings
│   ├── train.jsonl             # Training data
│   ├── val.jsonl               # Validation data
│   └── test.jsonl              # Test data
├── models/           # Model checkpoints
├── tests/            # Unit tests
└── cli.py            # CLI interface
```

## Configuration

### Data Generation (`configs/data_config.yaml`)

- **Multiple Hugging Face Datasets**: Configure a list of datasets with `name`, `column`, `split`, `streaming`, and `weight` for each
- **Generator Parameters**: Vowel drop probability, space drop, punctuation drop, etc.
- **Dataset Splits**: Train/val/test ratios

Example multi-dataset configuration:

```yaml
dataset:
  datasets:
    - name: "dataset1/name"
      column: "output"
      split: "train"
      streaming: true
      weight: 1.0
    - name: "dataset2/name"
      column: "response"
      split: "train"
      streaming: true
      weight: 1.0
```

### Training (`configs/train_config.yaml`)

- **Model**: ByT5-small (300M parameters)
- **Batch Size**: 16 with gradient accumulation (effective batch = 64)
- **Learning Rate**: 5e-5 with 1000 warmup steps
- **Epochs**: 3

### Inference (`configs/model_config.yaml`)

- **Beam Search**: `num_beams=5`, `num_return_sequences=5`
- **Quantization**: Optional CPU quantization for 2-3x speedup

## API Usage

```python
from src.inference.expander import expand

# Expand shorthand (auto-detects device: MPS > CUDA > CPU)
results = expand(
    shorthand="ruavailrn",
    model_path="models/byt5-shorthand-v1",
    k=5,
    device=None,  # None = auto-detect, or specify "mps", "cuda", "cpu"
    quantize=False,  # Set to True for faster CPU inference
)

# Results: [{"text": "are you available right now", "score": -2.34}, ...]
for result in results:
    print(f"{result['text']} (score: {result['score']:.4f})")
```

## Evaluation Metrics

- **Top-1 Exact Match**: Exact match after normalization
- **Top-K Exact Match**: Is gold in top-K predictions?
- **Character Error Rate (CER)**: Edit distance / gold length
- **Levenshtein Distance**: Edit distance between prediction and gold
- **Normalized Edit Distance**: Edit distance / max(pred_len, gold_len)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_generator.py
pytest tests/test_metrics.py
```

## Data Format

Training data is stored in JSONL format:

```json
{"shorthand": "ruavailrn", "full": "are you available right now", "left": "", "right": ""}
{"shorthand": "whtaddr", "full": "what's the address", "left": "", "right": ""}
```

The `left` and `right` context fields are reserved for future use.

## Lexicon

The lexicon (`data/lexicon.json`) contains phrase-to-abbreviation mappings:

- `"right now" → "rn"`
- `"I don't know" → "idk"`
- `"are you" → "ru"`
- And 50+ more mappings

## Limitations

- English only
- No user personalization
- No IME integration (CLI-only)
- Requires initial model download (offline after that)

## License

[Add your license here]
