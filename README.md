# Tokenizer From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Parveshiiii/microtok/blob/main/assets/Tokenizer_from_scratch.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

A highly efficient, customizable library for training **Byte Pair Encoding (BPE)** and **TikToken (GPT-4 style)** tokenizers from scratch. Designed for ease of use, it leverages Hugging Face's streaming datasets to train on massive corpuses (like `FineWeb-Edu`) with minimal memory overhead. **Train a production-ready tokenizer in only 4 lines of code.**

## Features

*   **Dual Modes**: Support for both standard **BPE** (BERT/RoBERTa style) and **TikToken** (GPT-4/cl100k style) training.
*   **Memory Efficient**: Uses streaming datasets and batch iterators to train on huge datasets without loading them entirely into RAM.
*   **Fully Customizable**: Easily configure vocabulary size, special tokens, and batch sizes.
*   **Hugging Face Compatible**: Exports tokenizers directly to the `tokenizer.json` format, compatible with `transformers`.

## Installation

You can install this package directly from GitHub using `pip`:

```bash
pip install git+https://github.com/Parveshiiii/microtok.git
```

## Quick Start (Train in 4 Lines)

Train a production-ready tokenizer with best-practice defaults (FineWeb-Edu dataset, 64k Vocab, 10k Batch Size) in just 4 lines of code:

```python
from microtok import BPETrainer, batch_iterator

# Automatically streams FineWeb-Edu and trains a 64k vocab BPE tokenizer
BPETrainer(batch_iterator()) 
```

## Advanced Usage

### 1. Training a BPE Tokenizer
The standard BPE implementation includes automatic `[CLS]` and `[SEP]` token injection (Template Processing).

```python
from microtok import BPETrainer, batch_iterator

# Initialize the data stream (defaults to FineWeb-Edu 10BT sample)
# Batches are yielded to keep memory usage low
batch_iter = batch_iterator(BATCH_SIZE=10_000)

# Start Training
BPETrainer(
    batch_iter, 
    vocab_size=64_000, 
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    save_to="my_bpe_tokenizer"
)
```

### 2. Training a TikToken (GPT-4) Tokenizer
This module replicates the **GPT-4 (cl100k_base)** pre-tokenization logic using specific Regex patterns and ByteLevel encoding.

```python
from microtok import TikTokenTrainer, batch_iterator

# Initialize data stream
batch_iter = batch_iterator(BATCH_SIZE=10_000)

# Train using TikToken logic
TikTokenTrainer(
    batch_iter,
    vocab_size=64_000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    save_to="my_tiktoken_tokenizer"
)
```

## Configuration

### batch_iterator
Handles the streaming of data from Hugging Face.
*   `Dataset` (default: `"HuggingFaceFW/fineweb-edu"`): The HF dataset ID.
*   `BATCH_SIZE` (default: `10,000`): Number of text samples per batch. 10k is optimized for ~24GB RAM environments.
*   `split`: Dataset split to use (e.g., `"train"`).

### Trainer
The main training entry point.
*   `vocab_size`: Target vocabulary size (default `64,000`).
*   `special_tokens`: List of special tokens. The tokenizer automatically detects mapping (e.g., `<cls>` or `[CLS]`).
*   `save_to`: Directory/filename prefix for the saved tokenizer.

## Loading and Using the Tokenizer

Once trained, your tokenizer is saved as a JSON file (and a folder for Hugging Face compatibility). You can load it easily with the `transformers` library:

```python
from transformers import PreTrainedTokenizerFast

# Load the tokenizer (replace 'tokenizer_tiktoken' with your save_to name)
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_tiktoken")

# Encode text
text = "Hello, world! This is a test."
encoded = tokenizer(text)
print(f"Tokens: {encoded.tokens()}")
print(f"IDs: {encoded['input_ids']}")

# Decode back to string
decoded = tokenizer.decode(encoded['input_ids'])
print(f"Decoded: {decoded}")
```

## Project Structure

*   **BPE/**: Contains the logic for standard Byte-Pair Encoding training.
*   **TikToken/**: Implements the specific Regex-split BPE used by models like GPT-4.
*   **main.py**: A simple entry point script for testing.
