# Chronowords

A Python library for analyzing how word meanings change over time using word embeddings. Built for processing large text corpora efficiently using probabilistic data structures.

## Features

- Memory-efficient processing of large text corpora using Count-Min Sketch
- SVD-based word embeddings with PPMI (Positive Pointwise Mutual Information)
- Fast computation using Cython optimizations
- Support for word similarity and analogy operations
- Memory-bounded processing suitable for very large datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chronowords.git
cd chronowords

# Install with Poetry
poetry install
```

## Quick Start

```python
from chronowords.algebra.svd import SVDAlgebra

# Initialize and train the model
model = SVDAlgebra(
    n_components=300,    # Number of dimensions for word vectors
    window_size=5,       # Context window size
    min_word_length=3    # Minimum word length to consider
)

# Train on your corpus
def read_corpus():
    with open('your_corpus.txt', 'r') as f:
        for line in f:
            yield line.strip()

model.train(read_corpus())

# Find similar words
similar_words = model.most_similar("freedom", n=10)
print(similar_words)

# Calculate word distances
distance = model.distance("freedom", "liberty")
print(f"Distance: {distance}")

# Find analogies
result = model.analogy(
    positive=["king", "woman"],
    negative="man"
)
print(result)
```

## Memory Usage

The library uses Count-Min Sketch for efficient counting with bounded memory usage. You can control memory usage with these parameters:

```python
model = SVDAlgebra(
    cms_width=1_000_000,  # Width of Count-Min Sketch tables
    cms_depth=10          # Number of hash functions
)
```

These settings use approximately:
- Memory: cms_width * cms_depth * 4 bytes
- Error bound: ≈ 2/cms_width with probability 1 - 1/2^cms_depth

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
pytest

# Build Cython extensions
poetry build
```

## Requirements

- Python 3.8+
- numpy
- scipy
- datasketch (for probabilistic counting)
- cython (for performance optimizations)

## Usage in Your Work

If you use this library in your work, please reference this repository:

```
https://github.com/crow-intelligence/chronowords
```

The code in this library is partially based on techniques described in ["Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change" (Hamilton et al., 2016)](https://arxiv.org/abs/1605.09096).

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.