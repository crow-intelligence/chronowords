# chronowords

Detect semantic shifts over time in word embeddings. Train small PPMI-based language models, create topic models using NMF, and analyze semantic changes using Procrustes alignment.

## Features

- Memory-efficient word embedding training using Count-Min Sketch
- Topic modeling with Non-negative Matrix Factorization
- Temporal alignment of word embeddings using Procrustes analysis
- Cython-optimized PPMI matrix computation

## Installation

```bash
pip install chronowords
```

## Quick Start
```python
from chronowords.algebra import SVDAlgebra
from chronowords.topics import TopicModel

# Train word embeddings
model = SVDAlgebra(n_components=300)
model.train(your_corpus_iterator)

# Find similar words
similar = model.most_similar('computer')
for word in similar:
    print(f"{word.word}: {word.similarity:.3f}")

# Create topic model
topic_model = TopicModel(n_topics=10)
topic_model.fit(ppmi_matrix, vocabulary)
```

## Documentation
Full documentation available at ReadTheDocs.

## Requirements

Python ≥ 3.10
NumPy
SciPy
scikit-learn
Cython

## Contributing
Pull requests welcome. For major changes, open an issue first.

## License
MIT
