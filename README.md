<p align="center">
  <img src="https://raw.githubusercontent.com/crow-intelligence/chronowords/main/img/chronowords.svg" alt="chronowords" width="450"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/chronowords/"><img src="https://img.shields.io/pypi/v/chronowords.svg" alt="PyPI"></a>
  <a href="https://chronowords.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/chronowords" alt="Docs"></a>
</p>

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

# Train word embeddings on any iterable of text lines
# (a list, a generator, or an open file).
model = SVDAlgebra(n_components=300)
with open("corpus.txt", encoding="utf-8") as fh:
    model.train(fh)

# Find similar words
for hit in model.most_similar("computer", n=10):
    print(f"{hit.word}: {hit.similarity:.3f}")

# Topic model over the PPMI matrix that train() computed
topic_model = TopicModel(n_topics=10)
topic_model.fit(model._ppmi_sparse, model.vocabulary)
topic_model.print_topics()
```

See the [quickstart](https://chronowords.readthedocs.io/en/latest/quickstart.html)
for a complete runnable example and the
[tutorial](https://chronowords.readthedocs.io/en/latest/tutorial.html) for
detecting semantic shift across time slices.

## Links
- Documentation: <https://chronowords.readthedocs.io/en/latest/>
- PyPI: <https://pypi.org/project/chronowords/>

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

## Made by
Built and maintained by [Crow Intelligence](https://crowintelligence.org/).
