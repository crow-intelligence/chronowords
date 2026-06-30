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

## Roadmap / further work

The following are known limitations and improvements not yet addressed. They are
documented in `PRE-MORTEM.md` (fragility analysis) and `CHANGES_SUMMARY.md`.

- **Python 3.13 support** — `requires-python` is currently `>=3.10,<3.13`, so
  chronowords cannot be installed on Python 3.13 even though the Cython extension
  builds and the test suite passes there in downstream (kenon) CI. Lift the cap,
  add 3.13 to the CI matrix and classifiers, validate against the test suite, and
  release. This unblocks downstream packages (e.g. kenon) that advertise 3.13.
- **Robustness / error reporting**
  - `CountMinSketch.estimate_error` currently ignores its `confidence` argument
    (the result depends only on `width` and `total`) — decide the intended bound
    and honour the parameter.
  - Narrow the broad `except Exception` blocks in `SVDAlgebra.train` (the silent,
    noise-injecting dense-SVD fallback) and `TopicModel._compute_topic_similarity`
    (returns `0.0` on any failure) so real errors surface; log when a fallback fires.
  - Add a zero-norm guard to `ProcrustesAligner.get_word_similarity` (it can return
    `nan` today).
- **Input validation** — validate constructor and `train`/`fit` inputs (array
  shapes, positive counts, `n_components` / `n_topics` ranges) so invalid input
  fails early with a clear message instead of an opaque NumPy/scikit-learn error.
- **Configurability** — promote the hard-coded minimum-count threshold (`> 5`) in
  the PPMI kernel to a named constant / parameter.
- **Determinism** — seed the dense-SVD fallback so embeddings are reproducible.
- **Tooling** — wire mutation testing into CI (needs `src`-layout configuration for
  `mutmut`, or an alternative such as `cosmic-ray`).
- **Coverage** — extend property-based and mutation testing to the Cython PPMI
  kernel and the NMF topic-alignment path.

## Contributing
Pull requests welcome. For major changes, open an issue first.

## License
MIT

## Made by
Built and maintained by [Crow Intelligence](https://crowintelligence.org/).
