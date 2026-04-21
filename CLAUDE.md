# chronowords

Detect semantic shifts in word embeddings over time. PPMI-based small LMs, NMF topic models, and Procrustes alignment across time slices. Cython-accelerated PPMI. Python 3.10–3.12.

## Layout

- `src/chronowords/algebra/svd.py` — `SVDAlgebra` (PPMI + truncated SVD embeddings, `most_similar`).
- `src/chronowords/alignment/procrustes.py` — Orthogonal Procrustes alignment between two embedding spaces.
- `src/chronowords/topics/nmf.py` — `TopicModel` (NMF over PPMI).
- `src/chronowords/utils/probabilistic_counter.py` — Count-Min Sketch for memory-efficient counting.
- `src/chronowords/utils/count_skipgrams.pyx` — Cython kernel for skip-gram counts.
- `tests/` — pytest suite mirroring the package layout, with shared fixtures in `tests/conftest.py`.
- `examples/presidential_speeches.ipynb` — end-to-end usage demo.

## Tooling (preferred stack)

- **Package management + Python version**: `uv`. Use `uv sync` to install, `uv add <pkg>` / `uv add --group dev <pkg>` for deps, `uv run <cmd>` to execute inside the project env, and `uv python install <ver>` / `uv venv --python <ver>` to pin the interpreter. Do not use pip/poetry/pipenv.
- **Type checking**: `ty` (Astral). Prefer `uv run ty check` over mypy for new work. Note: the repo currently still wires mypy into pre-commit and CI — when touching type-check config, migrate to `ty` rather than extending the mypy setup.
- **Testing**: `pytest` + `hypothesis`. Use hypothesis strategies for property-based tests whenever inputs have structure (vectors, matrices, vocabularies, counts). Run with `uv run pytest`. `hypothesis` is not yet a dependency — add it via `uv add --group dev hypothesis` the first time you need it.
- **Formatting + linting**: `ruff` only (`ruff format` + `ruff check`). Config already lives in `pyproject.toml` under `[tool.ruff]` — respect the existing rule set (line-length 88, target py310, isort force-single-line, `known-first-party = ["chronowords"]`). Do not introduce black, isort, flake8, or pylint.

## Commands

```bash
uv sync --all-groups        # install project + dev + docs deps
uv run pytest               # run tests
uv run ruff format .        # format
uv run ruff check . --fix   # lint + autofix
uv run ty check src tests   # type-check (preferred)
```

## Conventions

- Docstrings: pydocstyle is on via ruff `D`. Module/package/`__init__` docstrings and `D203`/`D213`/`D200`/`D205` are intentionally ignored — follow the existing style, don't re-enable them.
- Imports: single-line (`force-single-line = true`), two blank lines after imports.
- Quotes: double. Don't hand-edit quote style; let `ruff format` own it.
- Numpy/scipy are core deps; sparse matrices (`scipy.sparse.csr_matrix`) are the expected shape for PPMI.
- The Cython extension (`count_skipgrams.pyx`) is built via `setup.py` / `build-system` in `pyproject.toml`. Rebuild after edits with `uv sync --reinstall-package chronowords`.
