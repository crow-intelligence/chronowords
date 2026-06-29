# Mutation Testing Summary

**Date:** 2026-06-29
**Scope:** numerical core — `probabilistic_counter.py`, `alignment/procrustes.py`,
`topics/nmf.py`, `algebra/svd.py`.

## Method

`mutmut` (v3) was added as instructed but does not work out of the box with this
project's `src/` layout + editable install + Cython extension: it runs the suite
from its generated `mutants/` copy where `chronowords.algebra` is not importable
(`ModuleNotFoundError`). Rather than ship broken tooling config, the dependency
and `[tool.mutmut]` config were removed, and a **guided mutation pass** was run
instead — the method the `/mutation-testing` skill actually prescribes: apply one
mutation, run the covering tests, record killed/survived, revert
(`git checkout`). Nine high-value mutations were chosen from the catalogue
(boundary flips, operator swaps, deleted guards, swapped estimators).

## Results

### Before adding tests: 5 / 9 killed (56%)

| # | File | Mutation | Result |
|---|------|----------|--------|
| 1 | probabilistic_counter.py | `CMS.query`: `np.min` → `np.max` | **survived** |
| 2 | probabilistic_counter.py | `CMS.merge`: `+=` → `-=` | killed |
| 3 | probabilistic_counter.py | `get_heavy_hitters`: `>` → `>=` | **survived** |
| 4 | probabilistic_counter.py | `CMS.update`: `total +=` → `-=` | killed |
| 5 | procrustes.py | `find_common_words`: `intersection` → `union` | **survived** |
| 6 | procrustes.py | `transform`: not-fitted guard inverted | killed |
| 7 | nmf.py | `align_with`: `>=` → `>` (min_similarity) | **survived** |
| 8 | svd.py | `most_similar`: `!=` → `==` (exclude self) | killed |
| 9 | svd.py | `distance`: `min` → `max` clamp | killed |

### After adding targeted tests: 8 / 9 killed (89%)

Three new tests close the worthwhile gaps; all three mutations are now killed
(re-verified):

- `test_query_returns_minimum_across_rows` — kills #1. The previous tests could
  not distinguish `min` from `max` because, without forced collisions,
  `min == max`. This white-box test sets distinct per-row counters and asserts
  the minimum is returned (the defining CMS estimator).
- `test_get_heavy_hitters_threshold_is_strict` — kills #3. Asserts a key whose
  count exactly equals `threshold * total` is excluded (strict `>`).
- `test_find_common_words_returns_intersection` — kills #5. The fit-based tests
  missed it because `fit` silently skips words absent from a vocabulary, so
  `union` behaved like `intersection` there; the direct unit test pins the
  contract.

### Surviving mutation (left intentionally)

- #7 `align_with` `>=` → `>`: triggers only when a topic-pair similarity equals
  `min_similarity` *exactly*. With floating-point similarities this boundary is
  effectively unreachable, so a test would be brittle (asserting exact float
  equality). Documented rather than tested.

## Notes

- The Cython kernel `count_skipgrams.pyx` is not covered by this pass (mutmut
  mutates `.py` only); its non-negativity contract is covered by a new
  property-based test in the test-hardening PR.
- If the maintainer wants automated mutation runs in CI, `mutmut` would need
  `src`-layout configuration (or `cosmic-ray`); that is a follow-up, not done
  here.
