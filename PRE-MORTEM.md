# Pre-Mortem Report

**Scope:** `src/chronowords/` — `algebra/svd.py`, `alignment/procrustes.py`,
`topics/nmf.py`, `utils/probabilistic_counter.py`, `utils/count_skipgrams.pyx`
**Date:** 2026-06-29

## Summary

chronowords is a small, mostly-correct numerical library. Its fragilities are
not bugs today; they are places where a reasonable future edit would break
something non-obviously. The dominant themes are **silent fallbacks that hide
failure** (a noise-injecting dense-SVD fallback, several broad `except` blocks,
`None`/empty returns that collapse distinct error conditions) and **invisible
invariants between parallel data structures** (vocabulary↔embedding row order,
the asymmetric PPMI smoothing, magic count thresholds). Eight post-mortems
follow. A final section lists fixes that were identified but **not applied**
because they would change public behaviour — these need a human decision.

## Post-Mortems

### 1. Embeddings silently became non-deterministic noise

**Severity:** High
**Component:** `algebra/svd.py` — `SVDAlgebra.train` (the `try/except` around `svds`)
**Fragility type:** Coincidental correctness / silenced errors

#### What happened
A downstream user reported that `most_similar` returned different neighbours on
identical reruns of the same corpus, and that a CI snapshot test for embeddings
became flaky. Investigation traced it to the training step, which had quietly
stopped using sparse SVD for a class of inputs.

#### The change that caused it
A developer tuning memory use lowered the default `cms_width`, which made the
PPMI matrix smaller and more often rank-deficient. They tested that training
"still worked" (it did — no error) and shipped it.

#### Why it broke
`train` wraps `svds(M_float, k=k)` in `except Exception`, and on any failure
falls back to a dense SVD of `M_dense += np.random.normal(0, 1e-10, ...)`. The
fallback injects random noise and is never logged, so smaller matrices silently
shifted onto the noisy path. Two reasonable-looking inputs now produce different
embeddings.

#### How it was caught
Only by a flaky downstream test. Nothing in chronowords surfaces the fallback —
no log, no flag, no metric. With no such test it would have shipped as a quiet
quality regression.

#### Hardening suggestions
- Log a warning when the dense fallback triggers (observability; smallest fix).
- Seed the noise (`np.random.default_rng(seed)`) so the fallback is at least
  reproducible.
- Narrow `except Exception` to the specific exception `svds` raises
  (`scipy.sparse.linalg.ArpackError`/`ValueError`) so unrelated bugs surface.

### 2. PPMI values changed when the "asymmetric" smoothing was "fixed"

**Severity:** High
**Component:** `utils/count_skipgrams.pyx` — `compute_ppmi_batch`
**Fragility type:** Invisible invariant / assumptions baked into a transform

#### What happened
After a refactor, all downstream embeddings and topics shifted noticeably and a
paper's numbers stopped reproducing, even though "nothing functional changed".

#### The change that caused it
A contributor noticed that the context-distribution smoothing exponent `alpha`
is applied to the focus word's probability (`pa = pow(p + eps, alpha)`) but not
the context word's (`pb = p + eps`). Believing it a copy-paste bug, they made
`pb` symmetric.

#### Why it broke
The asymmetry is the intended SGNS-style context smoothing (Levy & Goldberg):
applying `alpha` to only one side is a deliberate part of the PPMI definition.
"Fixing" it silently redefined every PPMI value the library produces.

#### How it was caught
A numerical regression test on a fixed corpus would have caught it immediately;
without one, only an end user noticing changed results would.

#### Hardening suggestions
- Add a comment at the `pa`/`pb` lines stating the asymmetry is intentional and
  citing the reference.
- Add a small golden-value test: assert a known `(word, context)` PPMI entry on
  a fixed mini-corpus, so any change to the formula fails loudly.

### 3. `estimate_error(confidence=...)` never honoured `confidence`

**Severity:** Medium
**Component:** `utils/probabilistic_counter.py` — `CountMinSketch.estimate_error`
**Fragility type:** Coincidental correctness (dead parameter)

#### What happened
A user tightened `confidence` from 0.95 to 0.999 expecting a larger error bound
and got the identical number, then filed a bug doubting the whole error model.

#### The change that caused it
No change was needed — the parameter has been inert since it was written.

#### Why it broke
The method computes `delta = pow(2.0, -depth)` and `delta = delta / confidence`,
then returns `epsilon * self.total`, discarding `delta` entirely. The return
value depends only on `width` and `total`; `confidence` and `depth` are computed
and thrown away. (This is a genuine current bug, documented in the docstring in
the docs PR and surfaced here per the pre-mortem rules.)

#### How it was caught
By a user comparing outputs across `confidence` values. No test exercises the
relationship.

#### Hardening suggestions
- Decide the intended formula and either use `delta` in the return value or drop
  the parameter. (Behaviour change — see "Needs human decision".)
- Add a test asserting the bound is monotonic in `confidence` once fixed.

### 4. Zero-norm vector turned a stability score into `nan`

**Severity:** Medium
**Component:** `alignment/procrustes.py` — `get_word_similarity`
**Fragility type:** Assumptions baked into a transform (no zero-norm guard)

#### What happened
A semantic-shift report listed several words with `nan` "stability", which broke
the sort and the downstream visualisation.

#### The change that caused it
A user fed embeddings that included a few all-zero rows (e.g. out-of-vocabulary
placeholders, or a sparser training run that left some words with no context).

#### Why it broke
`get_word_similarity` normalises with `vec / np.linalg.norm(vec)` and has no
zero-norm guard (unlike `most_similar`/`distance` in `svd.py`, which floor the
norm at 1e-10). A zero vector yields `nan`/`inf` and a silent `RuntimeWarning`,
not a `None` or an exception.

#### How it was caught
Only when `nan` propagated into a plot. The function's own contract suggests it
returns `float | None`, so callers don't expect `nan`.

#### Hardening suggestions
- Floor the norm at a small epsilon, or return `None` for zero-norm vectors, to
  match the sibling methods. (Behaviour change — see "Needs human decision".)

### 5. A real failure was reported as "topics are 0% similar"

**Severity:** Medium
**Component:** `topics/nmf.py` — `_compute_topic_similarity`
**Fragility type:** Silenced errors (broad `except Exception`)

#### What happened
Topic alignment quietly returned all-zero similarities for a dataset, sending
every topic below `min_similarity` so `align_with` returned an empty list — read
by the team as "no topics matched" rather than "something failed".

#### The change that caused it
A developer changed how `distribution` vectors are built (e.g. dtype or shape)
upstream, introducing an exception inside the cosine computation.

#### Why it broke
`_compute_topic_similarity` wraps its body in `except Exception: return 0.0`, so
a genuine error is indistinguishable from a true zero similarity. The empty
alignment looked like a modelling result, not a crash.

#### How it was caught
By manually re-running with the `except` removed. As written, the failure is
invisible.

#### Hardening suggestions
- Narrow the `except` to the specific numerical errors expected, and let others
  propagate. (Behaviour change — see "Needs human decision".)
- Log at warning level when the fallback returns 0.0.

### 6. Filtering the vocabulary corrupted every lookup

**Severity:** Critical
**Component:** `algebra/svd.py` — `vocabulary` / `embeddings` / `_vocab_index`
**Fragility type:** Invisible invariant between parallel structures

#### What happened
After a "clean up the vocabulary" change, `most_similar` and `get_vector` began
returning the wrong words — `get_vector("king")` returned the vector for some
unrelated token — but never raised.

#### The change that caused it
A contributor added post-training vocabulary filtering (dropping stopwords) by
editing `model.vocabulary` and calling `_build_vocab_index()`, without dropping
the corresponding rows of `model.embeddings`.

#### Why it broke
`embeddings[i]` is addressed by the position of a word in `vocabulary`, an
invariant maintained only by convention. `_build_vocab_index` rebuilds the
name→index map from the new (shorter) list, so indices now point at the wrong
embedding rows. Lengths silently disagree.

#### How it was caught
By eyeballing obviously-wrong neighbours. No assertion ties
`len(vocabulary) == embeddings.shape[0]`.

#### Hardening suggestions
- Add an assertion (or a property) enforcing `len(vocabulary) ==
  len(embeddings)` after any mutation.
- Provide a single supported method to subset a trained model that updates both
  structures together, instead of leaving them publicly mutable.

### 7. Empty PPMI matrix from a quietly raised count threshold

**Severity:** Medium
**Component:** `utils/count_skipgrams.pyx` — `> 5` count thresholds
**Fragility type:** Load-bearing magic constant

#### What happened
A user training on a modest domain corpus got a model whose `most_similar`
always returned `[]`. The PPMI matrix was entirely empty, but training reported
success.

#### The change that caused it
No code change — the user simply had a corpus where most words occur fewer than
six times after windowing. The hard-coded `if count > 5` (words) and
`if pair_count <= 5: continue` (pairs) filtered everything out.

#### Why it broke
The threshold `5` is duplicated in two places and is invisible to callers — it
is not a constructor argument and not documented at the API surface. A perfectly
valid small corpus silently produces a degenerate matrix.

#### How it was caught
By the user noticing empty results. `compute_ppmi_matrix_with_sketch` returns an
empty matrix without complaint.

#### Hardening suggestions
- Promote the threshold to a named module constant (single source of truth) and,
  ideally, a `PPMIComputer` parameter.
- Have `SVDAlgebra.train` warn when the PPMI matrix has zero non-zeros.

### 8. Reusing a `TopicModel` across datasets mixed two corpora

**Severity:** Medium
**Component:** `topics/nmf.py` — `TopicModel` (stateful `self.nmf`) and `align_with`
**Fragility type:** Implicit ordering / shared mutable state

#### What happened
A script that reused one `TopicModel` instance to fit several corpora in a loop
produced topics that blended vocabulary from different corpora, and
`get_document_topics` returned projections against the wrong fit.

#### The change that caused it
A developer "optimised" by constructing the model once outside the loop and
calling `fit` repeatedly, assuming `fit` fully resets state.

#### Why it broke
`fit` overwrites `topics`/`vocabulary`/`topic_word_matrix` but the sklearn `NMF`
object (`self.nmf`) and the `vocabulary` used later by `align_with` create
implicit coupling between a fit and the data it was fit on. Mixing instances
across corpora silently compares incompatible spaces.

#### How it was caught
By a reviewer noticing cross-corpus words in a single topic. Nothing enforces
"one model per corpus".

#### Hardening suggestions
- Document that a `TopicModel` instance is single-use per corpus, or reset
  `self.nmf` at the start of `fit`.
- Have `align_with` assert the two models were fit (already done) and consider
  warning when their vocabularies barely overlap.

## Themes and Recommendations

1. **Make silent fallbacks observable.** Items 1, 3, 5 all hide failure behind a
   default value. The cheapest systemic win is to `log.warning` on every
   fallback path and to narrow broad `except` blocks so only the expected error
   is swallowed.
2. **Encode the invisible invariants.** Items 2, 6, 7 depend on conventions that
   no test or assertion enforces (PPMI asymmetry, vocab↔embedding alignment, the
   `>5` threshold). Golden-value tests and a couple of `assert`s would convert
   these into loud failures. The property-based and mutation tests in the
   test-hardening phase target exactly these.
3. **Distinguish "no result" from "error".** Several methods return
   `None`/`[]`/`0.0` for both. Where a richer signal is acceptable, prefer
   distinct outcomes; where not, document it (done in the docs PR).

## Needs Human Decision

These were identified by the design review but **not applied** because they
change public behaviour. Listed for the maintainer to decide.

- **`CountMinSketch.estimate_error` ignores `confidence`** (Post-mortem 3). It is
  a real bug, but any fix changes returned values. Decide the intended bound
  formula before changing it.
- **Narrow the broad `except Exception` blocks** in `SVDAlgebra.train` (svds
  fallback) and `TopicModel._compute_topic_similarity` (Post-mortems 1, 5).
  Narrowing is desirable but will let currently-swallowed exceptions surface —
  a behaviour change that needs sign-off and test coverage first.
- **Add a zero-norm guard to `get_word_similarity`** (Post-mortem 4). Changes a
  `nan` result to `None` or 0.0; pick the contract.
- **Add input validation** to constructors and `fit`/`train` (shapes, positive
  counts, `n_components`/`n_topics` ranges). Currently invalid inputs fail late
  with opaque NumPy/sklearn errors. Adding explicit checks changes which
  exception type callers see.
- **Promote the `>5` PPMI count threshold** to a constant/parameter
  (Post-mortem 7). Exposing it is an API addition.

### Applied in this PR (safe, behaviour-preserving)

- Added the missing return annotation `-> float | None` to
  `ProcrustesAligner.get_word_similarity` (it already returns `None` on a missing
  word; the annotation now matches the docstring and the code).
- Added `-> None` to `ProcrustesAligner.__init__`.
