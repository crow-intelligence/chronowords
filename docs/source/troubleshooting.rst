Troubleshooting
===============

Common errors and surprises, with the cause and the fix. These are drawn from
the documented contracts of each method (see :doc:`api`).

Training
--------

``ValueError: No words found meeting minimum frequency threshold``
   :meth:`~chronowords.algebra.svd.SVDAlgebra.train` builds its vocabulary from
   words above a 0.01% heavy-hitter threshold. This error means no word cleared
   it — usually the corpus is empty, far too small, or was already consumed.

   - Pass a fresh iterable. A generator or file handle is **consumed once**; if
     you iterate it before calling ``train`` (or call ``train`` twice on the same
     generator), the second pass sees nothing.
   - Give it more data. The internal PPMI step also requires word and skip-gram
     counts above 5, so very small corpora can train but yield a near-empty
     vocabulary.

``most_similar`` returns ``[]`` / ``distance`` and ``get_vector`` return ``None``
   The model has not been trained, the word is not in the vocabulary, or the
   word's vector has effectively zero norm. These cases are deliberately
   non-raising and are indistinguishable from each other. Check
   ``len(model.vocabulary)`` and ``word in model.vocabulary`` first.

Results change between identical runs
   If sparse SVD fails on a small or degenerate matrix,
   :meth:`~chronowords.algebra.svd.SVDAlgebra.train` silently falls back to a
   dense SVD **with a tiny amount of random noise added**, so embeddings can vary
   slightly run-to-run. A larger corpus avoids the fallback path.

Memory and accuracy (Count-Min Sketch)
--------------------------------------

The counter is a :class:`~chronowords.utils.probabilistic_counter.CountMinSketch`
sized by ``cms_width`` and ``cms_depth``.

- **Memory** is ``cms_width * cms_depth * 4`` bytes per sketch. The default
  ``cms_width=1_000_000`` is ~20 MB per sketch (two are used during training).
- **Accuracy**: counts are never underestimated but can be *over*-estimated on
  hash collisions. If rare words leak into the vocabulary, increase ``cms_width``.
  If memory is tight on a small corpus, lower it.

Saving and loading
------------------

``FileNotFoundError`` when calling ``load_model``
   The directory is missing ``embeddings.npy`` or ``vocabulary.pkl``. The most
   common cause is calling
   :meth:`~chronowords.algebra.svd.SVDAlgebra.save_model` **before** ``train``:
   only attributes that are set are written, so an untrained model saves an empty
   ``vocabulary.pkl`` and no embeddings. Train before saving.

``ValueError: Directory not found``
   :meth:`~chronowords.algebra.svd.SVDAlgebra.load_model` was given a path that
   is not an existing directory. ``save_model`` writes a *directory* of files, not
   a single file — pass that directory to ``load_model``.

Loading executes code
   Both ``SVDAlgebra.load_model`` and
   :meth:`~chronowords.alignment.procrustes.ProcrustesAligner.load` unpickle
   their input, which can run arbitrary code. Only load files you produced or
   trust.

Alignment
---------

``ValueError: No common words found for alignment`` / ``No valid anchor words found``
   :meth:`~chronowords.alignment.procrustes.ProcrustesAligner.fit` could not
   assemble anchors. Either the two vocabularies share too few words in the
   frequency-rank window, or every candidate anchor was dropped for being missing
   from one space or having a near-zero vector. Widen ``max_freq_rank``, or pass
   an explicit ``anchor_words`` list of words you know are in both vocabularies.

``ValueError: Aligner must be fit before transform``
   Call :meth:`~chronowords.alignment.procrustes.ProcrustesAligner.fit` before
   :meth:`~chronowords.alignment.procrustes.ProcrustesAligner.transform`.

``get_word_similarity`` raises or returns ``nan``
   It returns ``None`` if the word is missing from either vocabulary, but it does
   **not** guard against zero-norm vectors (a zero vector yields ``nan`` and a
   ``RuntimeWarning``) and it assumes the aligner has been fit. Make sure the word
   is shared and the aligner is fitted.

Topic modeling
--------------

``ValueError`` from ``TopicModel.fit``
   Raised by scikit-learn's NMF when ``n_topics`` exceeds the matrix dimensions,
   or if the matrix has negative entries. PPMI matrices are non-negative, so the
   usual cause is too many topics for a small vocabulary — reduce ``n_topics``.

``ValueError: Model must be fit before getting document topics``
   Call :meth:`~chronowords.topics.nmf.TopicModel.fit` before
   :meth:`~chronowords.topics.nmf.TopicModel.get_document_topics`.

Empty or all-zero topics
   On a tiny PPMI matrix, NMF can produce degenerate components whose weights sum
   to zero; chronowords leaves those distributions unnormalised rather than
   raising. Use a larger corpus for meaningful topics.

Build / install
---------------

The Cython extension fails to import
   ``chronowords.utils.count_skipgrams`` is a compiled Cython module. After
   editing the ``.pyx`` source, rebuild it with::

      uv sync --reinstall-package chronowords

   A C/C++ compiler must be available on the build machine.
