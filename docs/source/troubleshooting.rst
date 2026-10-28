Troubleshooting
===============

Common errors and surprises, with the cause and the fix. These are drawn from
the documented contracts of each method (see :doc:`api`).

Training
--------

``ValueError: Empty vocabulary: no word occurs at least min_count=100 times``
   :meth:`~chronowords.algebra.svd.SVDAlgebra.train` keeps the ``vocab_size``
   most frequent words that occur at least ``min_count`` times (defaults 50,000
   and 100). The message names the most frequent word and its count.

   - For a small corpus, lower ``min_count`` (``SVDAlgebra(min_count=5)``).
   - Pass a fresh iterable. A generator or file handle is **consumed once**; if
     you iterate it before calling ``train`` (or call ``train`` twice on the same
     generator), training sees nothing.
   - The PPMI step also needs word and pair counts above ``ppmi_min_count``
     (default 5). A corpus can therefore pass the vocabulary rule and still give
     an empty PPMI matrix, which logs a ``WARNING``.

``TypeError: corpus is a single string``
   ``train("some text")`` would iterate characters. Pass a list of lines, or a
   :class:`pathlib.Path` to read a file.

``ValueError: invalid UTF-8`` when training on a file
   Files are read as strict UTF-8. Re-encode the file (``iconv -f latin1 -t utf-8``)
   or read it in Python with the right encoding and pass the lines.

``most_similar`` returns ``[]`` / ``distance`` and ``get_vector`` return ``None``
   The model has not been trained, the word is not in the vocabulary, or the
   word's vector has effectively zero norm. These cases are deliberately
   non-raising and are indistinguishable from each other. Check
   ``len(model.vocabulary)`` and ``word in model.vocabulary`` first.

Results change between identical runs
   Training is deterministic on the normal (sparse SVD) path. If sparse SVD fails
   on a small or degenerate matrix,
   :meth:`~chronowords.algebra.svd.SVDAlgebra.train` falls back to a dense SVD
   **with a tiny amount of random noise added**. It logs a ``WARNING`` and sets
   ``training_record["svd"] == "dense_fallback"``. A larger corpus avoids the
   fallback path.

Memory and time
---------------

Exact counting (the default) holds one entry per distinct word type in pass 1.
In pass 2 it holds one 8-byte entry per distinct vocabulary pair. Both grow with
the vocabulary and the variety of the text and level off as the corpus grows;
they do not grow with the corpus length. The PPMI matrix is sparse; no dense
``V x V`` matrix is ever built. ``SVDAlgebra`` keeps only the pairs that can
receive a PPMI entry.

Measured on a 4-core laptop (default parameters: ``vocab_size=50_000``,
``min_count=100``, 3 counting threads, other jobs running):

=============================  ==========  =========  ===============  ===========
corpus                         tokens      counting   distinct pairs   peak memory
=============================  ==========  =========  ===============  ===========
Hungarian Wikipedia, 1 GB      102 M       32 s       91.6 M           2.7 GB
English Wikipedia, 3 GB        405 M       152 s      not measured     2.8 GB
=============================  ==========  =========  ===============  ===========

For comparison, chronowords 0.3 counted the same Hungarian gigabyte in pure
Python at about 28,000 tokens per second: 61 minutes. It peaked at only 0.74 GB,
because it kept fixed-size Count-Min Sketches and a vocabulary of just 1,084
words, and its PPMI matrix was dominated by hash-collision artefacts (see
``experiments/sketch_accuracy``).

``counting="sketch"`` uses two
:class:`~chronowords.utils.probabilistic_counter.CountMinSketch` tables of
``cms_width * cms_depth * 4`` bytes each, plus the set of distinct words. Sketch
counts are never under-estimated, but at the default width they are inflated by
roughly ``skip-grams / cms_width`` per pair. That is about 18 on a 23 MB corpus,
enough to put collision artefacts above the PPMI count threshold (see
``experiments/sketch_accuracy``). Use it only with a ``cms_width`` well above the
number of skip-grams.

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

The compiled extension fails to import
   ``chronowords._core`` is a compiled Rust module (built by maturin). After
   editing ``rust/src/*.rs``, rebuild it with::

      uv sync --reinstall-package chronowords

   Building from source needs a Rust toolchain (``rustup``) on ``PATH``.
