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
   - The PPMI step also needs pair counts above ``ppmi_min_count``
     (default 1). A corpus can therefore pass the vocabulary rule and still give
     an empty PPMI matrix, which logs a ``WARNING``.

``TypeError: corpus is a single string``
   ``train("some text")`` would iterate characters. Pass a list of lines, or a
   :class:`pathlib.Path` to read a file.

``ValueError: invalid UTF-8`` when training on a file
   Files are read as strict UTF-8. Re-encode the file (``iconv -f latin1 -t utf-8``)
   or read it in Python with the right encoding and pass the lines.

``most_similar`` returns ``[]`` / ``distance`` returns ``None``
   The model has not been trained, the word is not in the vocabulary, or the
   word is in the vocabulary but **has no vector**. These cases are deliberately
   non-raising. ``model.has_vector(word)`` tells the last case apart.

Some vocabulary words have no vector
   A word gets an all-zero vector when no pair it heads received a PPMI entry:
   every pair it occurs in was seen ``ppmi_min_count`` times or fewer, or is not
   positively associated. The model has no information about such a word, so
   ``has_vector`` is ``False``, ``most_similar`` never returns it, ``distance``
   and ``analogy`` return ``None``, and
   :meth:`~chronowords.alignment.procrustes.ProcrustesAligner.get_word_similarity`
   returns ``None``. ``training_record["zero_vectors"]`` counts them.

   They are typically discourse words that combine with everything
   (*moreover*, *presumably*) and, in small corpora, most of the vocabulary.
   Lowering ``ppmi_min_count`` keeps rarer pairs and gives most of them a
   vector. On English Wikipedia this also improves word-similarity scores
   (``experiments/evaluation``), at the cost of more memory; see
   ``max_memory``.

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

Measured end to end (``SVDAlgebra.train`` on a ``Path``, including SVD with
``n_components=100``) on a 4-core, 25 GB laptop, with default parameters
(``vocab_size=50_000``, ``min_count=100``, 4 threads):

=============================  ==========  =========  =======  ===========
corpus                         tokens      counting   total    peak memory
=============================  ==========  =========  =======  ===========
English Wikipedia, 100 MB      13 M        1.8 s      3.5 s    0.4 GB
Hungarian Wikipedia, 1 GB      102 M       25 s       33 s     2.2 GB
English Wikipedia, 1 GB        135 M       25 s       33 s     2.0 GB
English Wikipedia, 3 GB        405 M       81 s       91 s     3.4 GB
=============================  ==========  =========  =======  ===========

On 1 GB of Hungarian Wikipedia, pass 2 observes 91.6 M distinct vocabulary pairs.
``SVDAlgebra`` keeps the 8.0 M that can enter PPMI. The scripts and raw records
are in ``experiments/scaling`` in the repository.

For comparison, chronowords 0.3 trained on the same Hungarian gigabyte in 61
minutes, counting in pure Python at about 28,000 tokens per second. It peaked at only 0.74 GB,
because it kept fixed-size Count-Min Sketches and a vocabulary of just 1,084
words, and its PPMI matrix was dominated by hash-collision artefacts (see
``experiments/sketch_accuracy``).

Setting a memory budget
   Pass ``max_memory`` (bytes, or a string such as ``"2GB"``) to
   :class:`~chronowords.algebra.svd.SVDAlgebra` or
   :func:`~chronowords.utils.cooccurrence.count_cooccurrences` to cap the
   memory used for counting. Half the budget goes to a Count-Min Sketch with
   conservative update (the design of gensim's *bounter*), which estimates
   every word and pair count. The other half holds exact counts, but only for
   words and pairs whose estimate reaches the threshold.

   - **What it guarantees.** A Count-Min Sketch never under-estimates, so a
     word or pair it rules out truly falls below the threshold. Everything
     that is kept gets exactly the count it would get without a budget: **the
     PPMI matrix and the embeddings are identical**. A smaller budget only lets
     more false candidates into the exact table.
   - **What it costs.** Two extra passes over the corpus. Input that is not a
     file is first copied to a temporary file. ``counts.n_types`` (distinct
     word types) becomes ``None``.
   - **When it fails.** If the exact candidates still do not fit, training
     raises :class:`MemoryError` naming the sizes, rather than degrading
     silently. Raise ``max_memory``, ``min_count`` or ``ppmi_min_count``.
   - **When it helps.** Only when there is a threshold to filter on:
     ``SVDAlgebra`` keeps pairs seen more than ``ppmi_min_count`` times, so
     ``ppmi_min_count=0`` leaves nothing to filter.

   ``training_record["memory"]`` reports the sketch sizes and how many
   candidates were counted exactly versus kept.

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
