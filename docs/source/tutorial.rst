Tutorial: Detecting Semantic Shift Over Time
============================================

The flagship workflow in chronowords is measuring how a word's meaning changes
between time periods. The recipe is:

#. Split your corpus into time slices (decades, reigns, regimes — whatever suits
   the question).
#. Train a separate :class:`~chronowords.algebra.svd.SVDAlgebra` embedding space
   on each slice.
#. Align the slices into a common space with
   :class:`~chronowords.alignment.procrustes.ProcrustesAligner` (embedding spaces
   from independent SVD runs are not directly comparable — axes are arbitrary).
#. Score each shared word by how stable its aligned vector is across slices.
#. Optionally, model topics per slice and align them to see how themes evolve.

The bundled ``examples/presidential_speeches.ipynb`` notebook works this through
end-to-end on U.S. presidential speeches grouped by quarter-century. The same
recipe drove Crow Intelligence's study of gender bias in Hungarian media, where
the question was how the words used around women shifted across decades.

This tutorial uses two small synthetic slices so it runs quickly; swap in your
own per-slice corpora to reproduce it for real.

1. Train one embedding space per slice
--------------------------------------

Each slice is just an iterable of text lines. Train one model per slice with the
**same** ``n_components`` so the spaces are the same dimensionality.

.. code-block:: python

   from chronowords.algebra import SVDAlgebra

   def train_slice(lines):
       model = SVDAlgebra(n_components=8, cms_width=10_000, cms_depth=4)
       model.train(iter(lines))
       return model

   model_early = train_slice(corpus_1900s)   # your slice-1 lines
   model_late = train_slice(corpus_1990s)    # your slice-2 lines

For a real study, read each slice from disk::

   def load_slice(path):
       with open(path, encoding="utf-8") as fh:
           return list(fh)

2. Align the two spaces
-----------------------

:class:`~chronowords.alignment.procrustes.ProcrustesAligner` learns the
orthogonal rotation that best maps the source space onto the target space, using
words shared by both vocabularies as anchors. The frequency-rank window
(``min_freq_rank`` / ``max_freq_rank``) selects stable, frequent anchors and
skips the rare tail.

.. code-block:: python

   from chronowords.alignment import ProcrustesAligner

   aligner = ProcrustesAligner(min_freq_rank=0, max_freq_rank=1000)
   metrics = aligner.fit(
       model_early.embeddings,
       model_late.embeddings,
       model_early.vocabulary,
       model_late.vocabulary,
   )
   print(f"anchored on {metrics.num_aligned_words} words")
   print(f"mean anchor cosine after alignment: {metrics.average_cosine_similarity:.3f}")

A high ``average_cosine_similarity`` means the two spaces aligned well on the
anchors; a low value means the slices are hard to compare (too few shared words,
or genuinely divergent usage).

.. note::

   :meth:`~chronowords.alignment.procrustes.ProcrustesAligner.fit` raises
   ``ValueError`` if there are no usable anchors. That usually means the two
   slices share too few frequent words — widen ``max_freq_rank`` or check that
   both vocabularies were built.

3. Score how much each word shifted
-----------------------------------

:meth:`~chronowords.alignment.procrustes.ProcrustesAligner.get_word_similarity`
returns the cosine similarity between a word's early and (aligned) late vectors.
**High similarity means the word stayed stable; low similarity flags a semantic
shift** — the words to investigate.

.. code-block:: python

   shared = [w for w in model_early.vocabulary if w in model_late.vocabulary]

   shifts = []
   for word in shared:
       sim = aligner.get_word_similarity(
           word, model_early.embeddings, model_late.embeddings
       )
       if sim is not None:
           shifts.append((word, sim))

   # Smallest similarity == largest shift.
   shifts.sort(key=lambda pair: pair[1])
   print("most shifted words:")
   for word, sim in shifts[:10]:
       print(f"  {word}: stability {sim:.3f}")

4. Interpret a shift
--------------------

A low stability score tells you *that* a word moved, not *how*. Inspect each
word's neighbours in both slices to read the change:

.. code-block:: python

   word = shifts[0][0]
   print(f"{word!r} early neighbours:",
         [h.word for h in model_early.most_similar(word, n=8)])
   print(f"{word!r} late neighbours:",
         [h.word for h in model_late.most_similar(word, n=8)])

If the neighbour sets differ markedly, the word is being used in a new context —
the semantic shift you set out to find.

5. (Optional) Track topics across slices
----------------------------------------

To see how whole themes evolve rather than single words, fit a
:class:`~chronowords.topics.nmf.TopicModel` per slice and align them. Topic
alignment uses the Hungarian algorithm to pair each source topic with its closest
target topic.

.. code-block:: python

   from chronowords.topics import TopicModel

   topics_early = TopicModel(n_topics=10)
   topics_early.fit(model_early._ppmi_sparse, model_early.vocabulary)

   topics_late = TopicModel(n_topics=10)
   topics_late.fit(model_late._ppmi_sparse, model_late.vocabulary)

   for pair in topics_early.align_with(topics_late):
       early_words = [w for w, _ in pair.source_topic.words[:5]]
       late_words = [w for w, _ in pair.target_topic.words[:5]]
       print(f"sim={pair.similarity:.2f}  {early_words}  ->  {late_words}")

Pairs with low similarity are topics whose vocabulary changed most between
slices.

Where to go next
----------------

- ``examples/presidential_speeches.ipynb`` — the full pipeline with real data and
  Altair visualisations.
- :doc:`troubleshooting` — what the common errors mean.
- :doc:`api` — full signatures and contracts for every method used above.
