Quickstart
==========

This page walks through a complete, runnable example: building word embeddings
from a corpus, finding similar words, and extracting topics. Every snippet below
runs as-is on a clean install — copy them into a Python session in order.

.. note::

   The corpus here is deliberately tiny so the example runs instantly. Embedding
   quality scales with corpus size, so the specific neighbours returned below are
   illustrative, not meaningful. For a realistic, end-to-end study see
   :doc:`tutorial` and the bundled
   ``examples/presidential_speeches.ipynb`` notebook.

Install
-------

.. code-block:: bash

   pip install chronowords

See :doc:`installation` for installing from source.

Build embeddings from a corpus
------------------------------

:class:`~chronowords.algebra.svd.SVDAlgebra` trains PPMI-weighted word
embeddings. It counts every word and skip-gram exactly, in two passes, in its
compiled core. It keeps the ``vocab_size`` most frequent words that occur at
least ``min_count`` times (defaults: 50,000 and 100), builds a sparse PPMI
matrix and factorises it with truncated SVD.

.. code-block:: python

   from chronowords.algebra import SVDAlgebra

   # Any iterable of strings works — a list, a generator, or an open file.
   animal = [
       "the cat chased the dog around the garden",
       "the dog and the cat played near the rabbit",
       "the rabbit and the dog ran past the cat",
       "a cat watched the rabbit and the dog",
   ]
   royal = [
       "the king ruled the kingdom beside the queen",
       "the queen and the king visited the kingdom",
       "the prince served the king and the queen",
       "the kingdom welcomed the queen and the prince",
   ]
   corpus = (animal * 15) + (royal * 15)

   # min_count defaults to 100, which suits real corpora; this toy corpus
   # repeats each sentence 15 times, so keep words seen at least 10 times.
   model = SVDAlgebra(n_components=10, min_count=10)
   model.train(corpus)

   print(f"vocabulary size: {len(model.vocabulary)}")

To train on files, pass a :class:`pathlib.Path` (or a list of them). Each line
is one context, and skip-grams never cross a line break. The files are read in
the compiled core, which is the fastest route for large corpora::

   from pathlib import Path

   model.train(Path("corpus.txt"))
   model.train([Path("part1.txt"), Path("part2.txt")])

Already tokenised (lemmatised, say)? Pass token lists instead of strings::

   model.train([["the", "king", "rule"], ["the", "queen", "rule"]])

A generator or an open file handle also works. It is read once and copied to a
temporary file for the second pass, so it needs free disk space about the size
of the corpus.

Query the embeddings
--------------------

.. code-block:: python

   # Nearest neighbours by cosine similarity.
   for hit in model.most_similar("king", n=3):
       print(f"{hit.word}: {hit.similarity:.3f}")

   # Cosine distance between two words (None if either is unknown).
   print(model.distance("cat", "dog"))

   # The raw vector for a word (None if the word is not in the vocabulary).
   vec = model.get_vector("queen")

Unknown words never raise: :meth:`~chronowords.algebra.svd.SVDAlgebra.most_similar`
returns an empty list and :meth:`~chronowords.algebra.svd.SVDAlgebra.distance`
returns ``None``.

Audit what was counted
----------------------

Every trained model keeps the counts behind its PPMI matrix, and a record of
the parameters and corpus that produced them:

.. code-block:: python

   print(model.training_record["word_total"], model.training_record["vocab_size"])
   print(model.training_record["params"]["min_count"])

   counts = model.counts             # exact word and pair counts
   print(counts.word_counts[:3])     # counts of the 3 most frequent words
   print(counts.matrix().nnz)        # distinct observed vocabulary pairs

Save and reload a model
-----------------------

.. code-block:: python

   model.save_model("my_model")        # writes a directory of .npy / .pkl files

   reloaded = SVDAlgebra()
   reloaded.load_model("my_model")

.. warning::

   :meth:`~chronowords.algebra.svd.SVDAlgebra.load_model` unpickles the saved
   vocabulary, which can execute arbitrary code. Only load model directories you
   trust.

Extract topics
--------------

:class:`~chronowords.topics.nmf.TopicModel` runs non-negative matrix
factorisation over the PPMI matrix that ``train`` already computed
(``model.ppmi``):

.. code-block:: python

   from chronowords.topics import TopicModel

   topics = TopicModel(n_topics=2)
   topics.fit(model.ppmi, model.vocabulary)
   topics.print_topics(top_n=5)

Next steps
----------

- :doc:`tutorial` — detect how a word's meaning shifts across time slices.
- :doc:`troubleshooting` — common errors and how to fix them.
- :doc:`api` — the full API reference.
