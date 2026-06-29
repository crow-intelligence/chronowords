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
embeddings from any iterable of text lines. Internally it counts words and
skip-grams with a Count-Min Sketch and factorises the PPMI matrix with truncated
SVD.

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

   model = SVDAlgebra(n_components=10, cms_width=10_000, cms_depth=4)
   model.train(iter(corpus))

   print(f"vocabulary size: {len(model.vocabulary)}")

To train on a file, pass the file handle directly — each line is one document::

   with open("corpus.txt", encoding="utf-8") as fh:
       model.train(fh)

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
(``model._ppmi_sparse``):

.. code-block:: python

   from chronowords.topics import TopicModel

   topics = TopicModel(n_topics=2)
   topics.fit(model._ppmi_sparse, model.vocabulary)
   topics.print_topics(top_n=5)

Next steps
----------

- :doc:`tutorial` — detect how a word's meaning shifts across time slices.
- :doc:`troubleshooting` — common errors and how to fix them.
- :doc:`api` — the full API reference.
