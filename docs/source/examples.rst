Examples
========

The worked example
------------------

The repository ships a complete, runnable notebook at
``examples/presidential_speeches.ipynb``. It applies the full chronowords
pipeline to U.S. presidential speeches grouped by quarter-century:

#. load and group the speeches into time slices,
#. build PPMI embeddings per slice with
   :class:`~chronowords.algebra.svd.SVDAlgebra`,
#. align the slices with
   :class:`~chronowords.alignment.procrustes.ProcrustesAligner`,
#. detect and visualise (with Altair) how individual words shifted, and
#. model topics per slice with :class:`~chronowords.topics.nmf.TopicModel`.

Clone the repository and open it with Jupyter to run it end-to-end.

Short snippets
--------------

Train embeddings and find similar words:

.. code-block:: python

   from chronowords.algebra import SVDAlgebra

   model = SVDAlgebra(n_components=300)
   with open("corpus.txt", encoding="utf-8") as fh:
       model.train(fh)

   for hit in model.most_similar("computer", n=10):
       print(f"{hit.word}: {hit.similarity:.3f}")

For step-by-step walkthroughs see the :doc:`quickstart` (single corpus) and the
:doc:`tutorial` (semantic shift across time slices).
