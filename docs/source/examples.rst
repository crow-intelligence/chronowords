Examples
========

Loading and Training a Model
--------------------------

.. code-block:: python

    from chronowords.algebra import SVDAlgebra

    # Initialize model
    model = SVDAlgebra(n_components=300)

    # Train on your corpus
    with open('your_corpus.txt', 'r') as f:
        model.train(f)

Finding Similar Words
-------------------

.. code-block:: python

    # Find similar words
    similar = model.most_similar('computer', n=10)
    for word in similar:
        print(f"{word.word}: {word.similarity:.3f}")
