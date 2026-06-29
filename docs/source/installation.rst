Installation
============

Install the released package from PyPI:

.. code-block:: bash

   pip install chronowords

chronowords ships a compiled Cython extension, so a C/C++ compiler is required if
no wheel is available for your platform. Python 3.10–3.12 is supported.

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/crow-intelligence/chronowords.git
   cd chronowords
   pip install -e .

Development install with uv
---------------------------

The project is developed with `uv <https://docs.astral.sh/uv/>`_. To set up a
full development environment (project, dev, and docs dependencies):

.. code-block:: bash

   uv sync --all-groups
   uv run pytest          # run the test suite

After editing the Cython source (``count_skipgrams.pyx``), rebuild the extension:

.. code-block:: bash

   uv sync --reinstall-package chronowords
