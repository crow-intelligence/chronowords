Installation
============

Install the released package from PyPI:

.. code-block:: bash

   pip install chronowords

chronowords ships a compiled Rust extension (``chronowords._core``). Installing
from a wheel needs no compiler; building from source (an sdist, a git checkout, or
a platform without a wheel) needs a Rust toolchain — install it with
`rustup <https://rustup.rs>`_. Python 3.10–3.12 is supported.

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

After editing the Rust source (``rust/src/*.rs``), rebuild the extension:

.. code-block:: bash

   uv sync --reinstall-package chronowords
