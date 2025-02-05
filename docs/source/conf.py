import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'chronowords'
copyright = '2025, Orsolya Putz, Zoltan Varju'
author = 'Orsolya Putz, Zoltan Varju'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.githubpages',  # Generate .nojekyll file
    'sphinx_rtd_theme',  # Read the Docs theme
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx_autodoc_typehints',  # Support for type hints
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

# AutoDoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__',
}