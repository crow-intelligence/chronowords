# setup.py
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "chronowords.utils.count_skipgrams",
        ["src/chronowords/utils/count_skipgrams.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
)
