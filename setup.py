from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# Add build_ext command
from setuptools.command.build_ext import build_ext

extensions = [
    Extension(
        "chronowords.utils.count_skipgrams",
        ["src/chronowords/utils/count_skipgrams.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    )
]

if __name__ == "__main__":
    # Add build command if not present
    if len(sys.argv) == 1:
        sys.argv.append('build_ext')

    setup(
        name="chronowords",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                'language_level': 3,
                'boundscheck': False,
                'wraparound': False,
                'nonecheck': False,
                'cdivision': True
            }
        ),
        cmdclass={'build_ext': build_ext},
        zip_safe=False,
    )