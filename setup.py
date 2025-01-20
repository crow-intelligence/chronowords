import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext

# Define the extension modules
extensions = [
    Extension(
        "chronowords.utils.count_skipgrams",
        sources=["src/chronowords/utils/count_skipgrams.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]

if __name__ == "__main__":
    # Add build command if not present
    if len(sys.argv) == 1:
        sys.argv.append("build_ext")

    setup(
        name="chronowords",
        packages=find_namespace_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "cdivision": True,
            },
        ),
        zip_safe=False,
    )
