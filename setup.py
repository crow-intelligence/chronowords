import sys

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import find_namespace_packages
from setuptools import setup


try:
    import numpy as np
except ImportError:
    # If numpy is not available during the build, we need to install it first
    from setuptools import dist

    dist.Distribution().fetch_build_eggs(["numpy"])
    import numpy as np

# Define Cython extension
extensions = [
    Extension(
        "chronowords.utils.count_skipgrams",
        ["src/chronowords/utils/count_skipgrams.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"],  # Ensure C++11 support
    )
]

# Always run setup with build_ext
if len(sys.argv) == 1:
    sys.argv.append("build_ext")
    sys.argv.append("--inplace")

setup(
    name="chronowords",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    ),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "nltk>=3.6.0",
        "mmh3>=3.0.0",  # For MurmurHash3
    ],
    python_requires=">=3.8",
    zip_safe=False,  # Required for Cython
)
