from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup


try:
    import numpy as np
except ImportError:
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
        extra_compile_args=["-std=c++11"],
    )
]

setup(
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
    zip_safe=False,
)
