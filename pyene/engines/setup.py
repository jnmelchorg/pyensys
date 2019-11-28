from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.extension import Extension

# needed to compile
setup_requires = [
    "cython", "setuptools_scm"
]
# needed to run
install_requires = []

# only needed for testing
test_requires = [
    "pytest"
]

setup(
    ext_modules=cythonize([Extension('_glpk', ['_glpk.pyx'], libraries=['glpk'],)], language_level = "3")
)
