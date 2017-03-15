from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import os

extension = Extension(
    name = "cinference",
    sources = ["cinference.pyx"],
    language="c++"
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extension),
)

os.rename("cinference.cpython-35m-x86_64-linux-gnu.so", "cinference.so")
