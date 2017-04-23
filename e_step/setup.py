from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import os

extension = Extension(
    name = "cinference",
    sources = ["cinference.pyx"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extension),
)

try:
    os.rename("cinference.cpython-35m-x86_64-linux-gnu.so", "cinference.so")
except Exception:
    try:
        os.rename("cinference.cpython-34m.so", "cinference.so")
    except Exception:
        os.rename("cinference.cpython-35m-darwin.so", "cinference.so")

