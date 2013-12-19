from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Time Frequency Features for phone classification",
    ext_modules = cythonize(['src/_tapers.pyx','src/_reassignment.pyx','src/hog_gray_scale.pyx','src/_bernoullimm.pyx']), # accepts a glob pattern
)
