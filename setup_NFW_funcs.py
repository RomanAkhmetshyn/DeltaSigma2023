# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:43:48 2023

@author: Admin
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('NFW_funcs.pyx',annotate=True),include_dirs=[numpy.get_include()])