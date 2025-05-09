# coding: utf-8

from setuptools import Extension, setup

setup(ext_modules=[Extension(name="kelvin_model.kelvin_lib",
                             sources=["src/kelvin_model/src/kelvin_lib.cpp"],
                             include_dirs=["/usr/include/eigen3",
                                           "src/kelvin_model/include"],
                             extra_compile_args=["-shared",
                                                 "-fPIC",
                                                 "-O3",
                                                 "-march=native",
                                                 "-ffast-math"],
                             language="c++",
                             optional=False,
                             py_limited_api=False)])
