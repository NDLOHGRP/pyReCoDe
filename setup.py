from setuptools import setup, Extension, find_packages
import numpy as np

module1 = Extension('c_recode',
                    define_macros=[('MAJOR_VERSION', '0'),
                                   ('MINOR_VERSION', '1')],
                    include_dirs=['pyrecode/c_extensions', np.get_include()],
                    sources=['pyrecode/pyrecode.cpp'],
                    extra_compile_args=["-O3"])

setup(name='pyrecode',
      version='0.1',
      description='Reduced Compressed Description for Direct Electron Microscopy Data',
      author='Abhik Datta',
      author_email='findabhik@gmail.com',
      url='https://github.com/NDLOHGRP/ReCoDe',
      long_description='Readers, writers and Converters for ReCoDe files',
      ext_modules=[module1],
      packages=find_packages(),
      python_requires='>3.6'
      )
