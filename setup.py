#!/usr/bin/env python
from setuptools import find_packages, setup
packages = find_packages()

with open("README.md", "r") as fh:
    long_description = fh.read()


VERSION = "0.1.0"
setup(name='PyTomoATT',
      version=VERSION,
      author='Mijian Xu, Masaru Nagaso',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author_email='mijian.xu@ntu.edu.sg',
      license='GPLv3',
      package_dir={'pytomoatt': 'pytomoatt'},
      packages=find_packages(),
      install_requires=[
                'numpy>=1.19.0',
                'scipy>=1.9.1',
                'pandas>=1.4.0',
                'h5py',
                'pyyaml',
                'xarray'],
      zip_safe=False,
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   'Programming Language :: Python :: 3.11']
      )
