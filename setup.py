#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Sparse Blind Deconvolution of density of states maps',
    version='0.0.0',
    description='Takes a density of states map that contains quasi-particle interference (QPI)'
                ' patterns and returns the defect map and the single-defect QPI',
    author='Dror Harush',
    author_email='drorharush@gmail.com',
    url='https://github.com/Drorharush/SparseBlindDeconv',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

