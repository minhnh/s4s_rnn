#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='s4s_rnn',
    packages=find_packages(where='./src'),
    package_dir={'': 'src'},
    install_requires=[
        'keras',
        'sklearn',
    ]
)
