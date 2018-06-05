#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='pyene',
    version='0.1',
    description='Python Energy and Networks Engine.',
    packages=find_packages(),
    install_requires=[
        'click', 'pandas', 'pyomo'
    ],
    entry_points='''
    [console_scripts]
    pyene=pyene.cli:cli
    ''',
)
