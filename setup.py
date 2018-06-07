#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='pyene',
    version='0.1',
    description='Python Energy and Networks Engine - pyene.',
    url='git@gitlab.hydra.org.uk:futuredams/test-case/DAMSEnergy.git',
    author='Dr. Eduardo Alejandro Martínez Ceseña',
    author_email='Eduardo.MartinezCesena@manchester.ac.uk',
    packages=find_packages(),
    package_data={'pyene': ['json/*.json']},
    install_requires=[
        'click', 'pandas', 'pyomo'
    ],
    entry_points='''
    [console_scripts]
    pyene=pyene.cli:cli
    ''',
)
