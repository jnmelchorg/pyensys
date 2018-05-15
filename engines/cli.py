# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:41:26 2018

@author: mchihem2
"""

# cli.py
import click
from pyene import pyeneClass as pc


#@click.command('run-pyene')
def run_test_pyene():
    # Create pyene object
    EN = pc()
    EN._runTests(EN)

run_test_pyene()
