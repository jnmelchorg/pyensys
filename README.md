# Python Energy Systems Simulator (PyEnSyS) for FutureDAMS

This repository contains the `PyEnSyS` model which was originally developed to 
model energy and water-energy systems within the context of the FutureDAMS project [1],
and, afterwards, extended to include investment planning capabilities in the context of
the ATTEST project [2]. PyEnSyS provides (i) a n integrated simulation framework which
combines an energy balancing engine and a steady-state electricity networks simulation 
engine for estimating energy use while considering losses and network limits, and 
(ii) a dedicated optimisation engine based on graph-theory and recursion-theory for
the optimisation of electricity distribution networks. 


## Getting started

### Dependencies

Running this code will require,

- pyomo
- pypsa (optional)
- glpk

### Recommended package management and environment management systems

For most users it will be easier to install the binary packages made available 
for the Anaconda Python distribution. The use of Anaconda is recommended for
the automatic installation of pyene and all its cython related dependencies.

### Installing

Use the command below to install pyensys in your device.
```bash
pip install pyensys
```

### Running

Examples on how to run pyenesys can be found clicking in the icon for Google Colaboratory below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnmelchorg/pyensys/blob/master/docs/Tutorial%20PowerTech%202021/PowerTech%20Tutorial%20PyEnSyS%20Part1.ipynb)

[1] https://www.futuredams.org/
[2] https://attest-project.eu/
