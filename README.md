# Python Energy and Networks Engine (pyene) for FutureDAMS

This repository contains the `pyene` model for testing energy systems within
the context of FutureDAMS. It combines an energy balancing engine `pyeneE` and
a steady-state electricity networks simulation engine `pyeneN` for estimating 
energy use while considering losses and network limits. 


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

If anaconda is used then open a console (e.g., Cmder, terminal), check out 
the repository, and install the package (in developer mode).

```bash
git checkout git@gitlab.hydra.org.uk:futuredams/test-case/DAMSEnergy.git
cd DAMSEnergy
python setup.py develop --with-glpk
```

If using another package management and environment management system then you
will need to create a new .bat file (Windows) or a new .sh (linux) with the
following content:

```bash
set LIBRARY_INC=path\to\glpk\include\folder
set LIBRARY_LIB=path\to\glpk\lib\folder
python setup.py build_ext -I"%LIBRARY_INC%" -L"%LIBRARY_LIB%" --inplace --with-glpk develop
```

LIBRARY_INC must contain the path to the folder that containts the file "glpk.h"
and LIBRARY_LIB must contain the path to the folder that containts either
"glpk.lib" or "libglpk.so".

### Running

The `pyene` model provides a basic command line interface (`pyene`) for running
the model. For instructions on the interface, use the consol to access the help
option:

```bash
pyene --help
```

There are three basic use cases for running an energy simulation (`pyeneE`),
network study (`pyeneN`) or combined energy and network study (`pyene`): 

```bash
pyene run-e
pyene run-n
pyene run-en
```

The simulations will run the tests defined in `DAMSEnergy\pyene\json`. Additional
tests can be found in `DAMSEnergy\tests`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnmelchorg/pyensys/blob/master/PowerTech_Tutorial_PyEnSyS_Part1.ipynb)
