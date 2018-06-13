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


### Installing

Using a console (e.g., Cmder), check out the repository, and install the 
package (in developer mode).

```bash
git checkout git@gitlab.hydra.org.uk:futuredams/test-case/DAMSEnergy.git
cd DAMSEnergy
pip install -e . --user
```


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