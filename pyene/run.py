import sys

from pyomo.core import Objective, minimize
from pyomo.environ import SolverFactory

from engines.pyene_Models import Energymodel
from engines.pyene import pyeneClass, pyeneConfig
from engines.pyeneE import EnergyClass

GLPK_DIR = "/usr/bin"
GLPK_EXE = "glpsol"

sys.path.append(GLPK_DIR)


def model():
    conf   = pyeneConfig()
    pyeCls = pyeneClass(conf)

    ec = EnergyClass()
    ec.initialise()

    lpModel = pyeCls.SingleLP(ec)

    em = Energymodel(ec)
    em.optimisation()

    lpModel.OF = Objective(rule=ec.OF_rule, sense=minimize)
    opt        = SolverFactory("glpk", executable=GLPK_EXE)

    return opt.solve(lpModel)


if __name__ == "__main__":
    results = model()
    print(results)
