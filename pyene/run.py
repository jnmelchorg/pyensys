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


def test_glpk():
    from engines._glpk import GLPKSolver

    rows_id = "prows"

    solver = GLPKSolver()
    solver.set_dir("max")
    solver.set_prob_name("PS GLPK test problem")
    solver.add_rows(rows_id, 2)
    solver.set_row_bnds(rows_id, 0, "upper", 0.0, 1.0);
    solver.set_row_bnds(rows_id, 1, "upper", 0.0, 2.0);


if __name__ == "__main__":
    results = model()
    print(results)

    test_glpk()
