from pyensys.readers.ReaderDataClasses import Parameters
from pyensys.Optimisers.RecursiveFunction import RecursiveFunction


def identify_and_solve(parameters: Parameters):
    RF = RecursiveFunction()
    RF.initialise(parameters)
        # RF.solve()
