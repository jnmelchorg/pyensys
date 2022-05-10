from pyensys.readers.ReaderDataClasses import Parameters
from pyensys.Optimisers.RecursiveFunction import RecursiveFunction


def identify_and_solve(parameters: Parameters):
    if parameters.problem_settings.problem_optimizer == "recursive_function":
        RF = RecursiveFunction()
        RF.initialise(parameters)
        RF.solve()
