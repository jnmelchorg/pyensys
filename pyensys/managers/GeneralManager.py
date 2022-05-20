from json import dump

from pyensys.readers.ReaderManager import read_parameters
# from json import load, dump
# from os.path import dirname, join, abspath
from pyensys.Optimisers.RecursiveFunction import RecursiveFunction, InterIterationInformation
from pyensys.Optimisers.NonAnticipativeRecursiveFunction import NonAnticipativeRecursiveFunction


def main_access_function(file_path: str) -> list:
    parameters = read_parameters(file_path)
    if not parameters.problem_settings.non_anticipative:
        RF = RecursiveFunction()
        RF.initialise(parameters)
    else:
        RF = NonAnticipativeRecursiveFunction()
        RF.initialise(parameters)
        info = InterIterationInformation()
        info.incumbent_graph_paths.create_list()
        info.incumbent_investment_costs.create_list()
        info.incumbent_operation_costs.create_list()
        info.incumbent_interventions.create_list()
        info.candidate_interventions_remaining_construction_time.create_list()
        info.candidate_interventions.create_list()
        info.candidate_solution_path.create_list()
        info.candidate_operation_cost.create_list()
        info.new_interventions.create_list()
        info.new_interventions_remaining_construction_time.create_list()
        info.complete_tree.graph_paths.create_list()
        info.complete_tree.investment_costs.create_list()
        info.complete_tree.operation_costs.create_list()
        info.complete_tree.interventions.create_list()
        RF.solve(info)
        return RF.get_solution(info)

    # path = join(dirname(__file__), "..", "tests", "json", "Output_Data_Investment_proposal.json")
    # with open(path) as output_data:
    #     output = load(output_data)
    # with open(file_path) as input_data:
    #     parameters = load(input_data)
    # with open(parameters["output_directory"] + "\\solution.json", "w") as output_dir:
    #     dump(output, output_dir, ensure_ascii=False, indent=2)


def save_in_json(data: list, file_path: str):
    for sol in data:
        sol["data"] = sol["data"].to_dict()
    with open(file_path, 'w') as fp:
        dump(data, fp, indent=4)
