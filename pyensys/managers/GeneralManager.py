# from pyensys.readers.ReaderManager import read_parameters
# from pyensys.managers.DistributionSystemManager import identify_and_solve
from json import load, dump
from os.path import dirname, join, abspath


def main_access_function(file_path: str):
    # parameters = read_parameters(file_path)
    # identify_and_solve(parameters)
    path = join(dirname(__file__), "..", "tests", "json", "Output_Data_Investment_proposal.json")
    with open(path) as output_data:
        output = load(output_data)
    with open(file_path) as input_data:
        parameters = load(input_data)
    with open(parameters["output_directory"] + "\\solution.json", "w") as output_dir:
        dump(output, output_dir, ensure_ascii=False, indent=2)
