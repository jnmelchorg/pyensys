from os.path import splitext
from pyensys.readers.JSONReader import ReadJSON
from pyensys.readers.ReaderDataClasses import Parameters

def read_parameters(file_path: str) -> Parameters:
    parameters = Parameters()
    _, file_extension = splitext(file_path)
    if file_extension == ".json":
        reader = ReadJSON()
        reader.read_json_data(file_path)
        parameters = reader.parameters
    return parameters