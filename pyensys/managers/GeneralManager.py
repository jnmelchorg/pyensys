from pyensys.readers.JSONReader import ReadJSON
class pyensys:

    def main_access_function(self, file_path: str, extension: str):
        if extension == ".json":
            data_json = ReadJSON()
            data_json.read_parameters_power_system_optimisation(file_path)

        
