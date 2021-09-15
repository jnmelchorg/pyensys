from os import read
from pyensys.readers.ReaderManager import read_parameters
class pyensys:

    def main_access_function(self, file_path: str):
        parameters = read_parameters(file_path)      
        
        
