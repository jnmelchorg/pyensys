from pyensys.wrappers.PandaPowerManager import PandaPowerManager
from pyensys.readers.ReaderDataClasses import Parameters

class RecursiveFunction:
        
    def operational_check(self):
        if self.opt_optimizer == "pandapower":
            self.pp_opf.run_timestep_opf_pandapower()
        
    def initialise(self, parameters: Parameters):
        if parameters.problem_settings.opf_optimizer == "pandapower":
            self.pp_opf = PandaPowerManager()
            self.opt_optimizer = "pandapower"
            self.pp_opf.initialise_pandapower_network(parameters)
    
    def solve(self):
        self.operational_check()




        

