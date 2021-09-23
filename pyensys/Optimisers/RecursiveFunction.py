from pandas.core.frame import DataFrame
from pyensys.wrappers.PandaPowerManager import PandaPowerManager
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfilesData
from typing import List
from dataclasses import dataclass, field

@dataclass
class DataMultiplier:
    data_multiplier: float = 0.0
    element_type: str = ''
    variable_name: str = ''

@dataclass
class DataMultipliers:
    multipliers: List[DataMultiplier] = field(default_factory=list)
    initialised: bool = False

class RecursiveFunction:

    def __init__(self):
        self.graph_nodes_multipliers: List[DataMultipliers] = []
        self.original_pp_profiles_data = PandaPowerProfilesData()
        
    def operational_check(self):
        if self.opt_optimizer == "pandapower" and self.intertemporal:
            self.pp_opf.run_timestep_opf_pandapower()
    
    def update_pandapower_controllers(self, current_node: int):
        new_profiles = PandaPowerProfilesData()
        new_profiles.initialised = True
        data_multipliers = self.get_data_multipliers_current_node(current_node)
        for multiplier in data_multipliers.multipliers:
            profile_position_to_update = self.get_profile_position_to_update(multiplier)
            profile = self.original_pp_profiles_data.data[profile_position_to_update]
            profile.data = profile.data * multiplier.data_multiplier
            new_profiles.data.append(profile)
        self.pp_opf.update_network_controllers(new_profiles)
    
    def get_profile_position_to_update(self, multiplier: DataMultiplier) -> int:
        profile_position_to_update = -1
        for position, pp_profile in enumerate(self.original_pp_profiles_data.data):
            if multiplier.element_type == pp_profile.element_type and \
                multiplier.variable_name == pp_profile.variable_name:
                profile_position_to_update = position
        return profile_position_to_update

    def get_data_multipliers_current_node(self, current_node: int) -> DataMultipliers:
        return self.graph_nodes_multipliers[current_node]
        
    def initialise(self, parameters: Parameters):
        if parameters.problem_settings.opf_optimizer == "pandapower":
            self.pp_opf = PandaPowerManager()
            self.opt_optimizer = "pandapower"
            self.intertemporal = parameters.problem_settings.intertemporal
            self.pp_opf.initialise_pandapower_network(parameters)
            self.original_pp_profiles_data = parameters.pandapower_profiles_data
    
    def solve(self):
        self.operational_check()




        

