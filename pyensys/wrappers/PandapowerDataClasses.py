from pandas import DataFrame
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Profile:
    components_indexes_in_power_system: List[int] = field(default_factory=list)
    data: DataFrame = field(default_factory=DataFrame)
    column_names: List[str] = field(default_factory=list)
    variable_name: str = ''
    components_type:str = ''

@dataclass
class TimeSeriesOutputFileSettings:
    number_time_steps: int = 0
    directory: str = ''
    format: str = ''

@dataclass
class OutputVariableSet:
    name_dataset: str = ''
    name_variable: str = ''
    variable_indexes : List[int] = field(default_factory=list)
    evaluation_function = None

@dataclass
class SimulationSettings:
    display_progress_bar: bool = False
    optimisation_software: str = ''
    opf_type: str = ''
    continue_on_divergence: bool = False
    time_steps: List[int] = field(default_factory=list)

@dataclass
class UpdateParameterData:
    component_type: str = ''
    parameter_name: str = ''
    parameter_position: int = 0
    new_value: Any = ''