from dataclasses import dataclass, field
from pandas.core.indexes.datetimes import DatetimeIndex
from typing import List, Any
from pandas import date_range, DataFrame

FREQUENCY_NAME_TO_PANDAS_ALIASES: dict = \
    {"hourly": "H"}

@dataclass
class ProblemSettings:
    system: str = ''
    problem: str = ''
    multi_objective: bool = False
    stochastic: bool = False
    intertemporal: bool = False
    initialised: bool = False
    opf_optimizer: str = ''
    problem_optimizer: str = ''

@dataclass
class DateTimeOptimisationSettings:
    date_time_settings: DatetimeIndex = \
        field(default_factory=lambda: date_range(start="2021-1-1", periods=1))
    initialised: bool = False

@dataclass
class PandaPowerMPCSettings:
    mat_file_path: str = ''
    system_frequency: float = 0.0
    initialised: bool = False

@dataclass
class ProfileData:
    element_type: str = ''
    variable_name: str = ''
    indexes: List[str] = field(default_factory=list)
    data: DataFrame = field(default_factory=DataFrame)

@dataclass
class ProfilesData:
    data: List[ProfileData] = field(default_factory=list)
    initialised: bool = False

@dataclass
class DataframeData:
    data: List[List[Any]] = field(default_factory=list)
    column_names: List[str] = field(default_factory=list)
    row_names: List[str] = field(default_factory=list)

@dataclass
class OutputVariable:
    name_dataset: str = ''
    name_variable: str = ''
    variable_indexes: List[int] = field(default_factory=list)

@dataclass
class OutputSettings:
    directory: str = ''
    format: str = ''
    output_variables: List[OutputVariable] = field(default_factory=list)
    initialised: bool = False

@dataclass
class PandaPowerOptimisationSettings:
    display_progress_bar: bool = False
    continue_on_divergence: bool = False
    optimisation_software: str = ''
    initialised: bool = False

@dataclass
class Parameters:
    problem_settings: ProblemSettings = \
        field(default_factory=lambda: ProblemSettings())
    date_time_optimisation_settings: DateTimeOptimisationSettings = \
        field(default_factory=lambda: DateTimeOptimisationSettings())
    pandapower_mpc_settings: PandaPowerMPCSettings = \
        field(default_factory=lambda: PandaPowerMPCSettings())
    profiles_data: ProfilesData = field(default_factory=lambda: ProfilesData())
    output_settings: OutputSettings = field(default_factory=lambda: OutputSettings())
    pandapower_optimisation_settings: PandaPowerOptimisationSettings = \
        field(default_factory=lambda: PandaPowerOptimisationSettings())
    initialised: bool = False
