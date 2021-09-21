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
    opf_type: str = ''

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
class PandaPowerProfileData:
    element_type: str = ''
    variable_name: str = ''
    indexes: List[int] = field(default_factory=list)
    all_indexes: bool = False
    data: DataFrame = field(default_factory=DataFrame)
    active_columns_names: List[str] = field(default_factory=list)

@dataclass
class PandaPowerProfilesData:
    data: List[PandaPowerProfileData] = field(default_factory=list)
    initialised: bool = False

@dataclass
class DataframeData:
    data: List[List[Any]] = field(default_factory=list)
    column_names: List[str] = field(default_factory=list)

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
    opf_time_settings: DateTimeOptimisationSettings = \
        field(default_factory=lambda: DateTimeOptimisationSettings())
    pandapower_mpc_settings: PandaPowerMPCSettings = \
        field(default_factory=lambda: PandaPowerMPCSettings())
    pandapower_profiles_data: PandaPowerProfilesData = \
        field(default_factory=lambda: PandaPowerProfilesData())
    output_settings: OutputSettings = field(default_factory=lambda: OutputSettings())
    pandapower_optimisation_settings: PandaPowerOptimisationSettings = \
        field(default_factory=lambda: PandaPowerOptimisationSettings())
    initialised: bool = False
