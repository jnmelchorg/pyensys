from dataclasses import dataclass, field
from typing import List, Any
from pandas import date_range, DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from json import load

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
    column_names: List[Any] = field(default_factory=list)
    row_names: List[Any] = field(default_factory=list)

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
    
class ReadJSON:
    def __init__(self):
        self.problem_settings: ProblemSettings = ProblemSettings()
        self.date_time_optimisation_settings: DateTimeOptimisationSettings = \
            DateTimeOptimisationSettings()
        self.pandapower_mpc_settings: PandaPowerMPCSettings = PandaPowerMPCSettings()
        self.profiles_data: ProfilesData = ProfilesData()
        self.output_settings: OutputSettings = OutputSettings()
        self.pandapower_optimisation_settings: PandaPowerOptimisationSettings = \
            PandaPowerOptimisationSettings()

    def read_parameters_power_system_optimisation(self, json_path: str):
        file = open(json_path)
        self.settings: dict = load(file)
        self.problem_settings = self._load_problem_settings()
        self.date_time_optimisation_settings = self._load_time_series_settings()
        self.pandapower_mpc_settings = self._load_pandapower_mpc_settings()
        self.profiles_data = self._load_profiles_data()
        self.output_settings = self._load_output_settings()
        self.pandapower_optimisation_settings = \
            self._load_pandapower_optimisation_settings()
    
    def _load_problem_settings(self) -> ProblemSettings:
        problem_settings = ProblemSettings()
        problem_settings_dict: dict = self.settings.pop("problem", None)
        if problem_settings_dict is not None:
            problem_settings.system = problem_settings_dict.pop("system")
            problem_settings.problem = problem_settings_dict.pop("name")
            problem_settings.multi_objective = \
                problem_settings_dict.pop("multi_objective")
            problem_settings.stochastic = \
                problem_settings_dict.pop("stochastic")
            problem_settings.intertemporal = \
                problem_settings_dict.pop("intertemporal")
            problem_settings.initialised = True
        return problem_settings
    
    def _load_time_series_settings(self) -> DateTimeOptimisationSettings:
        settings_dict: dict = self.settings.pop("time_related_settings", None)
        date_time_settings = DateTimeOptimisationSettings()
        if settings_dict is not None:
            begin: str = settings_dict.pop("begin")
            end: str = settings_dict.pop("end")
            frequency_name: str = settings_dict.pop("frequency")
            frequency_pandas_alias: str = FREQUENCY_NAME_TO_PANDAS_ALIASES[frequency_name]
            time_block: str = settings_dict.pop("time_block")
            frequency_pandas_alias = time_block + frequency_pandas_alias
            date_time_settings.date_time_settings = date_range(start=begin, \
                end=end, freq=frequency_pandas_alias)
            date_time_settings.initialised = True
        return date_time_settings
    
    def _load_pandapower_mpc_settings(self) -> PandaPowerMPCSettings:
        pandapower_mpc_settings = PandaPowerMPCSettings()
        pandapower_mpc_settings_dict: dict = self.settings.pop(\
            "pandapower_mpc_settings", None)
        if pandapower_mpc_settings_dict is not None:
            pandapower_mpc_settings.mat_file_path = \
                pandapower_mpc_settings_dict.pop("mat_file_path")
            pandapower_mpc_settings.system_frequency = \
                pandapower_mpc_settings_dict.pop("frequency")
            pandapower_mpc_settings.initialised = True
        return pandapower_mpc_settings


    def _load_profiles_data(self) -> ProfilesData:
        profiles_data = ProfilesData()
        profiles_data_dict = self.settings.pop("profiles_data", None)
        if profiles_data_dict is not None:
            for value in profiles_data_dict.values():
                profiles_data.data.append(self._load_profile_data(value))
            profiles_data.initialised = True
        return profiles_data

    def _load_profile_data(self, profile_data_dict: dict) -> ProfileData:
        profile_data = ProfileData()
        dataframe_data = DataframeData(data=profile_data_dict.pop("data"), \
            column_names=profile_data_dict.pop("dataframe_columns_names"), \
            row_names=profile_data_dict.pop("dataframe_rows_date_time"))
        profile_data.data = self._create_dataframe(dataframe_data)
        profile_data.element_type = profile_data_dict.pop("element_type")
        profile_data.variable_name = profile_data_dict.pop("variable_name")
        profile_data.indexes = profile_data_dict.pop("indexes")
        return profile_data
    
    def _create_dataframe(self, dataframe_data: DataframeData) -> DataFrame:
        return DataFrame(data=dataframe_data.data, index=dataframe_data.row_names, \
            columns=dataframe_data.column_names)


    def _load_output_settings(self) -> OutputSettings:
        output_settings = OutputSettings()
        output_settings_dict: dict = self.settings.pop("output_settings", None)
        if output_settings_dict is not None:
            output_settings.directory = output_settings_dict.pop("directory")
            output_settings.format = output_settings_dict.pop("format")
            output_settings.output_variables = \
                self._load_output_variables(output_settings_dict.pop("output_variables"))
            output_settings.initialised = True
        return output_settings

    def _load_output_variables(self, output_variables_dict: dict) -> List[OutputVariable]:
        output_variables = []
        for output_variable in output_variables_dict.values():
            output_variables.append(self._load_output_variable(output_variable))
        return output_variables

    def _load_output_variable(self, output_variable_dict: dict) -> OutputVariable:
        output_variable = OutputVariable()
        output_variable.name_dataset = output_variable_dict.pop("name_dataset")
        output_variable.name_variable = output_variable_dict.pop("name_variable")
        output_variable.variable_indexes = output_variable_dict.pop("variable_indexes")
        return output_variable


    def _load_pandapower_optimisation_settings(self) -> PandaPowerOptimisationSettings:
        pandapower_settings = PandaPowerOptimisationSettings()
        pandapower_optimisation_settings_dict: dict = \
            self.settings.pop("pandapower_optimisation_settings", None)
        if pandapower_optimisation_settings_dict is not None:
            pandapower_settings.continue_on_divergence = \
                pandapower_optimisation_settings_dict.pop("continue_on_divergence")
            pandapower_settings.display_progress_bar = \
                pandapower_optimisation_settings_dict.pop("display_progress_bar")
            pandapower_settings.optimisation_software = \
                pandapower_optimisation_settings_dict.pop("optimisation_software")
            pandapower_settings.initialised = True
        return pandapower_settings
