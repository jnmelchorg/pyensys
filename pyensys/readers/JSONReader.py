from typing import List
from pandas import date_range, DataFrame
from json import load
from pyensys.readers.ReaderDataClasses import *
    
class ReadJSON:
    def __init__(self):
        self.parameters = Parameters()

    def read_json_data(self, json_path: str):
        file = open(json_path)
        self.settings: dict = load(file)
        p = self.parameters
        p.problem_settings = self._load_problem_settings()
        p.date_time_optimisation_settings = self._load_time_series_settings()
        p.pandapower_mpc_settings = self._load_pandapower_mpc_settings()
        p.profiles_data = self._load_profiles_data()
        p.output_settings = self._load_output_settings()
        p.pandapower_optimisation_settings = \
            self._load_pandapower_optimisation_settings()
        p.initialised = True
    
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
            problem_settings.opf_optimizer = \
                problem_settings_dict.pop("opf_optimizer")
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
