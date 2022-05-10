from pandas import read_excel
from json import load
from os.path import splitext
from pyensys.readers.ReaderDataClasses import *
from typing import Any


def _read_active_columns_names(profile_data_dict: dict, data: DataFrame) -> List[str]:
    if profile_data_dict.get("active_columns_names", None) is not None:
        return profile_data_dict.pop("active_columns_names")
    elif profile_data_dict.get("all_active_columns_names", False):
        return data.columns.tolist()


def _create_dataframe(dataframe_data: DataframeData) -> DataFrame:
    return DataFrame(data=dataframe_data.data, columns=dataframe_data.column_names)


def _read_file(path: str, profile_data_dict: dict, header: Any = 0) -> DataFrame:
    if profile_data_dict.get("excel_sheet_name", None) is None:
        raise Exception("Excel sheet name does not exist.")
    _, file_extension = splitext(path)
    if file_extension == ".xlsx":
        name = profile_data_dict.pop("excel_sheet_name")
        data = read_excel(io=path, sheet_name=name, header=header)
        if isinstance(data, dict):
            return data.pop(name)
        elif isinstance(data, DataFrame):
            return data


def _read_dataframe_data(profile_data_dict: dict) -> DataFrame:
    if profile_data_dict.get("data", None) is not None and \
            profile_data_dict.get("dataframe_columns_names", None) is not None:
        dataframe_data = DataframeData(data=profile_data_dict.pop("data"),
                                       column_names=profile_data_dict.pop("dataframe_columns_names"))
        return _create_dataframe(dataframe_data)
    elif profile_data_dict.get("data_path", None) is not None and \
            profile_data_dict.get("excel_sheet_name", None) is not None:
        return _read_file(profile_data_dict.pop("data_path"), profile_data_dict)
    else:
        return DataFrame()


def _load_pandapower_profile_data(profile_data_dict: dict) -> List[PandaPowerProfileData]:
    if profile_data_dict.get("format_data") == "original":
        return _read_profile_data_with_original_format(profile_data_dict)
    elif profile_data_dict.get("format_data") == "attest":
        if profile_data_dict.get("data_path", None) is not None:
            return _read_profile_data_from_file_with_attest_format(profile_data_dict)


def _read_profile_data_from_file_with_attest_format(profile_data_dict):
    data_per_type_element, profiles, type_element = _divide_data_per_type(profile_data_dict)
    for number, _ in enumerate(type_element):
        profile_data = PandaPowerProfileData()
        _store_indexes_and_element_type(data_per_type_element, number, profile_data, profile_data_dict)
        _store_variables_names(number, profile_data, type_element)
        column_names = _store_dataframe(data_per_type_element, number, profile_data, type_element)
        _store_active_columns(column_names, profile_data, profile_data_dict)
        profiles.append(profile_data)
    return profiles


def _store_active_columns(column_names, profile_data, profile_data_dict):
    if profile_data_dict.get("active_columns_names", None) is not None:
        profile_data.active_columns_names = profile_data_dict.pop("active_columns_names")
    else:
        profile_data.active_columns_names = column_names


def _store_indexes_and_element_type(data_per_type_element, number, profile_data, profile_data_dict):
    profile_data.element_type = profile_data_dict.get("element_type")
    profile_data.indexes = list(data_per_type_element[number].pop(0))
    profile_data.indexes = [i - 1 for i in profile_data.indexes]


def _store_dataframe(data_per_type_element, number, profile_data, type_element):
    column_names = []
    if type_element[number] == "P":
        for index in profile_data.indexes:
            column_names.append(f"pd_{index}")
    elif type_element[number] == "Q":
        for index in profile_data.indexes:
            column_names.append(f"qd_{index}")
    data_per_type_element[number] = data_per_type_element[number].T
    data_per_type_element[number].columns = column_names
    profile_data.data = data_per_type_element[number]
    return column_names


def _store_variables_names(number, profile_data, type_element):
    if type_element[number] == "P":
        profile_data.variable_name = "p_mw"
    elif type_element[number] == "Q":
        profile_data.variable_name = "q_mvar"


def _divide_data_per_type(profile_data_dict):
    raw_data = _read_file(profile_data_dict.pop("data_path"), profile_data_dict, header=None)
    type_element = []
    rows_per_type_element = []
    for number, cell in enumerate(raw_data.iterrows()):
        if cell[1][1] not in type_element:
            type_element.append(cell[1][1])
            rows_per_type_element.append([number])
        else:
            rows_per_type_element[type_element.index(cell[1][1])].append(number)
    raw_data.pop(1)
    data_per_type_element = []
    for number, _ in enumerate(type_element):
        data_per_type_element.append(raw_data.iloc[rows_per_type_element[number], :])
    profiles = []
    return data_per_type_element, profiles, type_element


def _read_profile_data_with_original_format(profile_data_dict: dict) -> List[PandaPowerProfileData]:
    profile_data = PandaPowerProfileData()
    profile_data.data = _read_dataframe_data(profile_data_dict)
    profile_data.element_type = profile_data_dict.pop("element_type")
    profile_data.variable_name = profile_data_dict.pop("variable_name")
    profile_data.indexes = profile_data_dict.pop("indexes", [])
    profile_data.all_indexes = profile_data_dict.pop("all_indexes", False)
    profile_data.active_columns_names = _read_active_columns_names(profile_data_dict, profile_data.data)
    return [profile_data]


def _load_optimisation_profile_data(profile_data_dict: dict) -> OptimisationProfileData:
    profile_data = OptimisationProfileData()
    profile_data.data = _read_dataframe_data(profile_data_dict)
    profile_data.element_type = profile_data_dict.pop("element_type")
    profile_data.variable_name = profile_data_dict.pop("variable_name")
    return profile_data


def _load_output_variable(output_variable_dict: dict) -> OutputVariable:
    output_variable = OutputVariable()
    output_variable.name_dataset = output_variable_dict.pop("name_dataset")
    output_variable.name_variable = output_variable_dict.pop("name_variable")
    output_variable.variable_indexes = output_variable_dict.pop("variable_indexes")
    return output_variable


def _load_output_variables(output_variables_dict: dict) -> List[OutputVariable]:
    output_variables = []
    for output_variable in output_variables_dict.values():
        output_variables.append(_load_output_variable(output_variable))
    return output_variables


class ReadJSON:
    def __init__(self):
        self.settings = None
        self.parameters = Parameters()

    def read_json_data(self, json_path: str):
        file = open(json_path)
        self.settings: dict = load(file)
        p = self.parameters
        p.problem_settings = self._load_problem_settings()
        p.opf_time_settings = self._load_opf_time_settings()
        p.pandapower_mpc_settings = self._load_pandapower_mpc_settings()
        p.pandapower_profiles_data = self._load_pandapower_profiles_data()
        p.output_settings = self._load_output_settings()
        p.pandapower_optimisation_settings = \
            self._load_pandapower_optimisation_settings()
        p.optimisation_profiles_data = \
            self._load_optimisation_profiles_data()
        p.initialised = True

    def _load_problem_settings(self) -> ProblemSettings:
        problem_settings = ProblemSettings()
        problem_settings_dict: dict = self.settings.pop("problem", None)
        if problem_settings_dict is not None:
            problem_settings.system = problem_settings_dict.pop("system", '')
            problem_settings.problem = problem_settings_dict.pop("name", '')
            problem_settings.multi_objective = \
                problem_settings_dict.pop("multi_objective", False)
            problem_settings.stochastic = \
                problem_settings_dict.pop("stochastic", False)
            problem_settings.intertemporal = \
                problem_settings_dict.pop("intertemporal", False)
            problem_settings.opf_optimizer = \
                problem_settings_dict.pop("opf_optimizer", '')
            problem_settings.problem_optimizer = \
                problem_settings_dict.pop("problem_optimizer", '')
            problem_settings.opf_type = \
                problem_settings_dict.pop("opf_type", '')
            problem_settings.return_rate_in_percentage = \
                problem_settings_dict.pop("return_rate_in_percentage", 0.0)
            problem_settings.initialised = True
        return problem_settings

    def _load_opf_time_settings(self) -> DateTimeOptimisationSettings:
        settings_dict: dict = self.settings.pop("opf_time_settings", None)
        date_time_settings = DateTimeOptimisationSettings()
        if settings_dict is not None:
            begin: str = settings_dict.pop("begin")
            end: str = settings_dict.pop("end")
            frequency_name: str = settings_dict.pop("frequency")
            frequency_pandas_alias: str = FREQUENCY_NAME_TO_PANDAS_ALIASES[frequency_name]
            time_block: str = settings_dict.pop("time_block")
            frequency_pandas_alias = time_block + frequency_pandas_alias
            date_time_settings.date_time_settings = date_range(start=begin, end=end, freq=frequency_pandas_alias)
            date_time_settings.initialised = True
        return date_time_settings

    def _load_pandapower_mpc_settings(self) -> PandaPowerMPCSettings:
        pandapower_mpc_settings = PandaPowerMPCSettings()
        pandapower_mpc_settings_dict: dict = self.settings.pop("pandapower_mpc_settings", None)
        if pandapower_mpc_settings_dict is not None:
            pandapower_mpc_settings.mat_file_path = \
                pandapower_mpc_settings_dict.pop("mat_file_path")
            pandapower_mpc_settings.system_frequency = \
                pandapower_mpc_settings_dict.pop("frequency")
            pandapower_mpc_settings.initialised = True
        return pandapower_mpc_settings

    def _load_pandapower_profiles_data(self) -> PandaPowerProfilesData:
        profiles_data = PandaPowerProfilesData()
        profiles_data_dict: dict = self.settings.pop("pandapower_profiles_data", None)
        if profiles_data_dict is not None:
            for value in profiles_data_dict.values():
                profiles_data.data.extend(_load_pandapower_profile_data(value))
            profiles_data.initialised = True
        return profiles_data

    def _load_optimisation_profiles_data(self) -> OptimisationProfilesData:
        profiles_data = OptimisationProfilesData()
        profiles_data_dict = self.settings.pop("optimisation_profiles_data", None)
        if profiles_data_dict is not None:
            for value in profiles_data_dict.values():
                profiles_data.data.append(_load_optimisation_profile_data(value))
            profiles_data.initialised = True
        return profiles_data

    def _load_output_settings(self) -> OutputSettings:
        output_settings = OutputSettings()
        output_settings_dict: dict = self.settings.pop("output_settings", None)
        if output_settings_dict is not None:
            output_settings.directory = output_settings_dict.pop("directory")
            output_settings.format = output_settings_dict.pop("format")
            output_settings.output_variables = \
                _load_output_variables(output_settings_dict.pop("output_variables"))
            output_settings.initialised = True
        return output_settings

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

    def _load_optimisation_binary_variables(self):
        optimisation_binary_variables_list: List[dict] = \
            self.settings.pop("optimisation_binary_variables", [])
        for variable in optimisation_binary_variables_list:
            self.parameters.optimisation_binary_variables.append(
                OptimisationBinaryVariables(element_type=variable.get("element_type"),
                                            variable_name=variable.get("variable_name"),
                                            elements_ids=variable.get("elements_ids"),
                                            elements_positions=variable.get("elements_positions", []),
                                            costs=variable.get("costs", []),
                                            installation_time=variable.get("installation_time", []))
            )
