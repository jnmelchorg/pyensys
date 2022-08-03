from math import sqrt

from pandas import read_excel, DataFrame, date_range, concat
from json import load
from os.path import splitext
from typing import Any
from typing import List

from pyensys.readers.ReaderDataClasses import DataframeData, PandaPowerProfileData, OptimisationProfileData, \
    OutputVariable, Parameters, ProblemSettings, DateTimeOptimisationSettings, PandaPowerMPCSettings, \
    PandaPowerProfilesData, OptimisationProfilesData, OutputSettings, PandaPowerOptimisationSettings, \
    OptimisationBinaryVariables, FREQUENCY_NAME_TO_PANDAS_ALIASES


def _build_mat_file(path):
    with open(path, 'r') as f:
        contents = f.readlines()

    # We know the file will contain the following elements:
    Elements = ["version", "baseMVA", "bus", "gen", "branch", "gencost"]
    Components = [1, 1, 13, 21, 13, 6]
    NoElements = len(Elements)
    NoContents = len(contents)
    Vars = {}
    Vars['mpc'] = {}
    xline = 0
    while xline < NoContents:
        Split_Line = contents[xline].split()
        # Check if the line is not empty
        if Split_Line != []:
            # Check if the string includes the = symbol
            Equal = Split_Line[0].find('=')
            if Equal >= 0:
                Text = Split_Line[0][0:Equal]
            else:
                Text = Split_Line[0]

            # Check if the line contains one of the elements
            xelement = 0
            while xelement < NoElements:
                if Text == "mpc." + Elements[xelement]:
                    break
                xelement += 1

            # Has an element been found?
            if xelement < NoElements:
                # Number of strings in this line
                NoStrings = len(Split_Line)

                if Components[xelement] == 1:
                    # Should the first sting be checked?
                    if Equal >= 0:
                        xstring = 0
                    else:
                        xstring = 1

                    # Look for the numeric value in this line
                    while xstring < NoStrings:
                        a, b = _get_digit_string(Split_Line[xstring])
                        if a:
                            Vars['mpc'][Elements[xelement]] = b
                            break
                        xstring += 1

                    # Raise exception if the number was not found
                    if xstring == NoStrings:
                        print('\nThe information in %s '
                              % Elements[xelement], end='')
                        print(' must be in a single line in the *.m file')
                        raise Exception('Sorry, please edit the *.m file')
                else:
                    # Check if there are values in the first line
                    xstring = 0
                    if NoStrings > Components[xelement]:
                        a = False
                        while not a:
                            a, _ = _get_digit_string(Split_Line[xstring])
                            xstring += 1
                    else:
                        xline += 1
                        Split_Line = contents[xline].split()
                        NoStrings = len(Split_Line)

                    Full_List = []
                    while NoStrings >= Components[xelement]:
                        Part_List = []
                        for xval in range(Components[xelement]):
                            a, b = _get_digit_string(Split_Line[xstring])

                            if not a:
                                print('\nThe numerical values in %s '
                                      % Elements[xelement], end='')
                                print(' could not be loaded.')

                                raise Exception('Sorry, please edit the',
                                                ' *.m file')

                            Part_List.append(b)
                            xstring += 1
                        Full_List.append(Part_List)

                        xstring = 0
                        xline += 1
                        Split_Line = contents[xline].split()
                        NoStrings = len(Split_Line)

                    Vars['mpc'][Elements[xelement]] = Full_List

        xline += 1
    path = path + 'at'
    from scipy.io import savemat
    savemat(path, Vars)


def _create_dataframe(dataframe_data: DataframeData) -> DataFrame:
    return DataFrame(data=dataframe_data.data, columns=dataframe_data.column_names)


def _get_digit_string(str):
    Num = []
    for character in str:
        if character.isdigit():
            Num.append(character)
    if Num == []:
        return False, ' '
    else:
        return True, float(''.join(Num))


def _read_active_columns_names(profile_data_dict: dict, data: DataFrame) -> List[str]:
    if profile_data_dict.get("active_columns_names", None) is not None:
        return profile_data_dict.pop("active_columns_names")
    elif profile_data_dict.get("all_active_columns_names", False):
        return data.columns.tolist()


def _read_file(path: str, profile_data_dict: dict, header: Any = 0) -> DataFrame:
    if profile_data_dict.get("excel_sheet_name", None) is None:
        raise NameError("Excel sheet name does not exist.")
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


def _calculate_total_apparent_power_per_scenario_and_per_year_for_optimisation_profiles(data: DataFrame) -> DataFrame:
    scenarios = data["scenario"].unique()
    years = data["year"].unique()
    apparent_power_system_for_all_scenarios_and_years = DataFrame()
    for scenario in scenarios:
        for year in years:
            apparent_power_system_for_all_scenarios_and_years = \
                concat([apparent_power_system_for_all_scenarios_and_years,
                        DataFrame(data=[[scenario, year, _calculate_total_system_apparent_power(data, scenario, year)]],
                                  columns=["scenario", "year", "s_mva"])], ignore_index=True)
    return apparent_power_system_for_all_scenarios_and_years


def _calculate_total_system_apparent_power(data, scenario, year):
    buses_data = data[(data["scenario"] == scenario) & (data["year"] == year)]
    apparent_power_system = 0.0
    for _, row in buses_data.iterrows():
        if row["p_mw"] is not None and row["q_mvar"] is not None:
            apparent_power_system += sqrt(row["p_mw"] ** 2 + row["q_mvar"] ** 2)
        else:
            raise ValueError(f"The dataframe does not contain data for p_mw and q_mvar for bus "
                             f"{row['bus_index']}")
    return apparent_power_system


def _normalise_system_apparent_power(data: DataFrame):
    if len(data[data["year"] == min(data["year"])]["s_mva"].unique()) > 1:
        raise ValueError(f"The dataframe does not contain the same system apparent power for the initial year "
                         f"{min(data['year'])} for all scenarios")
    s_mva = data["s_mva"]
    min_s = min(data[data["year"] == min(data["year"])]["s_mva"])
    normalised = s_mva/min_s
    data["normalised"] = normalised


class ReadJSON:
    def __init__(self):
        self.settings = None
        self.parameters = Parameters()

    def read_json_data(self, json_path: str):
        file = open(json_path)
        self.settings: dict = load(file)
        self._read_data_from_dictionary()

    def _read_data_from_dictionary(self):
        self.parameters.problem_settings = self._load_problem_settings()
        self.parameters.opf_time_settings = self._load_opf_time_settings()
        self.parameters.pandapower_mpc_settings = self._load_pandapower_mpc_settings()
        self.parameters.pandapower_profiles_data = self._load_pandapower_profiles_data()
        self.parameters.output_settings = self._load_output_settings()
        self.parameters.pandapower_optimisation_settings = self._load_pandapower_optimisation_settings()
        self.parameters.optimisation_profiles_data = self._load_optimisation_profiles_data()
        self._load_optimisation_binary_variables()
        self._adjust_pandapower_profiles_to_time_settings()
        self.parameters.initialised = True

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
            problem_settings.inter_temporal = \
                problem_settings_dict.pop("inter_temporal", False)
            problem_settings.opf_optimizer = \
                problem_settings_dict.pop("opf_optimizer", 'pandapower')
            problem_settings.problem_optimizer = \
                problem_settings_dict.pop("problem_optimizer", '')
            problem_settings.opf_type = \
                problem_settings_dict.pop("opf_type", 'ac')
            problem_settings.return_rate_in_percentage = \
                problem_settings_dict.pop("return_rate_in_percentage", 0.0)
            problem_settings.non_anticipative = \
                problem_settings_dict.pop("non_anticipative", False)
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
        pandapower_mpc_settings_dict: dict = \
            self.settings.pop("pandapower_mpc_settings", None)
        if pandapower_mpc_settings_dict is not None:

            # Is this a *.m file?
            MatPath = pandapower_mpc_settings_dict.pop("mat_file_path")
            if MatPath[-1] == 'm':
                # A *.mat file has to be created based on the *.m file
                _build_mat_file(MatPath)
                MatPath = MatPath + 'at'

            pandapower_mpc_settings.mat_file_path = MatPath
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
            format_data = profiles_data_dict.pop("format_data", None)
            if format_data is None:
                for value in profiles_data_dict.values():
                    profiles_data.data.append(_load_optimisation_profile_data(value))
                profiles_data.initialised = True
            elif format_data == "attest":
                data = profiles_data_dict.pop("data", None)
                if data is not None:
                    self._extract_optimisation_profiles_data(data)
                    profiles_data = self._create_optimisation_profiles_based_on_attest_dataframes()
                else:
                    raise ValueError("No data found in the profiles data")
            else:
                raise ValueError(f"Unknown format_data: {format_data}")
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
                pandapower_optimisation_settings_dict.pop("continue_on_divergence", False)
            pandapower_settings.display_progress_bar = \
                pandapower_optimisation_settings_dict.pop("display_progress_bar", False)
            pandapower_settings.optimisation_software = \
                pandapower_optimisation_settings_dict.pop("optimisation_software", "pypower")
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

    def _adjust_pandapower_profiles_to_time_settings(self):
        for profile in self.parameters.pandapower_profiles_data.data:
            if len(profile.data.index) > self.parameters.opf_time_settings.date_time_settings.size:
                self._recalculate_pandapower_profile_data(profile)
            elif len(profile.data.index) < self.parameters.opf_time_settings.date_time_settings.size:
                raise ValueError(f"Length of profile data for type {profile.element_type} and variable "
                                 f"{profile.variable_name} is {len(profile.data.index)}. The number of periods in the "
                                 f"OPF settings are {self.parameters.opf_time_settings.date_time_settings.size}. The"
                                 f" profile is not long enough to cover the time settings")

    def _recalculate_pandapower_profile_data(self, profile: PandaPowerProfileData):
        if len(profile.data.index) % self.parameters.opf_time_settings.date_time_settings.size == 0:
            number_of_sub_periods = \
                len(profile.data.index) // self.parameters.opf_time_settings.date_time_settings.size
            adjusted_data = []
            for period in range(self.parameters.opf_time_settings.date_time_settings.size):
                adjusted_data.append(list(profile.data.iloc[period * number_of_sub_periods:
                                                            (period + 1) * number_of_sub_periods, :].mean()))
            profile.data = DataFrame(adjusted_data, columns=profile.data.columns)
        else:
            raise ValueError(f"The number of periods in the profile {len(profile.data.index)} must be a multiple "
                             f"of the OPF time settings {self.parameters.opf_time_settings.date_time_settings.size}")

    def _extract_optimisation_profiles_data(self, data: List[dict]):
        self.parameters.optimisation_profiles_dataframes.create_dictionary()
        for profile in data:
            self.parameters.optimisation_profiles_dataframes.append(
                profile.pop("group"), DataFrame(data=profile.pop("data"), columns=profile.pop("columns_names")))

    def _create_optimisation_profiles_based_on_attest_dataframes(self) -> OptimisationProfilesData:
        year_scenario_apparent_power = \
            _calculate_total_apparent_power_per_scenario_and_per_year_for_optimisation_profiles(
                self.parameters.optimisation_profiles_dataframes["buses"])
        _normalise_system_apparent_power(year_scenario_apparent_power)
        scenarios = year_scenario_apparent_power["scenario"].unique()
        years = list(year_scenario_apparent_power["year"].unique())
        years.sort()
        multipliers = []
        for year in years:
            multipliers_per_year = []
            for scenario in scenarios:
                multipliers_per_year.append(float(year_scenario_apparent_power[
                                                (year_scenario_apparent_power["year"] == year) &
                                                (year_scenario_apparent_power["scenario"] == scenario)]["normalised"]))
            multipliers.append(multipliers_per_year)
        rows = []
        for year in years:
            rows.append(str(year))
        columns = []
        for scenario in scenarios:
            columns.append(f"scenario {scenario}")
        optimisation_data = OptimisationProfileData()
        optimisation_data.data = DataFrame(data=multipliers, columns=columns, index=rows)
        profiles_data = OptimisationProfilesData()
        profiles_data.data = [optimisation_data]
        profiles_data.initialised = True
        return profiles_data
