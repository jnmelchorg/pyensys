#include <vector>
#include <string>
#include <set>

//      NAMES ELEMENTS NODES

std::set<std::string> integer_parameters_nodes {
    "number",
    "typePF"
};

std::set<std::string> double_parameters_nodes {
    "Pd",
    "Qd",
    "Gs",
    "Bs",
    "baseV",
    "Vmpr",
    "Vapr",
    "Vmax",
    "Vmin"
};

std::set<std::string> string_parameters_nodes {
    "ID",
    "name_bus",
    "zone",
    "group",
    "subtype"
};

std::set<std::string> bool_parameters_nodes {};

//      NAMES ELEMENTS BRANCHES

std::set<std::string> integer_parameters_branches {
    "from",
    "to"
};

std::set<std::string> double_parameters_branches {
    "resistance",
    "reactance",
    "LCsusceptance",
    "maxPflow",
    "CTEP",
    "Vmpr",
    "Vapr",
    "Vmax",
    "Vmin"
};

std::set<std::string> string_parameters_branches {
    "ID",
    "subtype",
    "group"
};

std::set<std::string> bool_parameters_branches {
    "status",
    "vTEP"
};

//      NAMES ELEMENTS GENERATORS

std::set<std::string> integer_parameters_generators {
    "number",
    "model",
    "pieces"
};

std::set<std::string> double_parameters_generators {
    "Pmax",
    "Pmin",
    "Pfix",
    "Qmax",
    "Qmin",
    "Qfix",
    "cUC",
    "cGEP",
    "fCPg",
    "vCPg",
    "emissions",
    "startup",
    "shutdown",
    "cost function",
    "coefficients"
};

std::set<std::string> string_parameters_generators {
    "ID",
    "subtype",
    "group"
};

std::set<std::string> bool_parameters_generators {
    "status",
    "vUC",
    "vGEP"
};

// PROBLEMS NAMES

std::set<std::string> problems_names_nodes {
    "SEP"
};

std::set<std::string> problems_names_branches {
    "TEP"
};

std::set<std::string> problems_names_generators {
    "UC",
    "GEP"
};

std::set<std::string> problems_names_system {
    "DC ED",
    "DC OPF",
    "AC PF",
    "NetR",
    "BT"
};

// SUBTYPES

std::set<std::string> subtypes_branches = {
    "TL",
    "inter",
    "trafo",
    "user"
};

std::set<std::string> subtypes_generators {
    "thermal",
    "hydro",
    "wind",
    "solar",
    "user",
    "diesel"
};

// TREE PARAMETERS
std::set<std::string> integer_tree_characteristics = {
    "level"
};

std::set<std::string> double_tree_characteristics = {
    "input",
    "output",
    "weight"
};

std::set<std::string> string_tree_characteristics = {
    "name_node"
};

// MODEL PARAMETERS

std::set<std::string> bool_model_characteristics = {
    "loss",
    "multiperiod"
};

std::set<std::string> double_model_characteristics = {
    "Sbase"
};

std::set<std::string> string_model_characteristics = {
    "solver",
    "engine",
    "output file name"
};

// CONNECTIONS PARAMETERS

std::set<std::string> string_connections = {
    "type",
    "ID",
    "group",
    "zone",
    "problems",
    "variables",
    "subtype",
    "pt"
};

// OUTPUTS PARAMETERS
std::set<std::string> string_outputs = {
    "problem",
    "type information",
    "reference",
    "data device",
    "function"
};

// ACCUMULATED VALUES

std::set<std::string> integer_parameters { };
std::set<std::string> double_parameters {
    "hour"
};
std::set<std::string> string_parameters {
    "type",
    "name"
};
std::set<std::string> bool_parameters { };

std::set<std::string> problems;

std::set<std::string> all_parameters {
    "pt"
 };

std::set<std::string> variables_balance_tree {
    "input",
    "output",
    "flow",
    "surplus",
    "deficit"
};
std::set<std::string> variables_OPF {
    "active power generation",
    "active power flow",
    "voltage angle",
    "generation curtailment",
    "load curtailment",
    "active power generation cost"
};

void concatenate_names()
{
    integer_parameters.insert(integer_parameters_nodes.begin(), integer_parameters_nodes.end());
    integer_parameters.insert(integer_parameters_branches.begin(), integer_parameters_branches.end());
    integer_parameters.insert(integer_parameters_generators.begin(), integer_parameters_generators.end());
    integer_parameters.insert(integer_tree_characteristics.begin(), integer_tree_characteristics.end());

    double_parameters.insert(double_parameters_nodes.begin(), double_parameters_nodes.end());
    double_parameters.insert(double_parameters_branches.begin(), double_parameters_branches.end());
    double_parameters.insert(double_parameters_generators.begin(), double_parameters_generators.end());
    double_parameters.insert(double_tree_characteristics.begin(), double_tree_characteristics.end());

    string_parameters.insert(string_parameters_nodes.begin(), string_parameters_nodes.end());
    string_parameters.insert(string_parameters_branches.begin(), string_parameters_branches.end());
    string_parameters.insert(string_parameters_generators.begin(), string_parameters_generators.end());
    string_parameters.insert(string_tree_characteristics.begin(), string_tree_characteristics.end());
    string_parameters.insert(string_connections.begin(), string_connections.end());
    string_parameters.insert(string_outputs.begin(), string_outputs.end());

    bool_parameters.insert(bool_parameters_nodes.begin(), bool_parameters_nodes.end());
    bool_parameters.insert(bool_parameters_branches.begin(), bool_parameters_branches.end());
    bool_parameters.insert(bool_parameters_generators.begin(), bool_parameters_generators.end());

    problems.insert(problems_names_nodes.begin(), problems_names_nodes.end());
    problems.insert(problems_names_branches.begin(), problems_names_branches.end());
    problems.insert(problems_names_generators.begin(), problems_names_generators.end());
    problems.insert(problems_names_system.begin(), problems_names_system.end());

    all_parameters.insert(integer_parameters.begin(), integer_parameters.end());
    all_parameters.insert(double_parameters.begin(), double_parameters.end());
    all_parameters.insert(bool_parameters.begin(), bool_parameters.end());
    all_parameters.insert(string_parameters.begin(), string_parameters.end());

    all_parameters.insert(problems.begin(), problems.end());

    all_parameters.insert(double_model_characteristics.begin(), double_model_characteristics.end());
    all_parameters.insert(bool_model_characteristics.begin(), bool_model_characteristics.end());
    all_parameters.insert(string_model_characteristics.begin(), string_model_characteristics.end());
}
