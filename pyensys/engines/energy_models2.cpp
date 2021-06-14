#include "energy_models2.h"
#include "energy_models_names.h"

//TODO Change arguments functions to information

models::models()
{
    concatenate_names();
};

models::~models() 
{
};

void models::create_parameter()
{
    candidate.clear_characteristics();
};

void models::load_parameter(const std::string& na, const value_T& val, const bool& is_vector)
{
    if (na == "value")
        candidate.set_value(val);
    else
        candidate.set_characteristic(characteristic(na, val, is_vector), is_vector);
};

void models::load_double(const std::string& na, const double& val, const bool& is_vector)
{
    load_parameter(na, val, is_vector);
};

void models::load_integer(const std::string& na, const int& val, const bool& is_vector)
{ 
    load_parameter(na, val, is_vector);
};

void models::load_bool(const std::string& na, const bool& val, const bool& is_vector)
{ 
    load_parameter(na, val, is_vector);
};

void models::load_string(const std::string& na, const std::string& val, const bool& is_vector)
{
    load_parameter(na, val, is_vector);
};

void models::set_parameter(const std::string& typ)
{
    if (typ == "network") data_parameters.push_back(candidate, "network");
    else if (typ == "tree") data_parameters.push_back(candidate, "tree");
    else if (typ == "model") data_parameters.push_back(candidate, "model");
    else if (typ == "connections") data_parameters.push_back(candidate, "connections");
    else if (typ == "outputs") data_parameters.push_back(candidate, "outputs");
}

void models::create_nodes_tree()
{
    std::vector<characteristic> levels = levels_tree();
    std::vector< std::vector<characteristic> > elements_levels = names_levels_tree();
    std::vector<graph_data> nodes;
    int nodes_levels = 1;
    for (int pos=0; pos < levels.size(); pos++)
    {
        for (int i = 0; i < nodes_levels; i++)
        {
            for (characteristic& element : elements_levels[pos])
            {
                graph_data node;
                node.set_characteristic(levels[pos], false);
                node.set_characteristic(element, false);
                nodes.push_back(node);
            }
        }
        nodes_levels *= int(elements_levels[pos].size());
    }

    for(information& param : data_parameters.get_all_parameters_type("tree"))
    {
        for (graph_data& node : nodes)
        {
            if (node.get_characteristic("level").get_value() == param.get_characteristic("level").get_value() && node.get_characteristic("name_node").get_value() == param.get_characteristic("name_node").get_value())
            {
                node.push_back(param, "parameters");
                break;
            }
        }
    }
    for (graph_data& node : nodes)
        grafos.add_vertex("tree", node);

}

void models::create_edges_tree()
{
    std::vector<bool> connected_nodes (grafos.number_nodes("tree"), false);
    std::vector<characteristic> levels = levels_tree();
    std::vector< std::vector<characteristic> > elements_levels = names_levels_tree();
    for (size_t i = 0; i < levels.size() - 1; i++)
    {
        for (characteristic& reference : elements_levels[i])
        {
            for (characteristic& next : elements_levels[i + 1])
            {
                int origin = -1, destiny = -1;
                for(int pos = 0; pos < grafos.number_nodes("tree"); pos++)
                {
                    graph_data node = grafos.get("tree", int(pos), "vertex");
                    if (node.get_characteristic("level").get_value() == levels[i].get_value() && node.get_characteristic("name_node").get_value() == reference.get_value())
                    {
                        connected_nodes[pos] = true;
                        origin = pos;
                    }
                    else if (node.get_characteristic("level").get_value() == levels[i + 1].get_value() && node.get_characteristic("name_node").get_value() == next.get_value() && !connected_nodes[pos])
                    {
                        connected_nodes[pos] = true;
                        destiny = pos;
                    }
                    if (origin != -1 && destiny != -1)
                    {
                        graph_data branch;
                        branch.set_characteristic(characteristic("from", origin, false), false);
                        branch.set_characteristic(characteristic("to", destiny, false), false);
                        grafos.add_edge("tree", origin, destiny, branch);
                        break;
                    }
                }
            }
        }
    }
}

std::vector<characteristic> models::levels_tree()
{
    std::vector<characteristic> levels;
    for(information& param : data_parameters.get_all_parameters_type("tree"))
    {
        if (boost::get<std::string>(param.get_characteristic("name").get_value()) == "weight")
        {
            bool exist = false;
            for (characteristic& level : levels)
            {
                if (level.get_value() == param.get_characteristic("level").get_value())
                {
                    exist = true;
                    break;
                }
            }
            if (!exist)
                levels.push_back(param.get_characteristic("level"));
        }
    }
    std::sort(levels.begin(), levels.end(), [](characteristic& a, characteristic& b) {return a.get_value() < b.get_value();});
    return levels;
}

std::vector< std::vector<characteristic> > models::names_levels_tree()
{
    std::vector<characteristic> levels = levels_tree();
    std::vector< std::vector<characteristic> > elements_levels;
    for (characteristic& level : levels)
    {
        std::vector<characteristic> elements;
        for(information& param : data_parameters.get_all_parameters_type("tree"))
        {
            if (boost::get<std::string>(param.get_characteristic("name").get_value()) == "weight" && param.get_characteristic("level").get_value() == level.get_value())
            {
                bool exist = false;
                for (characteristic& element : elements)
                {
                    if (param.get_characteristic("name_node").get_value() == element.get_value())
                    {
                        exist = true;
                        break;
                    }
                }
                if (!exist)
                    elements.push_back(param.get_characteristic("name_node"));
            }
        }
        elements_levels.push_back(elements);
    }
    return elements_levels;
}

void models::create_nodes_network()
{
    std::vector<graph_data> nodes;
    for(information& param : data_parameters.get_all_parameters_type("network"))
    {
        bool exist = false;
        for (graph_data& node : nodes)
        {
            characteristic c1 = param.get_characteristic("ID");
            characteristic c2 = node.get_characteristic("ID");
            if (c1.get_name() == c2.get_name() && c1.get_value() == c2.get_value() && c1.get_values() == c2.get_values())
            {
                node.push_back(param, "parameters");
                exist = true;
                break;
            }
        }
        if (!exist)
        {
            graph_data vertice;
            vertice.push_back(param, "parameters");
            vertice.set_characteristic(param.get_characteristic("ID"), false);
            vertice.set_characteristic(param.get_characteristic("type"), false);
            if (param.exist("subtype"))
                vertice.set_characteristic(param.get_characteristic("subtype"), true);
            if (param.exist("typePF"))
                vertice.set_characteristic(param.get_characteristic("typePF"), true);
            if (param.get_characteristic("type").get_value() == value_T(std::string("branch")))
            {
                vertice.set_characteristic(param.get_characteristic("from"), false);
                vertice.set_characteristic(param.get_characteristic("to"), false);
            }
            else
                vertice.set_characteristic(param.get_characteristic("number"), false);
            nodes.push_back(vertice);
        }
    }
    for (graph_data& node : nodes)
        grafos.add_vertex("network", node);

}

void models::create_edges_network()
{
    for (size_t i = 0; i < grafos.number_nodes("network"); i++)
    {
        graph_data node = grafos.get("network", int(i), "vertex");
        if(node.get_characteristic("type").get_value() == value_T(std::string("generator")))
        {
            for (size_t j = 0; j < grafos.number_nodes("network"); j++)
            {
                graph_data comp = grafos.get("network", int(j), "vertex");
                if (comp.get_characteristic("type").get_value() == value_T(std::string("bus")) && node.get_characteristic("number").get_value() == comp.get_characteristic("number").get_value())
                {
                    graph_data branch;
                    grafos.add_edge("network", int(j), int(i), branch);
                    break;
                }
            }
        }
        else if(node.get_characteristic("type").get_value() == value_T(std::string("branch")))
        {
            bool flag1 = false, flag2 = false;
            for (size_t j = 0; j < grafos.number_nodes("network"); j++)
            {
                graph_data comp = grafos.get("network", int(j), "vertex");
                if (comp.get_characteristic("type").get_value() == value_T(std::string("bus")) && node.get_characteristic("from").get_value() == comp.get_characteristic("number").get_value())
                {
                    graph_data branch;
                    grafos.add_edge("network", int(j), int(i), branch);
                    flag1 = true;
                }
                else if (comp.get_characteristic("type").get_value() == value_T(std::string("bus")) && node.get_characteristic("to").get_value() == comp.get_characteristic("number").get_value())
                {
                    graph_data branch;
                    grafos.add_edge("network", int(j), int(i), branch);
                    flag2 = true;
                }
                else if (flag1 && flag2) break;
            }
        }
    }
}

std::pair<std::vector<std::string>, int> models::connections_energy_tree_model()
{
    std::vector<std::string> connections_IDs;
    std::string name_compare = "";
    for (information& info : data_parameters.get_all_parameters_type("connections"))
    {
        std::vector<value_T> values = info.get_characteristic("problems").get_values();
        std::vector<value_T>::iterator it_values = std::find(values.begin(), values.end(), value_T(std::string("BT")));
        int pos = int(std::distance(values.begin(), it_values));
        if (it_values != values.end() && name_compare == "")
        {
            if (pos < info.get_characteristic("variables").get_values().size())
            {
                name_compare = boost::get<std::string>(info.get_characteristic("variables").get_values()[pos]);
                connections_IDs.push_back(boost::get<std::string>(info.get_characteristic("ID").get_value()));
            }
            else
            {
                std::cout << "Incoherent data for connections, impossible to create balance tree model" << std::endl;
                connections_IDs.clear();
                return std::make_pair(connections_IDs, -1); // Code for failure in data connections
            }
        }
        else if (it_values != values.end() && name_compare ==  boost::get<std::string>(info.get_characteristic("variables").get_values()[std::distance(values.begin(), it_values)]))
            connections_IDs.push_back(boost::get<std::string>(info.get_characteristic("ID").get_value()));
    }
    return std::make_pair(connections_IDs, 0);
}

int models::create_problem_element(const std::string& name_var, const double max, const double min, std::vector<std::pair<characteristic, bool> >& extra_characteristics, const int node_graph, const std::string& name_graph, const std::string& graph_component, const std::string& type_info, const double cost)
{
    information info;
    int position;
    graph_data data = grafos.get(name_graph, node_graph, graph_component);
    info.set_characteristics(data.get_characteristics());
    info.set_characteristic(characteristic("name", name_var, false), false);
    if (max == COIN_DBL_MAX) info.set_characteristic(characteristic("max", std::string("infinite"), false), false);
    else info.set_characteristic(characteristic("max", max, false), false);
    if (min == -COIN_DBL_MAX) info.set_characteristic(characteristic("min", std::string("-infinite"), false), false);
    else info.set_characteristic(characteristic("min", min, false), false);

    if (type_info == "variables")
    {
        info.set_characteristic(characteristic("cost", cost, false), false);
        position = problem_matrix.add_variable(max, min, cost);
        info.set_characteristic(characteristic("position matrix", position, false), false);
    }
    else if (type_info == "constraints")
    {
        position = problem_matrix.add_constraint(max, min);
        info.set_characteristic(characteristic("position matrix", position, false), false);
    }
    for (const std::pair<characteristic, bool>& charac : extra_characteristics)
        info.set_characteristic(charac.first, charac.second);
    info.set_value(0.0);
    // grafos.push_back_info(name_graph, node_graph, graph_component, info, type_info);
    // data.push_back(info, type_info);
    // grafos.set(name_graph, node_graph, data, graph_component);

    // Test hash tables
    information set;
    set.set_characteristic(characteristic("name", name_var, false), false);
    if (info.exist("pt")) set.set_characteristic(info.get_characteristic("pt"), true);
    if (info.exist("hour")) set.set_characteristic(info.get_characteristic("hour"), false);
    if (info.exist("piece"))
    {
        set.set_characteristic(info.get_characteristic("piece"), false);
    }

    grafos.insert2component(name_graph, node_graph, graph_component, type_info, set, info);
    return position;
}

int models::declare_balance_tree_variables(const std::string& id)
{
    std::vector<std::pair<characteristic, bool> > extra_characteristics;
    std::vector<characteristic> additional_characteristics;
    if (id != std::string("N/A"))
    {
        extra_characteristics.push_back(std::make_pair(characteristic("reference", id, false), false));
        additional_characteristics.push_back(characteristic("reference", id, false));
    }
    // Surplus

    int position_matrix;
    create_problem_element("surplus", COIN_DBL_MAX, 0.0, extra_characteristics, 0, "tree", "vertex", "variables", 0.0);

    // Defitic
    create_problem_element("deficit", COIN_DBL_MAX, 0.0, extra_characteristics, 0, "tree", "vertex", "variables", PENALTY);

    // inputs
    for (int node = 0; node < grafos.number_nodes("tree"); node++)
    {
        additional_characteristics.push_back(characteristic("name", std::string("input"), false));
        std::vector<information> multi_info = grafos.get("tree", node, "vertex").get_multi_info(additional_characteristics, "parameters", false);
        if (multi_info.size() == 1) 
        {
            position_matrix = create_problem_element("input", boost::get<double>(multi_info[0].get_value()) , boost::get<double>(multi_info[0].get_value()), extra_characteristics, node, "tree", "vertex", "variables", 0.0);
            information new_info = multi_info[0];
            new_info.set_characteristic(characteristic("column", position_matrix, true), true);
            graph_data vertex = grafos.get("tree", node, "vertex");
            vertex.replace_information(multi_info[0], new_info, "parameters");
            grafos.set("tree", node, vertex, "vertex");
        }
        else
        {
            std::cout << "Incoherent data for input, impossible to create balance tree model" << std::endl;
            return -2;
        }
    }

    // outputs
    for (int node = 0; node < grafos.number_nodes("tree"); node++)
    {
        std::vector<information> multi_info = grafos.get("tree", node, "vertex").get_multi_info(std::vector<characteristic>({characteristic("name", std::string("output"), false), characteristic("reference", id, false)}), "parameters", false);
        if (multi_info.size() == 1) 
            create_problem_element("output", boost::get<double>(multi_info[0].get_value()) , boost::get<double>(multi_info[0].get_value()), extra_characteristics, node, "tree", "vertex", "variables", 0.0);
        else
        {
            std::cout << "Incoherent data for output, impossible to create balance tree model" << std::endl;
            return -2;
        }
    }

    // flows
    for (int branch = 0; branch < grafos.number_edges("tree"); branch++)
        create_problem_element("flow", COIN_DBL_MAX ,0.0, extra_characteristics, branch, "tree", "edge", "variables", 0.0);
    
    return 0;
}

int models::declare_balance_tree_constraints(const std::string& id)
{
    std::vector<std::pair<characteristic, bool> > extra_characteristics;
    if (id != std::string("N/A")) extra_characteristics.push_back(std::make_pair(characteristic("reference", id, false), false));
    // balance
    for (int node = 0; node < grafos.number_nodes("tree"); node++)
        create_problem_element("balance", 0.0, 0.0, extra_characteristics, node, "tree", "vertex", "constraints", 0.0);
    return 0;
}

std::pair<information, int> models::get_information(const graph_data &data, const std::string& name_info, const std::vector<characteristic>& extra_characteristics, const std::string& name_datainfo, bool silent, const int pos, const std::string name_graph, const std::string component)
{
    if (pos == -1)
    {    
        std::vector<characteristic> characteristics = data.get_characteristics();
        characteristics.push_back(characteristic("name", name_info, false));
        characteristics.insert(characteristics.end(), extra_characteristics.begin(), extra_characteristics.end());
        std::vector<information> info = data.get_multi_info(characteristics, name_datainfo, silent);
        if (info.size() > 1)
        {
            if (!silent) std::cout << "Incoherent data for name *" << name_info << "* in data *" << name_datainfo << "*." << std::endl;
            return std::make_pair(information(), -3);
        }
        if (info.size() == 0) return std::make_pair(information(), 0);
        return std::make_pair(info[0], 0);
    }
    else
    {
        std::vector<characteristic> characteristics = grafos.get(name_graph, pos, component).get_characteristics();
        characteristics.push_back(characteristic("name", name_info, false));
        characteristics.insert(characteristics.end(), extra_characteristics.begin(), extra_characteristics.end());
        std::vector<information> info = grafos.get_multi_info(name_graph, pos, component, characteristics, name_datainfo, silent);
        if (info.size() > 1)
        {
            if (!silent) std::cout << "Incoherent data for name *" << name_info << "* in data *" << name_datainfo << "*." << std::endl;
            return std::make_pair(information(), -3);
        }
        if (info.size() == 0) return std::make_pair(information(), 0);
        return std::make_pair(info[0], 0);
    }
}

int models::create_balance_tree_matrix(const std::string& id)
{
    std::vector<characteristic> extra_characteristics;
    if (id != std::string("N/A")) extra_characteristics.push_back(characteristic("reference", id, false));
    // balance
    std::pair<information, int> retrieve_info;
    int row = -1, column = -1;
    double value = 0.0;
    out_edge_iterator out, out_end;
    in_edge_iterator in, in_end;

    for (int node = 0; node < grafos.number_nodes("tree"); node++)
    {
        graph_data vertex = grafos.get("tree", node, "vertex");
        // extracting constraint position in matrix
        retrieve_info = get_information(vertex, "balance", extra_characteristics, "constraints", false);
        if (retrieve_info.second != 0) return retrieve_info.second;
        row = boost::get<int>(retrieve_info.first.get_characteristic("position matrix").get_value());
        problem_matrix.add_active("row", row);
        if (node == 0)
        {
            // Adding surplus to initial node
            retrieve_info = get_information(vertex, "surplus", extra_characteristics, "variables", false);
            if (retrieve_info.second != 0) return retrieve_info.second;
            column = boost::get<int>(retrieve_info.first.get_characteristic("position matrix").get_value());
            problem_matrix.add_value2matrix(-1.0, row, column);

            // Adding deficit to initial node
            retrieve_info = get_information(vertex, "deficit", extra_characteristics, "variables", false);
            if (retrieve_info.second != 0) return retrieve_info.second;
            column = boost::get<int>(retrieve_info.first.get_characteristic("position matrix").get_value());
            problem_matrix.add_value2matrix(1.0, row, column);
        }
        // Retrieving weight
        retrieve_info = get_information(vertex, "weight", std::vector<characteristic>(), "parameters", false);
        if (retrieve_info.second != 0) return retrieve_info.second;
        value = boost::get<double>(retrieve_info.first.get_value());

        // Adding input
        retrieve_info = get_information(vertex, "input", extra_characteristics, "variables", false);
        if (retrieve_info.second != 0) return retrieve_info.second;
        column = boost::get<int>(retrieve_info.first.get_characteristic("position matrix").get_value());
        problem_matrix.add_value2matrix(value, row, column);

        // Adding output
        retrieve_info = get_information(vertex, "output", extra_characteristics, "variables", false);
        if (retrieve_info.second != 0) return retrieve_info.second;
        column = boost::get<int>(retrieve_info.first.get_characteristic("position matrix").get_value());
        problem_matrix.add_value2matrix(-value, row, column);
        
        // adding in-flows
        GraphType grafo = grafos.get_grafo("tree");
        for (boost::tie(in, in_end)=boost::in_edges(node, grafo); in != in_end; ++in)
        {
            graph_data edge = grafo[*in];
            retrieve_info = get_information(edge, "flow", extra_characteristics, "variables", false);
            if (retrieve_info.second != 0) return retrieve_info.second;
            column = boost::get<int>(retrieve_info.first.get_characteristic("position matrix").get_value());
            problem_matrix.add_value2matrix(1.0, row, column);
        }

        // adding out-flows
        for (boost::tie(out, out_end)=boost::out_edges(node, grafo); out != out_end; ++out)
        {
            graph_data edge = grafo[*out];
            retrieve_info = get_information(edge, "flow", extra_characteristics, "variables", false);
            if (retrieve_info.second != 0) return retrieve_info.second;
            column = boost::get<int>(retrieve_info.first.get_characteristic("position matrix").get_value());
            problem_matrix.add_value2matrix(-1.0, row, column);
        }
    }
}

void models::update_parameters_references_tree(const std::vector<std::string>& references, const std::string& parameter_name)
{
    for (int node = 0; node < grafos.number_nodes("tree"); node++)
    {
        graph_data vertex = grafos.get("tree", node, "vertex");
        std::vector<information> m_info = vertex.get_multi_info(std::vector<characteristic>({characteristic("name", parameter_name, false)}), "parameters", false);
        if (m_info.size() == 1)
        {
            information base = m_info[0];
            int counter = 0;
            for (const std::string& id : references)
            {
                information child = base;
                child.set_characteristic(characteristic("reference", id, false), false);
                if (counter == 0) vertex.replace_information(base, child, "parameters");
                else
                    vertex.push_back(child, "parameters");
                counter++;
            }
            grafos.set("tree", node, vertex, "vertex");
        }
    }
}

int models::create_balance_tree_model()
{
    std::pair<std::vector<std::string>, int> aux;
    information DC_OPF = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("DC OPF"), false), characteristic("engine", std::string("pyene"), false)}), "model");
    if (DC_OPF.get_characteristics().size() > 0 && boost::get<bool>(DC_OPF.get_value()))
    {
        aux = connections_energy_tree_model();
        update_parameters_references_tree(aux.first, "input");
        update_parameters_references_tree(aux.first, "output");
    }
    int code = aux.second;
    if (code != 0) return code;
    // Declaring variables
    if (aux.first.size() > 0)
    {
        for (const std::string& id : aux.first)
        {
            code = declare_balance_tree_variables(id);
            if (code != 0) return code;
        }
    }
    else code = declare_balance_tree_variables("N/A");
    if (code != 0) return code;

    // Declaring constraints
    if (aux.first.size() > 0)
    {
        for (const std::string& id : aux.first)
        {
            code = declare_balance_tree_constraints(id);
            if (code != 0) return code;
        }
    }
    else code = declare_balance_tree_constraints("N/A");
    if (code != 0) return code;

    // Creating matrix of coefficients
    if (aux.first.size() > 0)
    {
        for (const std::string& id : aux.first)
        {
            code = create_balance_tree_matrix(id);
            if (code != 0) return code;
        }
    }
    else code = create_balance_tree_matrix("N/A");
    if (code != 0) return code;

    return 0;
}

std::vector< std::pair<value_T, value_T> > models::breakpoints_piecewise_linearisation(const information& info, const value_T max, const value_T min)
{
    // stores breakpoints of piecewise funtion
    std::vector< std::pair<value_T, value_T> > breakpoints;
    // If function is piecewise
    if (boost::get<int>(info.get_characteristic("model").get_value()) == 1)
    {
        bool even=true;
        std::pair<value_T, value_T> storage;
        for(const value_T& val : info.get_characteristic("coefficients").get_values())
        {
            if (even)
            {
                storage.first = boost::get<double>(val)/boost::get<double>(data_parameters.get_parameter_type(std::vector<characteristic>{characteristic("name", std::string("Sbase"), false)}, "model").get_value());
                even = false;
            }
            else
            {
                storage.second = val;
                breakpoints.push_back(storage);
                even = true;
            }
        }
    }
    // Else if function is polynomial
    else if (boost::get<int>(info.get_characteristic("model").get_value()) == 2)
    {
        value_T pieces = info.get_characteristic("pieces").get_value();
        for (int i=0; i < boost::get<int>(pieces) + 1; i++)
        {
            std::pair<value_T, value_T> storage;
            if (breakpoints.size() == 0) storage.first = min;
            else storage.first = boost::get<double>(breakpoints[i - 1].first)+(boost::get<double>(max)/double(boost::get<int>(pieces)));
            double y_value = 0, exponent = double(info.get_characteristic("coefficients").get_values().size() - 1);
            for(const value_T& val : info.get_characteristic("coefficients").get_values())
            {
                y_value += boost::get<double>(val) * std::pow(boost::get<double>(storage.first) * boost::get<double>(data_parameters.get_parameter_type(std::vector<characteristic>{characteristic("name", std::string("Sbase"), false)}, "model").get_value()), exponent);
                exponent--;
            }
            storage.second = value_T(y_value);
            breakpoints.push_back(storage);
        }
    }
    return breakpoints;
}

std::pair<std::vector<std::pair<characteristic, bool> >, int> models::piecewise_linearisation(information& info, const value_T max, const value_T min)
{
    std::vector< std::pair<value_T, value_T> > breakpoints = breakpoints_piecewise_linearisation(info, max, min);
    if (breakpoints.size() == 0)
    {
        std::cout << "WARNING! Issue creating piecewise linearisation of parameter *" << info.get_characteristic("name").get_value() << "* with ID *" << info.get_characteristic("ID").get_value() << "*." << std::endl;
        return std::make_pair(std::vector<std::pair<characteristic, bool> >(), -7);
    }
    // Calculate Slope and end point
    std::vector<value_T> slopes;
    std::vector<value_T> y_intercepts;
    for (size_t pos=1; pos < breakpoints.size(); pos++)
    {   
        if ((boost::get<double>(breakpoints[pos].first) - boost::get<double>(breakpoints[pos - 1].first)) != 0)
            slopes.push_back(((boost::get<double>(breakpoints[pos].second) - boost::get<double>(breakpoints[pos - 1].second))/(boost::get<double>(breakpoints[pos].first) - boost::get<double>(breakpoints[pos - 1].first))));
        else slopes.push_back(0.0);
        y_intercepts.push_back(boost::get<double>(boost::get<double>(breakpoints[pos - 1].second)) - (boost::get<double>(boost::get<double>(breakpoints[pos - 1].first)) * boost::get<double>(slopes[pos - 1])));
    }
    std::vector<std::pair<characteristic, bool>> p_cha;
    characteristic cha;
    cha.set_name("slopes");
    cha.insert(slopes);
    p_cha.push_back(std::make_pair(cha, true));
    cha.set_name("y_intercepts");
    cha.clear_values();
    cha.insert(y_intercepts);
    p_cha.push_back(std::make_pair(cha, true));
    return std::make_pair(p_cha, 0);
}

int models::declare_dc_opf_variables(const std::vector<information>& subscripts)
{
    int column;
    for (int &node : grafos.get_positions("network", "bus"))
    {
        for(const information& subscript : subscripts)
        {
            std::vector<std::pair<characteristic, bool> > extra_characteristics;
            if (subscript.get_characteristic("pt").get_values().size() > 0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("pt"), true));
            if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("hour"), false));
            
            // Voltage angle
            column = create_problem_element("voltage angle", COIN_DBL_MAX , -COIN_DBL_MAX, extra_characteristics, node, "network", "vertex", "variables", 0.0);
            problem_matrix.add_active("column", column, subscript);
            // Load curtailment
            column = create_problem_element("load curtailment", COIN_DBL_MAX , 0.0, extra_characteristics, node, "network", "vertex", "variables", PENALTY);
            problem_matrix.add_active("column", column, subscript);
            // Generation curtailment
            column = create_problem_element("generation curtailment", COIN_DBL_MAX , 0.0, extra_characteristics, node, "network", "vertex", "variables", PENALTY);
            problem_matrix.add_active("column", column, subscript);
        }
    }

    for (int &node : grafos.get_positions("network", "generator"))
    {
        for(const information& subscript : subscripts)
        {
            std::vector<std::pair<characteristic, bool> > extra_characteristics;
            if (subscript.get_characteristic("pt").get_values().size() > 0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("pt"), true));
            if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("hour"), false));
            bool status;
            std::pair<information, int> info;
            std::vector<characteristic> characteristics;
            for (std::pair<characteristic, bool>& cha : extra_characteristics)
                characteristics.push_back(cha.first);
            // Extracting status of generator
            info = get_information(graph_data(), "status", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "status", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            status = boost::get<bool>(info.first.get_value());

            double cost, min, max;
            // Extracting cost for variable generation
            info = get_information(graph_data(), "vCPg", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.first.get_characteristics().size() == 0 && info.second != 0)
            {
                info = get_information(graph_data(), "vCPg", characteristics, "parameters", false, node, "network", "vertex");
                if (info.second != 0) return -6;
                cost = boost::get<double>(info.first.get_value());
            }
            else if (info.first.get_characteristics().size() == 0 && info.second == 0) cost = 0.0;
            else if (info.first.get_characteristics().size() > 0) cost = boost::get<double>(info.first.get_value());
            // Extracting minimum generation capacity
            info = get_information(graph_data(), "Pmin", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "Pmin", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            min = boost::get<double>(info.first.get_value());
            // Extracting maximum generation capacity
            info = get_information(graph_data(), "active power max limit", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "active power max limit", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            max = boost::get<double>(info.first.get_value());
            // Active power generation
            column = create_problem_element("active power generation", max, min, extra_characteristics, node, "network", "vertex", "variables", cost);
            if (status == true)
                problem_matrix.add_active("column", column, subscript);

            // Extracting cost for variable generation
            info = get_information(graph_data(), "cost function", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if ((info.first.get_characteristics().size() == 0 && info.second != 0) || info.first.get_characteristics().size() > 0)
            {
                if (info.second != 0)
                    info = get_information(graph_data(), "cost function", characteristics, "parameters", false, node, "network", "vertex");
                if (info.second != 0) return -6;
                if (!info.first.exist("slopes"))
                {
                    std::pair<std::vector< std::pair<characteristic, bool> >, int> pw = piecewise_linearisation(info.first, max, min);
                    if (pw.second != 0) return pw.second;
                    grafos.push_back_info("network", node, "vertex", info.first, "parameters", pw.first);
                }
                // Active power generation cost
                column = create_problem_element("active power generation cost", COIN_DBL_MAX, 0.0, extra_characteristics, node, "network", "vertex", "variables", 1.0);
                if (status == true)
                    problem_matrix.add_active("column", column, subscript);
            }
        }
    }

    return 0;
}

int models::convert2per_unit()
{
    double Sbase = boost::get<double>(data_parameters.get_parameter_type(std::vector<characteristic>{characteristic("name", std::string("Sbase"), false)}, "model").get_value());
    for (int node = 0; node < grafos.number_nodes("network"); node++)
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        bool changed = false;
        if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "bus")
        {
            std::vector<information> m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("Pd"), false)}, "parameters", false);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
        }
        else if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "generator")
        {
            std::vector<information> m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("active power max limit"), false)}, "parameters", false);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
            m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("Pmin"), false)}, "parameters", false);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
            m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("Qmax"), false)}, "parameters", true);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
            m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("Qmin"), false)}, "parameters", true);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
            m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("Pfix"), false)}, "parameters", true);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
            m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("Qfix"), false)}, "parameters", true);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
        }
        else if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "branch")
        {
            std::vector<information> m_info = vertex.get_multi_info(std::vector<characteristic>{characteristic("name", std::string("maxPflow"), false)}, "parameters", false);
            for (information& info : m_info)
            {
                info.set_value(boost::get<double>(info.get_value()) / Sbase);
                vertex.update_value(info, "parameters");
                changed = true;
            }
        }
        if (changed) grafos.set("network", node, vertex, "vertex");
    }
}

int models::declare_dc_opf_constraints(const std::vector<information>& subscripts)
{
    int row;
    for (int &node : grafos.get_positions("network", "bus"))
    {
        for(const information& subscript : subscripts)
        {
            std::vector<std::pair<characteristic, bool> > extra_characteristics;
            if (subscript.get_characteristic("pt").get_values().size() > 0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("pt"), true));
            if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("hour"), false));

            std::pair<information, int> info;
            std::vector<characteristic> characteristics;
            for (std::pair<characteristic, bool>& cha : extra_characteristics)
                characteristics.push_back(cha.first);
            
            double min, max;
            // Extracting active power demand
            info = get_information(graph_data(), "Pd", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "Pd", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            max = min = boost::get<double>(info.first.get_value());
            row = create_problem_element("active power balance", max, min, extra_characteristics, node, "network", "vertex", "constraints", 0.0);
            problem_matrix.add_active("row", row, subscript);
        }
    }
    
    for (int &node : grafos.get_positions("network", "generator"))
    {
        for(const information& subscript : subscripts)
        {
            std::vector<std::pair<characteristic, bool> > extra_characteristics;
            if (subscript.get_characteristic("pt").get_values().size() > 0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("pt"), true));
            if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("hour"), false));

            std::pair<information, int> info;
            std::vector<characteristic> characteristics;
            for (std::pair<characteristic, bool>& cha : extra_characteristics)
                characteristics.push_back(cha.first);
            
            // Extracting status of generator
            info = get_information(graph_data(), "status", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "status", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            bool status = boost::get<bool>(info.first.get_value());
            // Extracting cost for variable generation
            info = get_information(graph_data(), "cost function", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if ((info.first.get_characteristics().size() == 0 && info.second != 0) || info.first.get_characteristics().size() > 0)
            {
                if (info.second != 0)
                    info = get_information(graph_data(), "cost function", characteristics, "parameters", false, node, "network", "vertex");
                if (info.second != 0) return -6;
                std::vector<value_T> intercepts = info.first.get_characteristic("y_intercepts").get_values();
                for (int n_piece = 0; n_piece < boost::get<int>(info.first.get_characteristic("pieces").get_value()); n_piece++)
                {
                    std::vector<std::pair<characteristic, bool> > e_characteristics = extra_characteristics;
                    e_characteristics.push_back(std::make_pair(characteristic("piece", n_piece, false), false));
                    row = create_problem_element("piecewise generation cost", -boost::get<double>(intercepts[n_piece]), -COIN_DBL_MAX, e_characteristics, node, "network", "vertex", "constraints", 0.0);
                    if (status) problem_matrix.add_active("row", row, subscript);
                }
            }
        }
    }

    for (int &node : grafos.get_positions("network", "branch"))
    {
        for(const information& subscript : subscripts)
        {
            std::vector<std::pair<characteristic, bool> > extra_characteristics;
            if (subscript.get_characteristic("pt").get_values().size() > 0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("pt"), true));
            if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) extra_characteristics.push_back(std::make_pair(subscript.get_characteristic("hour"), false));

            std::pair<information, int> info;
            std::vector<characteristic> characteristics;
            for (std::pair<characteristic, bool>& cha : extra_characteristics)
                characteristics.push_back(cha.first);

            double min, max, max_flow, reactance;
            // Extracting max power flow
            info = get_information(graph_data(), "maxPflow", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "maxPflow", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            max_flow = boost::get<double>(info.first.get_value());
            // Extracting reactance
            info = get_information(graph_data(), "reactance", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "reactance", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            reactance = boost::get<double>(info.first.get_value());
            if (max_flow == 0.0)
            {
                max = COIN_DBL_MAX;
                min = -COIN_DBL_MAX;
            }
            else
            {
                max = max_flow * reactance;
                min = -max_flow * reactance;
            }
            int row = create_problem_element("angular difference", max, min, extra_characteristics, node, "network", "vertex", "constraints", 0.0);
            // Extracting status of branches
            info = get_information(graph_data(), "status", std::vector<characteristic>(), "parameters", true, node, "network", "vertex");
            if (info.second != 0)
                info = get_information(graph_data(), "status", characteristics, "parameters", false, node, "network", "vertex");
            if (info.second != 0) return -6;
            bool status = boost::get<bool>(info.first.get_value());
            if (status) problem_matrix.add_active("row", row, subscript);
        }
    }
}

std::pair<std::vector<std::vector<double> >, int> models::create_dc_opf_susceptance_matrix(const std::vector<value_T>& pt, const double hour)
{
    std::vector<std::pair<characteristic, bool> > extra_characteristics;
    if (pt.size() > 0)
    {
        characteristic extra;
        extra.set_name("pt");
        extra.insert(pt);
        extra_characteristics.push_back(std::make_pair(extra, true));
    }
    if (hour != -1.0)  extra_characteristics.push_back(std::make_pair(characteristic("hour", hour, false), false));

    int number_active_nodes = 0;
    std::vector<int> numbers;
    for (int node = 0; node < grafos.number_nodes("network"); node++)
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "bus")
        {
            int typePF = 1;
            if (vertex.get_characteristic("typePF").get_value() != value_T("N/A"))
                typePF = boost::get<int>(vertex.get_characteristic("typePF").get_value());
            if (typePF != 4) // Isolated nodes
            {
                number_active_nodes++;
                numbers.push_back(boost::get<int>(vertex.get_characteristic("number").get_value()));
            }
        }
    }

    std::vector<std::vector<double> > susceptance_matrix(number_active_nodes, std::vector<double>(number_active_nodes, 0.0));
    for (int node = 0; node < grafos.number_nodes("network"); node++)
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "branch")
        {
            int node, pos1, pos2;
            double reactance;
            bool status;
            if (vertex.get_characteristic("from").get_value() != value_T("N/A"))
                node = boost::get<int>(vertex.get_characteristic("from").get_value());
            std::vector<int>::iterator it = std::find(numbers.begin(), numbers.end(), node);
            if (it == numbers.end())
            {
                std::cout << "WARNING! Inconsistent data, impossible to create susceptance matrix" << std::endl;
                return std::make_pair(std::vector<std::vector<double> >(), -8);
            }
            pos1 = it - numbers.begin();
            if (vertex.get_characteristic("to").get_value() != value_T("N/A"))
                node = boost::get<int>(vertex.get_characteristic("to").get_value());
            it = std::find(numbers.begin(), numbers.end(), node);
            if (it == numbers.end())
            {
                std::cout << "WARNING! Inconsistent data, impossible to create susceptance matrix" << std::endl;
                return std::make_pair(std::vector<std::vector<double> >(), -8);
            }
            pos2 = it - numbers.begin();
            // Extracting reactance
            std::pair<information, int> info;
            std::vector<characteristic> characteristics;
            for (std::pair<characteristic, bool>& cha : extra_characteristics)
                characteristics.push_back(cha.first);
            info = get_information(vertex, "reactance", std::vector<characteristic>(), "parameters", true);
            if (info.second != 0)
                info = get_information(vertex, "reactance", characteristics, "parameters", false);
            if (info.second != 0) return std::make_pair(std::vector<std::vector<double> >(), -8);
            reactance = boost::get<double>(info.first.get_value());

            // Extracting status of branch
            info = get_information(vertex, "status", std::vector<characteristic>(), "parameters", true);
            if (info.second != 0)
                info = get_information(vertex, "status", characteristics, "parameters", false);
            if (info.second != 0) return std::make_pair(std::vector<std::vector<double> >(), -8);
            status = boost::get<bool>(info.first.get_value());
            if (status == true && std::abs(reactance) > TOLERANCE_REACTANCE)
            {
                susceptance_matrix[pos1][pos2] += 1/reactance;
                susceptance_matrix[pos2][pos1] += 1/reactance;
                susceptance_matrix[pos1][pos1] -= 1/reactance;
                susceptance_matrix[pos2][pos2] -= 1/reactance;
            }
        }
    }
    return std::make_pair(susceptance_matrix, 0);
}

int models::get_position_matrix_dc_opf(const std::string& name_var_con, graph_data& vertex, const std::vector<characteristic>& characteristics, const std::string& var_con)
{
    information info2;
    info2.set_characteristics(characteristics);

    information set;
    set.set_characteristic(characteristic("name", name_var_con, false), false);
    if (info2.exist("pt")) set.set_characteristic(info2.get_characteristic("pt"), true);
    if (info2.exist("hour")) set.set_characteristic(info2.get_characteristic("hour"), false);
    if (info2.exist("piece")) set.set_characteristic(info2.get_characteristic("piece"), false);

    info2 = vertex.find(var_con, set);

    if (info2.get_characteristics().size() == 0) return -1;
    return boost::get<int>(info2.get_characteristic("position matrix").get_value());
}

std::pair< int, std::vector<information> >  models::create_dc_opf_matrix(const std::vector<information>& subscripts){

    // auto start = std::chrono::steady_clock::now();
    std::vector<characteristic> characteristics;
    std::pair<information, int> info;

    int row = -1, column = -1;
    double value = 0.0;
    out_edge_iterator out, out_end;
    in_edge_iterator in, in_end;

    // Declaring information to be stored
    std::vector<information> coefficient_matrix_info;
    coefficient_matrix_info.push_back(information()); // Store matrix coefficients
    characteristic rows;
    rows.set_name("rows");
    characteristic columns;
    columns.set_name("columns");
    characteristic elements;
    elements.set_name("elements");

    int number_nodes = 0;
    std::vector<int> numbers;
    std::vector<int> position_matrix;
    for (int &node : grafos.get_positions("network", "bus"))
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        number_nodes++;
        numbers.push_back(boost::get<int>(vertex.get_characteristic("number").get_value()));
    }

    // auto end = std::chrono::steady_clock::now();
    // std::cout << "Elapsed time in milliseconds: "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //     << " ms" << std::endl;

    // start = std::chrono::steady_clock::now();
    for (int &node : grafos.get_positions("network", "bus"))
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        std::vector<double> susceptance_matrix_row(number_nodes, 0.0);
        // Active power balance
        std::vector<int>::iterator it = std::find(numbers.begin(), numbers.end(), boost::get<int>(vertex.get_characteristic("number").get_value()));
        if (it == numbers.end())
        {
            std::cout << "WARNING! Inconsistent data, impossible to create active power balace constraint coefficients" << std::endl;
            return std::make_pair(-9, std::vector<information>());
        }
        int bus_number = it - numbers.begin();
        GraphType grafo = grafos.get_grafo("network");
        for (boost::tie(in, in_end)=boost::in_edges(node, grafo); in != in_end; ++in)
        {
            graph_data source = grafos.get("network", int(boost::source(*in, grafo)), "vertex");
            for(const information& subscript : subscripts)
            {
                if (subscript.get_characteristic("pt").get_values().size() > 0) characteristics.push_back(subscript.get_characteristic("pt"));
                if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) characteristics.push_back(subscript.get_characteristic("hour"));
                // extracting constraint position in matrix
                
                row = get_position_matrix_dc_opf("active power balance", vertex, characteristics, "constraints");
                if (row == -1) return std::make_pair(-9, std::vector<information>());
                if (boost::get<std::string>(source.get_characteristic("type").get_value()) == "generator")
                {
                    // Adding active power generation
                    column = get_position_matrix_dc_opf("active power generation", source, characteristics, "variables");
                    if (column == -1) return std::make_pair(-9, std::vector<information>());
                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(1.0);
                }
                characteristics.clear();
            }
            if (boost::get<std::string>(source.get_characteristic("type").get_value()) == "branch")
            {
                info = get_information(source, "status", std::vector<characteristic>(), "parameters", true);
                if (info.second != 0)
                    info = get_information(source, "status", characteristics, "parameters", false);
                if (info.second != 0) return std::make_pair(-9, std::vector<information>());
                bool status = boost::get<bool>(info.first.get_value());
                info = get_information(source, "reactance", std::vector<characteristic>(), "parameters", true);
                if (info.second != 0)
                    info = get_information(source, "reactance", characteristics, "parameters", false);
                if (info.second != 0) return std::make_pair(-9, std::vector<information>());
                double reactance = boost::get<double>(info.first.get_value());
                if (status == true && std::abs(reactance) > TOLERANCE_REACTANCE)
                {
                    std::vector<int>::iterator it = std::find(numbers.begin(), numbers.end(), boost::get<int>(source.get_characteristic("from").get_value()));
                    if (it == numbers.end())
                    {
                        std::cout << "WARNING! Inconsistent data, impossible to create active power balace constraint coefficients" << std::endl;
                        return std::make_pair(-9, std::vector<information>());
                    }
                    int pos1 = it - numbers.begin();
                    it = std::find(numbers.begin(), numbers.end(), boost::get<int>(source.get_characteristic("to").get_value()));
                    if (it == numbers.end())
                    {
                        std::cout << "WARNING! Inconsistent data, impossible to create active power balace constraint coefficients" << std::endl;
                        return std::make_pair(-9, std::vector<information>());
                    }
                    int pos2 = it - numbers.begin();
                    if (bus_number == pos1)
                    {
                        susceptance_matrix_row[pos1] -= 1/reactance;
                        susceptance_matrix_row[pos2] += 1/reactance;
                    }
                    else
                    {
                        susceptance_matrix_row[pos2] -= 1/reactance;
                        susceptance_matrix_row[pos1] += 1/reactance;
                    }
                }
            }
        }
        for (boost::tie(out, out_end)=boost::out_edges(node, grafo); out != out_end; ++out)
        {
            graph_data target = grafos.get("network", int(boost::target(*out, grafo)), "vertex");
            for(const information& subscript : subscripts)
            {
                if (subscript.get_characteristic("pt").get_values().size() > 0) characteristics.push_back(subscript.get_characteristic("pt"));
                if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) characteristics.push_back(subscript.get_characteristic("hour"));
                // extracting constraint position in matrix
                row = get_position_matrix_dc_opf("active power balance", vertex, characteristics, "constraints");
                if (row == -1) return std::make_pair(-9, std::vector<information>());
                if (boost::get<std::string>(target.get_characteristic("type").get_value()) == "generator")
                {
                    // Adding active power generation
                    column = get_position_matrix_dc_opf("active power generation", target, characteristics, "variables");
                    if (column == -1) return std::make_pair(-9, std::vector<information>());
                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(1.0);
                }
                characteristics.clear();
            }
            if (boost::get<std::string>(target.get_characteristic("type").get_value()) == "branch")
            {
                info = get_information(target, "status", std::vector<characteristic>(), "parameters", true);
                if (info.second != 0)
                    info = get_information(target, "status", characteristics, "parameters", false);
                if (info.second != 0) return std::make_pair(-9, std::vector<information>());
                bool status = boost::get<bool>(info.first.get_value());
                info = get_information(target, "reactance", std::vector<characteristic>(), "parameters", true);
                if (info.second != 0)
                    info = get_information(target, "reactance", characteristics, "parameters", false);
                if (info.second != 0) return std::make_pair(-9, std::vector<information>());
                double reactance = boost::get<double>(info.first.get_value());
                if (status == true && std::abs(reactance) > TOLERANCE_REACTANCE)
                {
                    std::vector<int>::iterator it = std::find(numbers.begin(), numbers.end(), boost::get<int>(target.get_characteristic("from").get_value()));
                    if (it == numbers.end())
                    {
                        std::cout << "WARNING! Inconsistent data, impossible to create active power balace constraint coefficients" << std::endl;
                        return std::make_pair(-9, std::vector<information>());
                    }
                    int pos1 = it - numbers.begin();
                    it = std::find(numbers.begin(), numbers.end(), boost::get<int>(target.get_characteristic("to").get_value()));
                    if (it == numbers.end())
                    {
                        std::cout << "WARNING! Inconsistent data, impossible to create active power balace constraint coefficients" << std::endl;
                        return std::make_pair(-9, std::vector<information>());
                    }
                    int pos2 = it - numbers.begin();
                    if (bus_number == pos1)
                    {
                        susceptance_matrix_row[pos1] -= 1/reactance;
                        susceptance_matrix_row[pos2] += 1/reactance;
                    }
                    else
                    {
                        susceptance_matrix_row[pos2] -= 1/reactance;
                        susceptance_matrix_row[pos1] += 1/reactance;
                    }
                }
            }
        }
        for (int &node_aux : grafos.get_positions("network", "bus"))
        {
            graph_data angle_vertex = grafos.get("network", node_aux, "vertex");
            for(const information& subscript : subscripts)
            {
                if (subscript.get_characteristic("pt").get_values().size() > 0) characteristics.push_back(subscript.get_characteristic("pt"));
                if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) characteristics.push_back(subscript.get_characteristic("hour"));
                // extracting constraint position in matrix
                row = get_position_matrix_dc_opf("active power balance", vertex, characteristics, "constraints");
                if (row == -1) return std::make_pair(-9, std::vector<information>());
                std::vector<int>::iterator it = std::find(numbers.begin(), numbers.end(), boost::get<int>(angle_vertex.get_characteristic("number").get_value()));
                if (it == numbers.end())
                {
                    std::cout << "WARNING! Inconsistent data, impossible to create active power balace constraint coefficients" << std::endl;
                    return std::make_pair(-9, std::vector<information>());
                }
                int pos = it - numbers.begin();
                if (susceptance_matrix_row[pos] != 0)
                {
                    // Adding voltage angles
                    column = get_position_matrix_dc_opf("voltage angle", angle_vertex, characteristics, "variables");
                    if (column == -1) return std::make_pair(-9, std::vector<information>());
                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(susceptance_matrix_row[pos]);
                }
                characteristics.clear();
            }
        }
        for(const information& subscript : subscripts)
        {
            if (subscript.get_characteristic("pt").get_values().size() > 0) characteristics.push_back(subscript.get_characteristic("pt"));
            if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) characteristics.push_back(subscript.get_characteristic("hour"));

            // extracting constraint position in matrix
            row = get_position_matrix_dc_opf("active power balance", vertex, characteristics, "constraints");
            if (row == -1) return std::make_pair(-9, std::vector<information>());

            // Adding load curtailment
            column = get_position_matrix_dc_opf("load curtailment", vertex, characteristics, "variables");
            if (column == -1) return std::make_pair(-9, std::vector<information>());
            rows.push_back(row);
            columns.push_back(column);
            elements.push_back(1.0);

            // Adding generation curtailment
            column = get_position_matrix_dc_opf("generation curtailment", vertex, characteristics, "variables");
            if (column == -1) return std::make_pair(-9, std::vector<information>());
            rows.push_back(row);
            columns.push_back(column);
            elements.push_back(-1.0);
            characteristics.clear();
        }
    }
    // end = std::chrono::steady_clock::now();
    // std::cout << "Elapsed time in milliseconds: "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //     << " ms" << std::endl;

    // start = std::chrono::steady_clock::now();
    for (int &node : grafos.get_positions("network", "generator"))
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        // Extracting cost for variable generation
        info = get_information(vertex, "cost function", std::vector<characteristic>(), "parameters", true);
        if ((info.first.get_characteristics().size() == 0 && info.second != 0) || info.first.get_characteristics().size() > 0)
        {
            for(const information& subscript : subscripts)
            {
                if (subscript.get_characteristic("pt").get_values().size() > 0) characteristics.push_back(subscript.get_characteristic("pt"));
                if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) characteristics.push_back(subscript.get_characteristic("hour"));
                // extracting constraint position in matrix

                if (info.second != 0)
                info = get_information(vertex, "cost function", characteristics, "parameters", false);
                if (info.second != 0) return std::make_pair(-9, std::vector<information>());
                std::vector<value_T> slopes = info.first.get_characteristic("slopes").get_values();
                int pieces = boost::get<int>(info.first.get_characteristic("pieces").get_value());
                for (int n_piece = 0; n_piece < pieces; n_piece++)
                {
                    // Extracting row piecewise generation cost
                    std::vector<characteristic> e_characteristics = characteristics;
                    e_characteristics.push_back(characteristic("piece", n_piece, false));
                    row = get_position_matrix_dc_opf("piecewise generation cost", vertex, e_characteristics, "constraints");
                    if (row == -1) return std::make_pair(-9, std::vector<information>());

                    // Extracting column active power generation
                    column = get_position_matrix_dc_opf("active power generation", vertex, characteristics, "variables");
                    if (column == -1) return std::make_pair(-9, std::vector<information>());

                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(boost::get<double>(slopes[n_piece]));
                    // Extracting column active power generation cost
                    column = get_position_matrix_dc_opf("active power generation cost", vertex, characteristics, "variables");
                    if (column == -1) return std::make_pair(-9, std::vector<information>());

                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(-1.0);
                }
                characteristics.clear();
            }
        }
    }
    // end = std::chrono::steady_clock::now();
    // std::cout << "Elapsed time in milliseconds: "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //     << " ms" << std::endl;

    // start = std::chrono::steady_clock::now();
    for (int &node : grafos.get_positions("network", "branch"))
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        GraphType grafo = grafos.get_grafo("network");
        for (boost::tie(in, in_end)=boost::in_edges(node, grafo); in != in_end; ++in)
        {
            graph_data source = grafos.get("network", int(boost::source(*in, grafo)), "vertex");
            if (boost::get<std::string>(source.get_characteristic("type").get_value()) != "bus") return std::make_pair(-9, std::vector<information>());

            for(const information& subscript : subscripts)
            {
                if (subscript.get_characteristic("pt").get_values().size() > 0) characteristics.push_back(subscript.get_characteristic("pt"));
                if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) characteristics.push_back(subscript.get_characteristic("hour"));
                
                // Angular difference constraint
                // extracting constraint position in matrix
                row = get_position_matrix_dc_opf("angular difference", vertex, characteristics, "constraints");
                if (row == -1) return std::make_pair(-9, std::vector<information>());
                
                // extracting angle variable
                column = get_position_matrix_dc_opf("voltage angle", source, characteristics, "variables");
                if (column == -1) return std::make_pair(-9, std::vector<information>());
                if (int(in_end - in) % 2 == 0)
                {
                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(1.0);
                }
                else
                {
                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(-1.0);
                }
                characteristics.clear();
            }
        }
        for (boost::tie(out, out_end)=boost::out_edges(node, grafo); out != out_end; ++out)
        {
            graph_data target = grafos.get("network", int(boost::target(*out, grafo)), "vertex");
            if (boost::get<std::string>(target.get_characteristic("type").get_value()) != "bus") return std::make_pair(-9, std::vector<information>());
            
            for(const information& subscript : subscripts)
            {
                if (subscript.get_characteristic("pt").get_values().size() > 0) characteristics.push_back(subscript.get_characteristic("pt"));
                if (boost::get<double>(subscript.get_characteristic("hour").get_value()) != -1.0) characteristics.push_back(subscript.get_characteristic("hour"));
                
                // Angular difference constraint
                // extracting constraint position in matrix
                row = get_position_matrix_dc_opf("angular difference", vertex, characteristics, "constraints");
                if (row == -1) return std::make_pair(-9, std::vector<information>());
                
                // extracting angle variable
                column = get_position_matrix_dc_opf("voltage angle", target, characteristics, "variables");
                if (column == -1) return std::make_pair(-9, std::vector<information>());
                if (int(out_end - out) % 2 == 0)
                {
                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(-1.0);
                }
                else
                {
                    rows.push_back(row);
                    columns.push_back(column);
                    elements.push_back(1.0);
                }
                characteristics.clear();
            }
        }
    }
    // end = std::chrono::steady_clock::now();
    // std::cout << "Elapsed time in milliseconds: "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //     << " ms" << std::endl;
    coefficient_matrix_info[0].set_characteristic(rows, true);
    coefficient_matrix_info[0].set_characteristic(columns, true);
    coefficient_matrix_info[0].set_characteristic(elements, true);
    return std::make_pair(0, coefficient_matrix_info);
}

int models::connections_parameters_variables_dc_opf(const information& subscripts)
{
    std::vector<characteristic> extra_characteristics;
    if (subscripts.exist("pt") && subscripts.get_characteristic("pt").get_values().size() > 0)
    {
        characteristic extra;
        extra.set_name("pt");
        extra.insert(subscripts.get_characteristic("pt").get_values());
        extra_characteristics.push_back(extra);
    }
    if (subscripts.exist("hour") && boost::get<double>(subscripts.get_characteristic("hour").get_value()) != -1.0)  extra_characteristics.push_back(subscripts.get_characteristic("hour"));

    for (int node = 0; node < grafos.number_nodes("network"); node++)
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "branch")
        {
            characteristic extra;
            extra.set_name("name");
            extra.set_value(std::string("angular difference"));
            extra_characteristics.push_back(extra);
            std::vector<information> constraint = vertex.get_multi_info(extra_characteristics, "constraints", false);
            if (constraint.size() > 1 || constraint.size() == 0) return -1;

            extra.set_value(std::string("status"));
            std::vector<information> status = vertex.get_multi_info(std::vector<characteristic>({extra}), "parameters", false);
            if (status.size() > 1 || status.size() == 0) return -1;

            if (!status[0].exist("rows")) status[0].set_characteristic(characteristic("rows", constraint[0].get_characteristic("position matrix").get_value(), true), true);
        }
    }
}

int models::create_dc_opf_model()
{
    // Convert to per unit
    convert2per_unit();

    std::vector< std::vector<value_T> > periods(1, std::vector<value_T>());
    std::vector<double> hours (1, -1.0);

    information info = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("representative periods"), false)}),"model");
    if (info.get_characteristics().size() > 0 && boost::get<bool>(info.get_value()))
    {
        periods.clear();
        for (size_t node = 0; node < grafos.number_nodes("network"); node++)
        {
            std::vector<information> m_info = grafos.get("network", node, "vertex").get_all_info("parameters");
            for(const information& info : m_info)
            {
                for (const characteristic& cha: info.get_characteristics())
                {
                    if (cha.get_name() == "pt")
                    {
                        bool exist = false;
                        for (const std::vector<value_T>& rd : periods)
                        {
                            if (rd == cha.get_values())
                            {
                                exist = true;
                                break;
                            }
                        }
                        if (!exist) periods.push_back(cha.get_values());
                        break;
                    } 
                }
            }
            if (periods.size() > 0) break;
        }
        if (periods.size() == 0)
        {
            std::cout << "WARNING! The problem *balance tree* has been passed but the software could not find any representative days in the data" << std::endl;
            return -4;
        }
    }
    
    info = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("multiperiod"), false)}),"model");
    if (info.get_characteristics().size() > 0 && boost::get<bool>(info.get_value()))
    {
        hours.clear();
        for (size_t node = 0; node < grafos.number_nodes("network"); node++)
        {
            std::vector<information> m_info = grafos.get("network", node, "vertex").get_all_info("parameters");
            for(const information& info : m_info)
            {
                for (const characteristic& cha: info.get_characteristics())
                {
                    if (cha.get_name() == "hour")
                    {
                        bool exist = false;
                        for (const double rd : hours)
                        {
                            if (rd == boost::get<double>(cha.get_value()))
                            {
                                exist = true;
                                break;
                            }
                        }
                        if (!exist) hours.push_back(boost::get<double>(cha.get_value()));
                        break;
                    } 
                }
            }
            if (hours.size() > 0) break;
        }
        if (hours.size() == 0)
        {
            std::cout << "WARNING! The option *multiperiod* has been passed but the software could not find any hours in the data" << std::endl;
            return -5;
        }
    }

    std::vector<information> subscripts;
    for(std::vector<value_T>& period : periods)
    {
        for(double hour : hours)
        {
            information info;
            characteristic cha;
            cha.set_name("pt");
            cha.insert(period);
            info.set_characteristic(cha, true);
            cha.clear_values();
            cha.set_name("hour");
            cha.set_value(hour);
            info.set_characteristic(cha, false);
            info.set_value(true);
            subscripts.push_back(info);
        }
    }

    
    // Declaring variables OPF
    int code = declare_dc_opf_variables(subscripts);
    if (code != 0) return code;
    
    // Declaring constraints OPF
    code = declare_dc_opf_constraints(subscripts);
    if (code != 0) return code;


    // auto start = std::chrono::steady_clock::now();
    // Creating matrix of coefficients
    std::vector<std::future< std::pair< int, std::vector<information> > > > tasks;

    std::vector< std::vector<information> > matrix_coefficient_info;
    // for(std::vector<value_T>& period : periods)
    //     for(double hour : hours)
    //         tasks.push_back(std::async(std::launch::async, &models::create_dc_opf_matrix, this, std::ref(period), hour));

    // for(auto& task : tasks)
    // {
    //     int code = task.get().first;
    //     std::cout << code << std::endl;
    //     if (code != 0) return code;
    //     matrix_coefficient_info.push_back(task.get().second);
    // }
    
    std::pair<int, std::vector<information>> task = create_dc_opf_matrix(subscripts);
    if (task.first != 0) return task.first;
    
    matrix_coefficient_info.push_back(task.second);

    for (std::vector<information>& all_info : matrix_coefficient_info)
    {
        for (information& matrix_info : all_info)
        {
            std::vector<value_T> rows = matrix_info.get_characteristic("rows").get_values();
            std::vector<value_T> columns = matrix_info.get_characteristic("columns").get_values();
            std::vector<value_T> elements = matrix_info.get_characteristic("elements").get_values();
            for (size_t number = 0; number < rows.size(); number++)
                problem_matrix.add_value2matrix(boost::get<double>(elements[number]), boost::get<int>(rows[number]), boost::get<int>(columns[number]));
        }
    }
    // auto end = std::chrono::steady_clock::now();
    // std::cout << "Elapsed time in milliseconds: "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //     << " ms" << std::endl;
    return 0;
}

int models::recursive_balance_tree_search(information& info, std::vector<information>& storage)
{
    out_edge_iterator out, out_end;
    GraphType grafo = grafos.get_grafo("tree");
    int v_number = boost::get<int>(info.get_characteristic("vertex number").get_value());
    characteristic cha = info.get_characteristic("current position");
    value_T val = grafo[v_number].get_characteristic("name_node").get_value();
    cha.push_back(val);

    for (size_t pos = 0; pos < cha.get_values().size(); pos++)
        if(cha.get_values()[pos] != info.get_characteristic("final position").get_values()[pos])
            return 0;
    info.update_characteristic(cha);

    characteristic n_c = characteristic("name", boost::get<std::string>(info.get_characteristic("name characteristic").get_value()), false);
    std::string type_e = boost::get<std::string>(info.get_characteristic("type element").get_value());
    if (type_e == "parameters")
    {
        std::vector<information> m_info;
        if (info.exist("reference"))
        {
            characteristic reference = characteristic("reference", boost::get<std::string>(info.get_characteristic("reference").get_value()), false);
            m_info = grafo[v_number].get_multi_info(std::vector<characteristic>({n_c, reference}), type_e, false);
        }
        else
            m_info = grafo[v_number].get_multi_info(std::vector<characteristic>({n_c}), type_e, false);

        if (m_info.size() > 1 || m_info.size() == 0)
        {
            std::cout << "WARNING! Problem with recursive function for the balance tree" << std::endl;
            return -10;
        }
        m_info[0].set_characteristic(characteristic("vertex number", v_number, false), false);
        storage.push_back(m_info[0]);
    }
    else
    {
        std::vector<information> m_info;
        if (info.exist("reference"))
        {
            characteristic reference = characteristic("reference", boost::get<std::string>(info.get_characteristic("reference").get_value()), false);
            m_info = grafo[v_number].get_multi_info(std::vector<characteristic>({n_c, reference}), type_e, false);
        }
        else
            m_info = grafo[v_number].get_multi_info(std::vector<characteristic>({n_c}), type_e, false);
        
        if (m_info.size() == 0)
        {
            std::cout << "WARNING! Problem with recursive function for the balance tree" << std::endl;
            return -10;
        }
        else if (m_info.size() == 1)
        {
            m_info[0].set_characteristic(characteristic("vertex number", v_number, false), false);
            storage.push_back(m_info[0]);
        }
        else if (m_info.size() > 1)
        {
            for (information& m_i : m_info)
            {
                m_i.set_characteristic(characteristic("vertex number", v_number, false), false);
                storage.push_back(m_i);
            }
        }

    }

    if (cha.get_values() != info.get_characteristic("final position").get_values())
    {
        for (boost::tie(out, out_end)=boost::out_edges(v_number, grafo); out != out_end; ++out)
        {
            graph_data target = grafos.get("tree", int(boost::target(*out, grafo)), "vertex");
            info.update_characteristic(characteristic("vertex number", int(boost::target(*out, grafo)), false));
            int code = recursive_balance_tree_search(info, storage);
            if (code != 0) return code;
            if (info.get_characteristic("current position").get_values() == info.get_characteristic("final position").get_values()) return 0;
        }
    }
    return 0;
}

int models::declare_dc_opf_tree_links_constraints(const std::vector<value_T>& pt)
{
    std::vector<std::pair<characteristic, bool> > extra_characteristics;
    if (pt.size() > 0)
    {
        characteristic extra;
        extra.set_name("pt");
        extra.insert(pt);
        extra_characteristics.push_back(std::make_pair(extra, true));
    }
    for (information& info : data_parameters.get_all_parameters_type("connections"))
    {
        for (int node = 0; node < grafos.number_nodes("network"); node++)
        {
            graph_data vertex = grafos.get("network", node, "vertex");
            if (boost::get<std::string>(vertex.get_characteristic("ID").get_value()) == boost::get<std::string>(info.get_characteristic("ID").get_value()) )
            {
                create_problem_element("generation day aggregation", 0.0, 0.0, extra_characteristics, node, "network", "vertex", "constraints", 0.0);
                break;
            }
        }
    }
    return 0;
}

int models::update_bnds_obj(graph_data data, const std::string& name_graph, const std::string& component, const int position_in_graph, const std::string& name_info, const std::string& name_datainfo, const std::vector<characteristic>& extra_characteristics, bool silent, value_T new_max, value_T new_min, value_T new_cost, const bool clp_change)
{
    std::vector<characteristic> characteristics = data.get_characteristics();
    characteristics.push_back(characteristic("name", name_info, false));
    characteristics.insert(characteristics.end(), extra_characteristics.begin(), extra_characteristics.end());
    std::vector<information> info = data.get_multi_info(characteristics, name_datainfo, silent);
    if (info.size() > 1 || info.size() == 0)
    {
        if (!silent) std::cout << "Incoherent data for name *" << name_info << "* in data *" << name_datainfo << "*." << std::endl;
        return -11;
    }
    double min, max, cost;
    if(double *val = boost::get<double>(&new_min))
        min = *val;
    else min = -COIN_DBL_MAX;
    if(double *val = boost::get<double>(&new_max))
        max = *val;
    else max = COIN_DBL_MAX;
    if(double *val = boost::get<double>(&new_cost))
        cost = *val;
    else cost = COIN_DBL_MAX;
    
    information old_info = info[0];

    int position = boost::get<int>(info[0].get_characteristic("position matrix").get_value());
    if (info[0].get_characteristic("max").get_value() != new_max ||info[0].get_characteristic("min").get_value() != new_min)
    {
        info[0].update_characteristic(characteristic("max", new_max, false));
        info[0].update_characteristic(characteristic("min", new_min, false));
        if (clp_change && name_datainfo == "variables") solver.setColBounds(position, min, max);
        else if (clp_change && name_datainfo == "constraints") solver.setRowBounds(position, min, max);
    }
    if (name_datainfo == "variables" && info[0].get_characteristic("cost").get_value() != new_cost)
    {
        info[0].update_characteristic(characteristic("cost", new_cost, false));
        if (clp_change) solver.setObjCoeff(position, cost);
    }
    if (name_datainfo == "variables" && (old_info.get_characteristic("max").get_value() != new_max || old_info.get_characteristic("min").get_value() != new_min || old_info.get_characteristic("cost").get_value() != new_cost))
    {
        data.replace_information(old_info, info[0], name_datainfo);
        grafos.set(name_graph, position_in_graph, data, component);
        problem_matrix.update_variable(position, max, min, cost);
    }
    else if (name_datainfo == "constraints" && (old_info.get_characteristic("max").get_value() != new_max || old_info.get_characteristic("min").get_value() != new_min))
    {
        data.replace_information(old_info, info[0], name_datainfo);
        grafos.set(name_graph, position_in_graph, data, component);
        problem_matrix.update_constraint(position, max, min);
    }

    return 0;
}

int models::create_dc_opf_tree_links_matrix(const std::vector<value_T>& pt)
{
    std::vector<characteristic> characteristics;
    if (pt.size() > 0)
    {
        characteristic extra;
        extra.set_name("pt");
        extra.insert(pt);
        characteristics.push_back(extra);
    }

    std::pair<information, int> info_gen;
    int row = -1, column = -1;

    for (information& info : data_parameters.get_all_parameters_type("connections"))
    {
        std::vector<information> storage;
        information info_search;
        info_search.set_characteristic(characteristic("vertex number", 0, false), false);
        characteristic c_pos;
        c_pos.set_name("current position");
        info_search.set_characteristic(c_pos, false);
        characteristic f_pos;
        f_pos.set_name("final position");
        f_pos.insert(pt);
        info_search.set_characteristic(f_pos, false);
        info_search.set_characteristic(characteristic("name characteristic", std::string("output"), false), false);
        info_search.set_characteristic(characteristic("type element", std::string("variables"), false), false);
        info_search.set_characteristic(characteristic("reference", info.get_characteristic("ID").get_value(), false), false);
        int code = recursive_balance_tree_search(info_search, storage);

        int row, column;
        for (int node = 0; node < grafos.number_nodes("network"); node++)
        {
            graph_data vertex = grafos.get("network", node, "vertex");
            if (boost::get<std::string>(vertex.get_characteristic("ID").get_value()) == boost::get<std::string>(info.get_characteristic("ID").get_value()) )
            {
                // constraint
                info_gen = get_information(vertex, "generation day aggregation", characteristics, "constraints", false);
                if (info_gen.second != 0) return -9;
                row = boost::get<int>(info_gen.first.get_characteristic("position matrix").get_value());
                problem_matrix.add_active("row", row);
                // output tree variable
                for (information& st: storage)
                {
                    if (st.get_characteristic("name_node").get_value() == pt.back())
                    {
                        column = boost::get<int>(st.get_characteristic("position matrix").get_value());
                        int vertex_number = boost::get<int>(st.get_characteristic("vertex number").get_value());
                        std::string name_variable = boost::get<std::string>(st.get_characteristic("name").get_value());
                        std::vector<characteristic> extra_cha({characteristic("reference", info.get_characteristic("ID").get_value(), false)});
                        int code = update_bnds_obj(grafos.get("tree", vertex_number, "vertex"), "tree", "vertex", vertex_number, name_variable, "variables", extra_cha, false, std::string("infinite"), 0.0, 0.0, false);
                        if (code != 0) return code;
                        break;
                    }
                }
                problem_matrix.add_value2matrix(1.0, row, column);
                // active power generation
                std::vector<characteristic> extra_characteristics = characteristics;
                extra_characteristics.push_back(characteristic("name", std::string("active power generation"), false));
                std::vector<information> active_gen = vertex.get_multi_info(extra_characteristics, "variables", false);
                for (information& st: active_gen)
                {
                    column = boost::get<int>(st.get_characteristic("position matrix").get_value());
                    problem_matrix.add_value2matrix(-1.0, row, column);
                }
                break;
            }
        }
    }
}

int models::update_cost_variables_constraints_dc_opf_tree_links(const std::vector<value_T>& pt)
{
    std::vector<characteristic> characteristics;
    if (pt.size() > 0)
    {
        characteristic extra;
        extra.set_name("pt");
        extra.insert(pt);
        characteristics.push_back(extra);
    }
    int row = -1, column = -1;
    double new_cost;

    std::vector<information> storage;
    information info_search;
    info_search.set_characteristic(characteristic("vertex number", 0, false), false);
    characteristic c_pos;
    c_pos.set_name("current position");
    info_search.set_characteristic(c_pos, false);
    characteristic f_pos;
    f_pos.set_name("final position");
    f_pos.insert(pt);
    info_search.set_characteristic(f_pos, false);
    info_search.set_characteristic(characteristic("name characteristic", std::string("weight"), false), false);
    info_search.set_characteristic(characteristic("type element", std::string("parameters"), false), false);
    int code = recursive_balance_tree_search(info_search, storage);

    double accumulated_weight = 1.0;
    for (information& st: storage)
        accumulated_weight *= boost::get<double>(st.get_value());

    // Updating deficit cost
    graph_data vertex_tree = grafos.get("tree", 0, "vertex");
    std::vector<characteristic> extra_characteristics_tree;
    extra_characteristics_tree.push_back(characteristic("name", std::string("deficit"), false));
    std::vector<information> deficit = vertex_tree.get_multi_info(extra_characteristics_tree, "variables", false);
    for (information& de: deficit)
    {
        std::vector<characteristic> extra_cha;
        extra_cha.push_back(characteristic("reference", de.get_characteristic("reference").get_value(), false));
        column = boost::get<int>(de.get_characteristic("position matrix").get_value());
        value_T min = de.get_characteristic("min").get_value();
        value_T max = de.get_characteristic("max").get_value();
        value_T old_cost = de.get_characteristic("cost").get_value();
        new_cost = boost::get<double>(old_cost) * accumulated_weight * 10;
        int code = update_bnds_obj(vertex_tree, "tree", "vertex", 0, "deficit", "variables", extra_cha, false, max, min, new_cost, false);
        vertex_tree = grafos.get("tree", 0, "vertex");
        if (code != 0) return code;
    }

    for (int node = 0; node < grafos.number_nodes("network"); node++)
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "bus")
        {
            // Adding load curtailment
            std::vector<characteristic> extra_characteristics = characteristics;
            extra_characteristics.push_back(characteristic("name", std::string("load curtailment"), false));
            std::vector<information> active_load_curtailment = vertex.get_multi_info(extra_characteristics, "variables", false);
            for (information& alc: active_load_curtailment)
            {
                std::vector<characteristic> extra_cha = characteristics;
                extra_cha.push_back(characteristic("hour", alc.get_characteristic("hour").get_value(), false));
                column = boost::get<int>(alc.get_characteristic("position matrix").get_value());
                value_T min = alc.get_characteristic("min").get_value();
                value_T max = alc.get_characteristic("max").get_value();
                double old_cost = boost::get<double>(alc.get_characteristic("cost").get_value());
                new_cost = old_cost * accumulated_weight;
                int code = update_bnds_obj(vertex, "network", "vertex", node, "load curtailment","variables", extra_cha, false, max, min, new_cost, false);
                vertex = grafos.get("network", node, "vertex");
                if (code != 0) return code;
            }
            
            // Adding generation curtailment
            std::vector<characteristic> extra_characteristics1 = characteristics;
            extra_characteristics1.push_back(characteristic("name", std::string("generation curtailment"), false));
            std::vector<information> active_generation_curtailment = vertex.get_multi_info(extra_characteristics1, "variables", false);
            for (information& agc: active_generation_curtailment)
            {
                std::vector<characteristic> extra_cha = characteristics;
                extra_cha.push_back(characteristic("hour", agc.get_characteristic("hour").get_value(), false));
                column = boost::get<int>(agc.get_characteristic("position matrix").get_value());
                value_T min = agc.get_characteristic("min").get_value();
                value_T max = agc.get_characteristic("max").get_value();
                double old_cost = boost::get<double>(agc.get_characteristic("cost").get_value());
                new_cost = old_cost * accumulated_weight;
                int code = update_bnds_obj(vertex, "network", "vertex", node, "generation curtailment","variables", extra_cha, false, max, min, new_cost, false);
                vertex = grafos.get("network", node, "vertex");
                if (code != 0) return code;
            }
        }
        if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "generator")
        {
            std::vector<characteristic> extra_characteristics = characteristics;
            extra_characteristics.push_back(characteristic("name", std::string("active power generation"), false));
            std::vector<information> active_gen = vertex.get_multi_info(extra_characteristics, "variables", false);
            for (information& st: active_gen)
            {
                std::vector<characteristic> extra_cha = characteristics;
                extra_cha.push_back(characteristic("hour", st.get_characteristic("hour").get_value(), false));
                value_T min = st.get_characteristic("min").get_value();
                value_T max = st.get_characteristic("max").get_value();
                double old_cost = boost::get<double>(st.get_characteristic("cost").get_value());
                new_cost = old_cost * accumulated_weight;
                int code = update_bnds_obj(vertex, "network", "vertex", node, "active power generation", "variables", extra_cha, false, max, min, new_cost, false);
                vertex = grafos.get("network", node, "vertex");
                if (code != 0) return code;
            }
            // Extracting cost for variable generation
            extra_characteristics.pop_back();
            extra_characteristics.push_back(characteristic("name", std::string("piecewise generation cost"), false));
            std::vector<information> pw_gen_cost = vertex.get_multi_info(extra_characteristics, "constraints", true);
            value_T new_max, new_min;
            for (information& pwgc: pw_gen_cost)
            {
                extra_characteristics.pop_back();
                extra_characteristics.push_back(characteristic("hour", pwgc.get_characteristic("hour").get_value(), false));
                extra_characteristics.push_back(characteristic("piece", pwgc.get_characteristic("piece").get_value(), false));
                row = boost::get<int>(pwgc.get_characteristic("position matrix").get_value());

                value_T min = pwgc.get_characteristic("min").get_value();
                if (double *val = boost::get<double>(&min))
                    new_min = value_T(*val * accumulated_weight);
                else new_min = min;
                value_T max = pwgc.get_characteristic("max").get_value();
                if (double *val = boost::get<double>(&max))
                    new_max = value_T(*val * accumulated_weight);
                else new_max = max;
                int code = update_bnds_obj(vertex, "network", "vertex", node, "piecewise generation cost", "constraints", extra_characteristics, false, new_max , new_min, 0.0, false);
                vertex = grafos.get("network", node, "vertex");
                if (code != 0) return code;                

                // extracting variable
                extra_characteristics.pop_back();
                std::pair<information, int> info_costf = get_information(vertex, "active power generation", extra_characteristics, "variables", false);
                if (info_costf.second != 0) return -6;
                column = boost::get<int>(info_costf.first.get_characteristic("position matrix").get_value());
                std::pair<double, int> old_coefficient = problem_matrix.get_coefficient(row, column);
                if (old_coefficient.second != 0) return old_coefficient.second;
                problem_matrix.modify_coefficient(row, column, old_coefficient.first * accumulated_weight);
            }
        }
    }
    return 0;
}

int models::create_dc_opf_tree_links()
{
    std::vector< std::vector<value_T> > periods;

    for (size_t node = 0; node < grafos.number_nodes("network"); node++)
    {
        std::vector<information> m_info = grafos.get("network", node, "vertex").get_all_info("parameters");
        for(const information& info : m_info)
        {
            for (const characteristic& cha: info.get_characteristics())
            {
                if (cha.get_name() == "pt")
                {
                    bool exist = false;
                    for (const std::vector<value_T>& rd : periods)
                    {
                        if (rd == cha.get_values())
                        {
                            exist = true;
                            break;
                        }
                    }
                    if (!exist) periods.push_back(cha.get_values());
                    break;
                } 
            }
        }
        if (periods.size() > 0) break;
    }
    if (periods.size() == 0)
    {
        std::cout << "WARNING! The problem *balance tree* has been passed but the software could not find any representative days in the data" << std::endl;
        return -10;
    }
    
    // Declaring constraints DC OPF and balance tree links
    for(const std::vector<value_T>& period : periods)
    {
        int code = declare_dc_opf_tree_links_constraints(period);
        if (code != 0) return code;
    }

    // creating matrix for DC OPF and balance tree links
    for(const std::vector<value_T>& period : periods)
    {
        int code = create_dc_opf_tree_links_matrix(period);
        if (code != 0) return code;
    }

    for(const std::vector<value_T>& period : periods)
    {
        int code = update_cost_variables_constraints_dc_opf_tree_links(period);
        if (code != 0) return code;
    }

    return 0;
}

void models::initialise()
{
    information BT = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("BT"), false), characteristic("engine", std::string("pyene"), false)}),"model", true);
    if (BT.get_characteristics().size() > 0 && boost::get<bool>(BT.get_value()))
    {
        create_nodes_tree();
        create_edges_tree();
        if (create_balance_tree_model() != 0) return;
    }
    
    information DC_OPF = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("DC OPF"), false), characteristic("engine", std::string("pyene"), false)}), "model", true);
    if (DC_OPF.get_characteristics().size() > 0 && boost::get<bool>(DC_OPF.get_value()))
    {
        create_nodes_network();
        create_edges_network();
        if (create_dc_opf_model() != 0) return;
    }
    
    if (DC_OPF.get_characteristics().size() > 0 && boost::get<bool>(DC_OPF.get_value()) && BT.get_characteristics().size() > 0 && boost::get<bool>(BT.get_value()) && create_dc_opf_tree_links() != 0)
        return;
    
    number_times_optimisation = 0;

    solver.load_model(problem_matrix);
        
    std::vector<information> MOEA = data_parameters.get_multi_parameter_type(std::vector<characteristic>({characteristic("name", std::string("MOEA"), false)}), "model", true);
    if (MOEA.size() > 1 || MOEA.size() == 0) return;
    if (boost::get<bool>(MOEA[0].get_value()) && create_moea_problem() != 0)
        return;    
}

void models::store_solution()
{
    std::vector<double> sol = solver.get_solution();
    for (int node = 0; node < grafos.number_nodes("tree"); node++)
    {
        graph_data vertex = grafos.get("tree", node, "vertex");
        for (information& info: vertex.get_all_info("variables"))
        {
            info.set_value(sol[boost::get<int>(info.get_characteristic("position matrix").get_value())]);
            vertex.update_value(info, "variables");
        }
        grafos.set("tree", node, vertex, "vertex");
    }
    for (int edge = 0; edge < grafos.number_edges("tree"); edge++)
    {
        graph_data edge_data = grafos.get("tree", edge, "edge");
        for (information& info: edge_data.get_all_info("variables"))
        {
            info.set_value(sol[boost::get<int>(info.get_characteristic("position matrix").get_value())]);
            edge_data.update_value(info, "variables");
        }
        grafos.set("tree", edge, edge_data, "edge");
    }
    // network
    information load_cur;
    information gen_cur;
    information angle;

    angle.set_characteristic(characteristic("name", std::string("voltage angle"), false), false);
    load_cur.set_characteristic(characteristic("name", std::string("generation curtailment"), false), false);
    gen_cur.set_characteristic(characteristic("name", std::string("load curtailment"), false), false);

    for (auto& cha : last_subscript.get_characteristics())
    {
        angle.set_characteristic(cha, false);
        load_cur.set_characteristic(cha, false);
        gen_cur.set_characteristic(cha, false);
    }

    for (auto &node : grafos.get_positions("network", "bus"))
    {
        information info = grafos.find_component_information("network", node, "vertex", "variables", angle);
        double value = sol[boost::get<int>(info.get_characteristic("position matrix").get_value())];
        grafos.update_information_value("network", node, "vertex", "variables", angle, value);

        info = grafos.find_component_information("network", node, "vertex", "variables", load_cur);
        value = sol[boost::get<int>(info.get_characteristic("position matrix").get_value())];
        grafos.update_information_value("network", node, "vertex", "variables", load_cur, value);

        info = grafos.find_component_information("network", node, "vertex", "variables", gen_cur);
        value = sol[boost::get<int>(info.get_characteristic("position matrix").get_value())];
        grafos.update_information_value("network", node, "vertex", "variables", gen_cur, value);
    }

    information active_power_generation;
    information active_power_generation_cost;

    active_power_generation.set_characteristic(characteristic("name", std::string("active power generation"), false), false);
    active_power_generation_cost.set_characteristic(characteristic("name", std::string("active power generation cost"), false), false);

    for (auto& cha : last_subscript.get_characteristics())
    {
        active_power_generation.set_characteristic(cha, false);
        active_power_generation_cost.set_characteristic(cha, false);
    }
    
    for (auto &node : grafos.get_positions("network", "generator"))
    {
        information info = grafos.find_component_information("network", node, "vertex", "variables", active_power_generation);
        double value = sol[boost::get<int>(info.get_characteristic("position matrix").get_value())];
        grafos.update_information_value("network", node, "vertex", "variables", active_power_generation, value);

        info = grafos.find_component_information("network", node, "vertex", "variables", active_power_generation_cost);
        if (info.get_characteristics().size() > 0)
        {
            value = sol[boost::get<int>(info.get_characteristic("position matrix").get_value())];
            grafos.update_information_value("network", node, "vertex", "variables", active_power_generation_cost, value);
        }
    }
}

void models::evaluate()
{
    if (candidate.get_characteristics().size() == 0)
    {
        characteristic cha;
        cha.set_name("pt");
        cha.insert(std::vector<value_T>());
        candidate.set_characteristic(cha, true);
        cha.clear_values();
        cha.set_name("hour");
        cha.set_value(-1.0);
        candidate.set_characteristic(cha, false);
        candidate.set_value(true);
    }
    last_subscript = candidate;
    if (number_times_optimisation == 0) solver.solve("initial solve", candidate);
    else solver.solve("dual", candidate);
    store_solution();
    number_times_optimisation++;
}

void models::accumulate_unique_characteristics_outputs(std::vector<information>& information_required, const information& output, const std::string& name)
{
    std::vector<information> copy_information;
    if (output.exist(name))
    {
        for (value_T& val: output.get_characteristic(name).get_values())
        {
            if (information_required.size() == 0)
            {
                if (boost::get<std::string>(val) != "all")
                {
                    characteristic cha;
                    cha.set_name(name);
                    cha.set_value(val);
                    information info;
                    info.set_characteristic(cha, false);
                    information_required.push_back(info);
                }
            }
            else
            {
                copy_information = information_required;
                if (boost::get<std::string>(val) != "all")
                {
                    information_required.clear();
                    for (const information& cinfo : copy_information)
                    {
                        characteristic cha;
                        cha.set_name(name);
                        cha.set_value(val);
                        information info = cinfo;
                        info.set_characteristic(cha, false);
                        information_required.push_back(info);
                    }
                }
            }
        }
    }
}

void models::return_outputs(std::vector<double> &values, std::vector<int> &starts, std::vector<std::vector<std::vector<std::string>>> &characteristics)
{
    std::vector<double> values_return;
    std::vector<int> starts_return;
    std::vector<std::vector<std::vector<std::string>>> characteristics_return;
    for (information output : data_parameters.get_all_parameters_type("outputs"))
    {
        if (boost::get<double>(last_subscript.get_characteristic("hour").get_value()) != -1.0)
            for (auto &cha : last_subscript.get_characteristics())
                output.set_characteristic(cha, false);
        
        information DC_OPF = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("DC OPF"), false), characteristic("engine", std::string("pyene"), false)}), "model");
        if (boost::get<std::string>(output.get_characteristic("problem").get_values()[0]) == "DC OPF" && DC_OPF.get_characteristics().size() > 0 && boost::get<bool>(DC_OPF.get_value()))
        {
            std::vector<information> information_required;
            std::vector<std::vector<value_T>> periods;
            bool is_final_position_tree;
            information info = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("BT"), false), characteristic("engine", std::string("pyene"), false)}), "model", true);
            if (info.get_characteristics().size() > 0 && boost::get<bool>(info.get_value()))
            {
                is_final_position_tree = false;
                for (size_t node = 0; node < grafos.number_nodes("network"); node++)
                {
                    std::vector<information> m_info = grafos.get("network", node, "vertex").get_all_info("parameters");
                    for (const information &info : m_info)
                    {
                        for (const characteristic &cha : info.get_characteristics())
                        {
                            if (cha.get_name() == "pt")
                            {
                                bool exist = false;
                                for (const std::vector<value_T> &rd : periods)
                                {
                                    if (rd == cha.get_values())
                                    {
                                        exist = true;
                                        break;
                                    }
                                }
                                if (!exist)
                                    periods.push_back(cha.get_values());
                                break;
                            }
                        }
                    }
                    if (periods.size() > 0)
                        break;
                }
                for (std::vector<value_T> &period : periods)
                {
                    if (period == output.get_characteristic("pt").get_values())
                    {
                        characteristic cha;
                        cha.set_name("pt");
                        cha.insert(output.get_characteristic("pt").get_values());
                        information info;
                        info.set_characteristic(cha, false);
                        information_required.push_back(info);
                        is_final_position_tree = true;
                        break;
                    }
                }
            }
            else
                is_final_position_tree = true;
            info = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("multiperiod"), false)}), "model");
            if (info.get_characteristics().size() > 0 && boost::get<bool>(info.get_value()))
            {
                characteristic cha;
                cha.set_name("hour");
                cha.set_value(output.get_characteristic("hour").get_value());
                information info;
                info.set_characteristic(cha, false);
                information_required.push_back(info);
                accumulate_unique_characteristics_outputs(information_required, output, "hour");
            }
            accumulate_unique_characteristics_outputs(information_required, output, "ID");
            accumulate_unique_characteristics_outputs(information_required, output, "type");
            accumulate_unique_characteristics_outputs(information_required, output, "group");
            accumulate_unique_characteristics_outputs(information_required, output, "zone");
            for (information &info_req : information_required)
                info_req.set_characteristic(characteristic("name", output.get_characteristic("name").get_values()[0], false), false);
            std::vector<information> raw_info;
            for (const information &info_req : information_required)
            {
                for (int node = 0; node < grafos.number_nodes("network"); node++)
                {
                    if (grafos.get("network", node, "vertex").get_characteristic("type").get_value() == info_req.get_characteristic("type").get_value())
                    {
                        if (boost::get<std::string>(output.get_characteristic("type information").get_values()[0]) == "parameters")
                        {
                            std::vector<information> info = grafos.get("network", node, "vertex").get_multi_info(info_req.get_characteristics(), boost::get<std::string>(output.get_characteristic("type information").get_values()[0]), false);
                            raw_info.insert(raw_info.end(), info.begin(), info.end());
                        }
                        else
                        {
                            information set;
                            set.set_characteristic(info_req.get_characteristic("name"), false);
                            for (auto &cha : last_subscript.get_characteristics())
                                set.set_characteristic(cha, false);
                            information info = grafos.find_component_information("network", node, "vertex", "variables", set);
                            raw_info.push_back(info);
                        }
                    }
                }
            }
            for (information &info : raw_info)
            {
                info.erase_characteristic("position matrix");
                info.erase_characteristic("cost");
                info.erase_characteristic("min");
                info.erase_characteristic("max");
            }
            std::vector<information> refined_info;
            if (output.exist("type") && output.exist("subtype"))
            {
                for (const value_T &val : output.get_characteristic("subtype").get_values())
                {
                    for (const information &r_info : raw_info)
                    {
                        std::vector<value_T> subtypes = r_info.get_characteristic("subtype").get_values();
                        std::vector<value_T>::iterator subtype_it = std::find(subtypes.begin(), subtypes.end(), val);
                        if (subtype_it != subtypes.end())
                        {
                            bool equal;
                            if (refined_info.size() > 0)
                                equal = true;
                            else
                                equal = false;
                            for (const information &ref_info : refined_info)
                            {
                                for (characteristic &cha : r_info.get_characteristics())
                                {
                                    bool found_equal = false;
                                    for (characteristic &param_cha : ref_info.get_characteristics())
                                    {
                                        if (cha.get_name() == param_cha.get_name() && cha.get_value() == param_cha.get_value() && cha.get_values() == param_cha.get_values())
                                        {
                                            found_equal = true;
                                            break;
                                        }
                                    }
                                    if (!found_equal)
                                    {
                                        equal = false;
                                        break;
                                    }
                                }
                            }
                            if (!equal)
                                refined_info.push_back(r_info);
                        }
                    }
                }
            }
            else
                refined_info = raw_info;
            if (!is_final_position_tree)
            {
                raw_info = refined_info;
                std::vector<std::vector<value_T>> related_periods;
                for (std::vector<value_T> &period : periods)
                {
                    bool is_related = true;
                    for (value_T &val : output.get_characteristic("pt").get_values())
                    {
                        if (std::find(period.begin(), period.end(), val) == period.end())
                        {
                            is_related = false;
                            break;
                        }
                    }
                    if (is_related)
                        related_periods.push_back(period);
                }
                // Searching for initial node
                std::vector<information> storage;
                information info_search;
                info_search.set_characteristic(characteristic("vertex number", 0, false), false);
                characteristic c_pos;
                c_pos.set_name("current position");
                info_search.set_characteristic(c_pos, false);
                characteristic f_pos;
                f_pos.set_name("final position");
                f_pos.insert(output.get_characteristic("pt").get_values());
                info_search.set_characteristic(f_pos, false);
                info_search.set_characteristic(characteristic("name characteristic", std::string("weight"), false), false);
                info_search.set_characteristic(characteristic("type element", std::string("parameters"), false), false);
                int code = recursive_balance_tree_search(info_search, storage);
                characteristic initial_vertex;
                int v_number = -1;
                for (information &sto : storage)
                {
                    if (sto.get_characteristic("name_node").get_value() == output.get_characteristic("pt").get_values().back())
                    {
                        v_number = boost::get<int>(sto.get_characteristic("vertex number").get_value());
                        break;
                    }
                }
                // Searching for final nodes in tree and calculating accumulated weight
                std::vector<double> accumulated_weight_periods;
                for (const std::vector<value_T> &r_per : related_periods)
                {
                    info_search.update_characteristic(characteristic("vertex number", v_number, false));
                    info_search.update_characteristic(c_pos);
                    storage.clear();
                    f_pos.clear_values();
                    f_pos.insert(r_per);
                    info_search.update_characteristic(f_pos);
                    int code = recursive_balance_tree_search(info_search, storage);
                    double accumulated_weight = 1.0;
                    for (information &st : storage)
                        accumulated_weight *= boost::get<double>(st.get_value());
                    accumulated_weight_periods.push_back(accumulated_weight);
                }
                // Getting IDs
                std::vector<value_T> IDs;
                for (const information &info : raw_info)
                {
                    if (IDs.size() > 0 && std::find(IDs.begin(), IDs.end(), info.get_characteristic("ID").get_value()) == IDs.end())
                        IDs.push_back(info.get_characteristic("ID").get_value());
                    else if (IDs.size() == 0)
                        IDs.push_back(info.get_characteristic("ID").get_value());
                }
                refined_info.clear();
                int counter = 0;
                for (const std::vector<value_T> &r_per : related_periods)
                {
                    if (refined_info.size() == 0)
                    {
                        for (const information &info : raw_info)
                        {
                            if (info.get_characteristic("pt").get_values() == r_per)
                            {
                                information new_info = info;
                                new_info.update_characteristic(output.get_characteristic("pt"));
                                new_info.set_value(boost::get<double>(info.get_value()) * accumulated_weight_periods[counter]);
                                refined_info.push_back(new_info);
                            }
                        }
                    }
                    else
                    {
                        for (const information &info : raw_info)
                        {
                            if (info.get_characteristic("pt").get_values() == r_per)
                            {
                                for (information &ref_info : refined_info)
                                {
                                    bool equal = true;
                                    for (characteristic &cha : info.get_characteristics())
                                    {
                                        bool different = false;
                                        for (characteristic &param_cha : ref_info.get_characteristics())
                                        {
                                            if (cha.get_name() != "pt" && cha.get_name() == param_cha.get_name() && (cha.get_value() != param_cha.get_value() || cha.get_values() != param_cha.get_values()))
                                            {
                                                different = true;
                                                break;
                                            }
                                        }
                                        if (different)
                                        {
                                            equal = false;
                                            break;
                                        }
                                    }
                                    if (equal)
                                    {
                                        double new_val = boost::get<double>(info.get_value()) * accumulated_weight_periods[counter] + boost::get<double>(ref_info.get_value());
                                        ref_info.set_value(new_val);
                                    }
                                }
                            }
                        }
                    }
                    counter++;
                }
            }
            if (boost::get<std::string>(output.get_characteristic("function").get_values()[0]) == "none")
            {
                int counter = 0;
                for (information &info : refined_info)
                {
                    std::vector<std::vector<std::string>> storage_cha;
                    values_return.push_back(boost::get<double>(info.get_value()));
                    starts_return.push_back(counter);
                    if (output.get_characteristic("problem").get_values().size() > 0)
                    {
                        std::vector<std::string> aux_cha;
                        aux_cha.push_back("problem");
                        aux_cha.push_back("string");
                        for (value_T &val : output.get_characteristic("problem").get_values())
                            aux_cha.push_back(boost::get<std::string>(val));
                        storage_cha.push_back(aux_cha);
                    }
                    for (const characteristic &cha : info.get_characteristics())
                    {
                        if (cha.get_values().size() > 0)
                        {
                            std::vector<std::string> aux_cha;
                            aux_cha.push_back(cha.get_name());
                            aux_cha.push_back("string");
                            for (value_T &val : cha.get_values())
                                aux_cha.push_back(boost::get<std::string>(val));
                            storage_cha.push_back(aux_cha);
                        }
                        else
                        {
                            std::vector<std::string> aux_cha;
                            aux_cha.push_back(cha.get_name());
                            if (std::string *val = boost::get<std::string>(&cha.get_value()))
                            {
                                aux_cha.push_back("string");
                                std::string v = boost::get<std::string>(cha.get_value());
                                aux_cha.push_back(v);
                            }
                            else if (bool *val = boost::get<bool>(&cha.get_value()))
                            {
                                aux_cha.push_back("bool");
                                if (*val)
                                    aux_cha.push_back("True");
                                else
                                    aux_cha.push_back("False");
                            }
                            else if (int *val = boost::get<int>(&cha.get_value()))
                            {
                                aux_cha.push_back("integer");
                                aux_cha.push_back(std::to_string(*val));
                            }
                            else if (double *val = boost::get<double>(&cha.get_value()))
                            {
                                aux_cha.push_back("double");
                                aux_cha.push_back(std::to_string(*val));
                            }
                            storage_cha.push_back(aux_cha);
                        }
                    }
                    characteristics_return.push_back(storage_cha);
                    counter++;
                }
            }
            else if (boost::get<std::string>(output.get_characteristic("function").get_values()[0]) == "sum")
            {
                double total = 0;
                for (information &info : refined_info)
                    total += boost::get<double>(info.get_value());
                values_return.push_back(total);
                starts_return.push_back(0);
                std::vector<std::vector<std::string>> storage_cha;
                for (const characteristic &cha : output.get_characteristics())
                {
                    if (cha.get_name() != "data device" && cha.get_name() != "type information" && cha.get_name() != "function")
                    {
                        std::vector<std::string> aux_cha;
                        aux_cha.push_back(cha.get_name());
                        aux_cha.push_back("string");
                        for (value_T &val : cha.get_values())
                            aux_cha.push_back(boost::get<std::string>(val));
                        storage_cha.push_back(aux_cha);
                    }
                }
                characteristics_return.push_back(storage_cha);
            }
        }
        information BT = data_parameters.get_parameter_type(std::vector<characteristic>({characteristic("name", std::string("BT"), false), characteristic("engine", std::string("pyene"), false)}), "model", true);
        if (boost::get<std::string>(output.get_characteristic("problem").get_values()[0]) == "BT" && BT.get_characteristics().size() > 0 && boost::get<bool>(BT.get_value()))
        {
            // Searching for initial node
            std::vector<information> storage;
            information info_search;
            info_search.set_characteristic(characteristic("vertex number", 0, false), false);
            characteristic c_pos;
            c_pos.set_name("current position");
            info_search.set_characteristic(c_pos, false);
            characteristic f_pos;
            f_pos.set_name("final position");
            f_pos.insert(output.get_characteristic("pt").get_values());
            info_search.set_characteristic(f_pos, false);
            info_search.set_characteristic(characteristic("name characteristic", output.get_characteristic("name").get_values()[0], false), false);
            info_search.set_characteristic(characteristic("type element", output.get_characteristic("type information").get_values()[0], false), false);
            int code = recursive_balance_tree_search(info_search, storage);
            std::vector<information> refined_info;
            for (information &sto : storage)
                if (sto.get_characteristic("name_node").get_value() == output.get_characteristic("pt").get_values().back())
                    refined_info.push_back(sto);
            for (information &info : refined_info)
            {
                info.erase_characteristic("position matrix");
                info.erase_characteristic("cost");
                info.erase_characteristic("min");
                info.erase_characteristic("max");
                info.erase_characteristic("vertex number");
            }
            if (boost::get<std::string>(output.get_characteristic("function").get_values()[0]) == "none")
            {
                int counter = 0;
                for (information &info : refined_info)
                {
                    std::vector<std::vector<std::string>> storage_cha;
                    values_return.push_back(boost::get<double>(info.get_value()));
                    starts_return.push_back(counter);
                    if (output.get_characteristic("problem").get_values().size() > 0)
                    {
                        std::vector<std::string> aux_cha;
                        aux_cha.push_back("problem");
                        aux_cha.push_back("string");
                        for (value_T &val : output.get_characteristic("problem").get_values())
                            aux_cha.push_back(boost::get<std::string>(val));
                        storage_cha.push_back(aux_cha);
                    }
                    for (const characteristic &cha : info.get_characteristics())
                    {
                        if (cha.get_values().size() > 0)
                        {
                            std::vector<std::string> aux_cha;
                            aux_cha.push_back(cha.get_name());
                            aux_cha.push_back("string");
                            for (value_T &val : cha.get_values())
                                aux_cha.push_back(boost::get<std::string>(val));
                            storage_cha.push_back(aux_cha);
                        }
                        else
                        {
                            std::vector<std::string> aux_cha;
                            aux_cha.push_back(cha.get_name());
                            if (std::string *val = boost::get<std::string>(&cha.get_value()))
                            {
                                aux_cha.push_back("string");
                                std::string v = boost::get<std::string>(cha.get_value());
                                aux_cha.push_back(v);
                            }
                            else if (bool *val = boost::get<bool>(&cha.get_value()))
                            {
                                aux_cha.push_back("bool");
                                if (*val)
                                    aux_cha.push_back("True");
                                else
                                    aux_cha.push_back("False");
                            }
                            else if (int *val = boost::get<int>(&cha.get_value()))
                            {
                                aux_cha.push_back("integer");
                                aux_cha.push_back(std::to_string(*val));
                            }
                            else if (double *val = boost::get<double>(&cha.get_value()))
                            {
                                aux_cha.push_back("double");
                                aux_cha.push_back(std::to_string(*val));
                            }
                            storage_cha.push_back(aux_cha);
                        }
                    }
                    characteristics_return.push_back(storage_cha);
                    counter++;
                }
            }
            else if (boost::get<std::string>(output.get_characteristic("function").get_values()[0]) == "sum")
            {
                double total = 0;
                for (information &info : refined_info)
                    total += boost::get<double>(info.get_value());
                values_return.push_back(total);
                starts_return.push_back(0);
                std::vector<std::vector<std::string>> storage_cha;
                for (const characteristic &cha : output.get_characteristics())
                {
                    if (cha.get_name() != "data device" && cha.get_name() != "type information" && cha.get_name() != "function")
                    {
                        std::vector<std::string> aux_cha;
                        aux_cha.push_back(cha.get_name());
                        aux_cha.push_back("string");
                        for (value_T &val : cha.get_values())
                            aux_cha.push_back(boost::get<std::string>(val));
                        storage_cha.push_back(aux_cha);
                    }
                }
                characteristics_return.push_back(storage_cha);
            }
        }
    }
    values = values_return;
    starts = starts_return;
    characteristics = characteristics_return;
}

void models::convert2pu(information& candidate)
{
    double Sbase = boost::get<double>(data_parameters.get_parameter_type(std::vector<characteristic>{characteristic("name", std::string("Sbase"), false)}, "model").get_value());
    candidate.set_value(boost::get<double>(candidate.get_value()) / Sbase);
}

int models::update_parameter()
{
    if (boost::get<std::string>(candidate.get_characteristic("problem").get_value()) == "BT")
    {
        if (boost::get<std::string>(candidate.get_characteristic("resource").get_value()) == "energy")
        {
            double Sbase = boost::get<double>(data_parameters.get_parameter_type(std::vector<characteristic>{characteristic("name", std::string("Sbase"), false)}, "model").get_value());
            candidate.set_value(boost::get<double>(candidate.get_value()) / Sbase);
        }
        std::vector<information> storage;
        information info_search;
        info_search.set_characteristic(characteristic("vertex number", 0, false), false);
        characteristic c_pos;
        c_pos.set_name("current position");
        info_search.set_characteristic(c_pos, false);
        characteristic f_pos;
        f_pos.set_name("final position");
        f_pos.insert(candidate.get_characteristic("pt").get_values());
        info_search.set_characteristic(f_pos, false);
        info_search.set_characteristic(characteristic("name characteristic", candidate.get_characteristic("name").get_value(), false), false);
        info_search.set_characteristic(characteristic("type element", std::string("parameters"), false), false);
        if (candidate.exist("reference")) info_search.set_characteristic(candidate.get_characteristic("reference"), false);
        int code = recursive_balance_tree_search(info_search, storage);

        int vertex_number = -1;
        for (information& st : storage)
            if (st.get_characteristic("name_node").get_value() == candidate.get_characteristic("name_node").get_value())
                vertex_number = boost::get<int>(st.get_characteristic("vertex number").get_value());
        if (vertex_number == -1)
        {
            std::cout << "Incoherent data for name *" << candidate.get_characteristic("name").get_value() << "* in data *parameters*." << std::endl;
            return -1;
        }

        graph_data vertex = grafos.get("tree", vertex_number, "vertex");
        std::vector<characteristic> charac;
        charac.push_back(candidate.get_characteristic("name"));
        if (candidate.exist("reference")) charac.push_back(candidate.get_characteristic("reference"));
        std::vector<information> m_info = vertex.get_multi_info(charac, "parameters", true);
        if (m_info.size() > 1 || m_info.size() == 0)
        {
            std::cout << "Incoherent data for name *" << candidate.get_characteristic("name").get_value() << "* in data *parameters*." << std::endl;
            return -1;
        }
        information new_info = m_info[0];
        new_info.set_value(candidate.get_value());
        vertex.update_value(new_info, "parameters");
        grafos.set("tree", vertex_number, vertex, "vertex");
        vertex = grafos.get("tree", vertex_number, "vertex");

        for (const value_T& column: new_info.get_characteristic("column").get_values())
        {
            if (boost::get<std::string>(candidate.get_characteristic("name").get_value()) == "input")
            {
                std::vector<characteristic> extra_characteristics_tree;
                extra_characteristics_tree.push_back(characteristic("position matrix", column, false));
                m_info = vertex.get_multi_info(extra_characteristics_tree, "variables", false);
                if (m_info.size() > 1 || m_info.size() == 0)
                {
                    std::cout << "Incoherent data for name *" << candidate.get_characteristic("name").get_value() << "* in data *variables*." << std::endl;
                    return -1;
                }
                value_T min = candidate.get_value();
                value_T max = candidate.get_value();
                value_T cost = m_info[0].get_characteristic("cost").get_value();
                std::string name = boost::get<std::string>(m_info[0].get_characteristic("name").get_value());
                int code = update_bnds_obj(vertex, "tree", "vertex", vertex_number, name, "variables", extra_characteristics_tree, false, max, min, cost, true);
                if (code != 0) return code;
            }
        }
    }
    else if (boost::get<std::string>(candidate.get_characteristic("problem").get_value()) == "DC OPF")
    {
        if (boost::get<std::string>(candidate.get_characteristic("name").get_value()) == "active power max limit")
        {
            convert2pu(candidate);
            int required_position = -1;
            // Get pos corresponding to ID
            for (const int& pos_gen : grafos.get_positions("network", "generator"))
            {
                if (grafos.get_characteristic_component("network", pos_gen, "vertex", "ID").get_value() == candidate.get_characteristic("ID").get_value())
                {
                    required_position = pos_gen;
                    break;
                }
            }

            // TODO: Replace value or smart checking - t.b.ask
            // Finding target variable
            information target_variable;
            target_variable.set_characteristic(characteristic("name", std::string("active power generation"), false), false);
            if (candidate.exist("pt")) target_variable.set_characteristic(candidate.get_characteristic("pt"), false);
            if (candidate.exist("hour")) target_variable.set_characteristic(candidate.get_characteristic("hour"), false);

            characteristic new_max;
            new_max.set_name("max");
            new_max.set_value(candidate.get_value());

            grafos.update_information_characteristic("network", required_position, "vertex", "variables", target_variable, new_max);

            characteristic position_matrix = grafos.get_information_characteristic("network", required_position, "vertex", "variables", target_variable, "position matrix");

            solver.setColumnUpper(boost::get<int>(position_matrix.get_value()), boost::get<double>(new_max.get_value()));
        }
    }
    return 0;

}

int models::declare_moea_variables()
{
    moea_model.create_data_set("variables");
    
    std::vector<information> TEP = data_parameters.get_multi_parameter_type(std::vector<characteristic>({characteristic("name", std::string("TEP"), false)}), "model", false);

    for (int node = 0; node < grafos.number_nodes("network"); node++)
    {
        graph_data vertex = grafos.get("network", node, "vertex");
        if (boost::get<std::string>(vertex.get_characteristic("type").get_value()) == "branch")
        {
            if (TEP.size() > 1 || TEP.size() == 0) return -1;
            if (boost::get<std::string>(TEP[0].get_characteristic("engine").get_value()) == "fdif" && boost::get<bool>(TEP[0].get_value()))
            {
                std::vector<information> m_info = vertex.get_multi_info(std::vector<characteristic>({characteristic("name", std::string("vTEP"), false)}), "parameters", false);
                if (m_info.size() > 1 || m_info.size() == 0)
                {
                    std::cout << "Incoherent data for name *vTEP* in data *parameters*." << std::endl;
                    return -1;
                }
                if (boost::get<bool>(m_info[0].get_value()))
                {
                    int min = 0, max = 1;
                    m_info = vertex.get_multi_info(std::vector<characteristic>({characteristic("name", std::string("CTEP"), false)}), "parameters", false);
                    if (m_info.size() > 1 || m_info.size() == 0)
                    {
                        std::cout << "Incoherent data for name *vTEP* in data *parameters*." << std::endl;
                        return -1;
                    }
                    double cost = boost::get<double>(m_info[0].get_value());
                    information MOEA_variable;
                    MOEA_variable.set_characteristics(vertex.get_characteristics());
                    MOEA_variable.set_characteristic(characteristic("name", std::string("new branch variable"), false), false);
                    MOEA_variable.set_characteristic(characteristic("min", min, false), false);
                    MOEA_variable.set_characteristic(characteristic("max", max, false), false);
                    MOEA_variable.set_characteristic(characteristic("cost", cost, false), false);
                    MOEA_variable.set_characteristic(characteristic("type variable", std::string("binary"), false), false);
                    MOEA_variable.set_value(0);
                    moea_model.push_back("variables", MOEA_variable);
                }
            }
        }
    }
    
    return 0;
}

int models::declare_moea_objectives()
{
    moea_model.create_data_set("objectives");
    std::vector<information> TEP = data_parameters.get_multi_parameter_type(std::vector<characteristic>({characteristic("name", std::string("TEP"), false)}), "model", false);
    if (TEP.size() > 1 || TEP.size() == 0) return -1;
    if (boost::get<std::string>(TEP[0].get_characteristic("engine").get_value()) == "fdif" && boost::get<bool>(TEP[0].get_value()))
    {
        information MOEA_objective;
        MOEA_objective.set_characteristic(characteristic("name", std::string("transmission expansion cost"), false), false);
        MOEA_objective.set_value(0);
        moea_model.push_back("objectives", MOEA_objective);
    }
    return 0;
}

int models::create_moea_problem()
{
    int code = declare_moea_variables();
    if (code != 0) return code;
    code = declare_moea_objectives();
    if (code != 0) return code;

    return 0;
}

void models::get_MOEA_variables(std::vector<std::string>& IDs, std::vector<std::string>& names, std::vector<double>& min, std::vector<double>& max)
{
    for (const information& info : moea_model.get_data_set("variables"))
    {
        IDs.push_back(boost::get<std::string>(info.get_characteristic("ID").get_value()));
        names.push_back(boost::get<std::string>(info.get_characteristic("name").get_value()));
        min.push_back(double(boost::get<int>(info.get_characteristic("min").get_value())));
        max.push_back(double(boost::get<int>(info.get_characteristic("max").get_value())));
    }
}

void models::get_moea_objectives(std::vector<std::string>& names)
{
    for (const information& info : moea_model.get_data_set("objectives"))
        names.push_back(boost::get<std::string>(info.get_characteristic("name").get_value()));
}

