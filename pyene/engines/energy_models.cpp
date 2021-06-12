#include "energy_models.h"
#include <vector>
#include <string>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <stdlib.h> 

using namespace std;

// Global functions

void models_cpp::add_variables(string name, int number)
{
    initial_position_variables.insert(pair<string, int>(name, 
        number_variables));
    number_variables += number;
}

void models_cpp::add_constraints(string name, int number)
{
    initial_position_constraints.insert(pair<string, int>(name, 
        number_constraints));
    number_constraints += number;
}

models_cpp::models_cpp(){ };

models_cpp::~models_cpp() { 
    delete[] solution;
};

// Energy tree formulation

void models_cpp::load_energy_tree_information(int number_nodes, int number_tree, 
    const vector<vector<int> > &LLEB_in, const vector<vector<int> > &LLEA_in,
    const vector<double> &intake, const vector<double> &output,
    const vector<double> &weight){
    nodes_tree = number_nodes;
    number_trees = number_tree;
    LLEB = LLEB_in;
    LLEA = LLEA_in;
    energy_intake = intake;
    energy_output = output;
    weight_nodes = weight;    
}

void models_cpp::create_energy_tree_model(){
    number_variables = 0;
    number_constraints = 0;
    declaration_variables_em();
    energy_balance();
    energy_aggregation();    
}

void models_cpp::declaration_variables_em(){
    string aux_name;
    for (int xtr = 0; xtr < number_trees; xtr++)
    {
        aux_name = "PartialStorage("+to_string(xtr)+")";
        add_variables(aux_name, nodes_tree);
        aux_name = "TotalStorage("+to_string(xtr)+")";
        add_variables(aux_name, nodes_tree);
        aux_name = "InputsTree("+to_string(xtr)+")";
        add_variables(aux_name, nodes_tree);
        aux_name = "OutputsTree("+to_string(xtr)+")";
        add_variables(aux_name, nodes_tree);
    }

    // Declaration limit variables

    for (int xtr = 0; xtr < number_trees; xtr++)
    {
        // Limits partial storage
        colLower.push_back(0.0);
        colUpper.push_back(0.0);
        for (int xtn = 1; xtn < nodes_tree; xtn++)
        {
            colLower.push_back(0.0);
            colUpper.push_back(COIN_DBL_MAX);
        }

        // Limits total storage
        colLower.push_back(0.0);
        colUpper.push_back(0.0);
        for (int xtn = 1; xtn < nodes_tree; xtn++)
        {
            colLower.push_back(0.0);
            colUpper.push_back(COIN_DBL_MAX);
        }

        // Limits inputs tree
        for (int xtn = 0; xtn < nodes_tree; xtn++)
        {
            colLower.push_back(energy_intake[xtr*nodes_tree + xtn]);
            colUpper.push_back(energy_intake[xtr*nodes_tree + xtn]);
        }

        // Limits outputs tree
        for (int xtn = 0; xtn < nodes_tree; xtn++)
        {
            colLower.push_back(energy_output[xtr*nodes_tree + xtn]);
            colUpper.push_back(energy_output[xtr*nodes_tree + xtn]);
        }
    }
}

void models_cpp::energy_balance(){
    string aux_name;
    for (int xnt = 0; xnt < number_trees; xnt++)
    {
        for (int xnot = 1; xnot < nodes_tree; xnot++)
        {
            aux_name = "PartialStorage("+to_string(xnt)+")";
            rows.push_back(number_constraints + xnot - 1);
            columns.push_back(initial_position_variables[aux_name] + xnot);
            elements.push_back(1.0);
            rows.push_back(number_constraints + xnot - 1);
            elements.push_back(-1.0);
            if (LLEB[xnot][1] == 0){
                aux_name = "PartialStorage("+to_string(xnt)+")";
                columns.push_back(initial_position_variables[aux_name] + 
                    LLEB[xnot][0]);
            }
            else if (LLEB[xnot][1] == 1){
                aux_name = "TotalStorage("+to_string(xnt)+")";
                columns.push_back(initial_position_variables[aux_name] + 
                    LLEB[xnot][0]);
            }
            aux_name = "InputsTree("+to_string(xnt)+")";
            rows.push_back(number_constraints + xnot - 1);
            columns.push_back(initial_position_variables[aux_name] + xnot);
            elements.push_back(-1.0);
            aux_name = "OutputsTree("+to_string(xnt)+")";
            rows.push_back(number_constraints + xnot - 1);
            columns.push_back(initial_position_variables[aux_name] + xnot);
            elements.push_back(1.0);
        }
        aux_name = "treebalance("+to_string(xnt)+")";
        add_constraints(aux_name, nodes_tree - 1);
    }
    for (int xnt = 0; xnt < number_trees; xnt++)
    {
        for (int xnot = 1; xnot < nodes_tree; xnot++)
        {
            rowLower.push_back(0.0);
            rowUpper.push_back(0.0);
        }
    }
}

void models_cpp::energy_aggregation(){
    string aux_name;
    for (int xnt = 0; xnt < number_trees; xnt++)
    {
        for (int xnot = 1; xnot < nodes_tree; xnot++)
        {
            aux_name = "TotalStorage("+to_string(xnt)+")";
            rows.push_back(number_constraints + xnot - 1);
            columns.push_back(initial_position_variables[aux_name] + xnot);
            elements.push_back(1.0);
            rows.push_back(number_constraints + xnot - 1);
            elements.push_back(-weight_nodes[LLEA[xnot][0]]);
            if (LLEA[xnot][2] == 0){
                aux_name = "PartialStorage("+to_string(xnt)+")";
                columns.push_back(initial_position_variables[aux_name] + 
                    LLEA[xnot][1]);
            }
            else if (LLEA[xnot][2] == 1){
                aux_name = "TotalStorage("+to_string(xnt)+")";
                columns.push_back(initial_position_variables[aux_name] + 
                    LLEA[xnot][1]);
            }
            if ((1 - weight_nodes[LLEA[xnot][0]]) != 0)
            {
                rows.push_back(number_constraints + xnot - 1);
                elements.push_back(-(1 - weight_nodes[LLEA[xnot][0]]));
                if (LLEB[LLEA[xnot][0]][1] == 0){
                    aux_name = "PartialStorage("+to_string(xnt)+")";
                    columns.push_back(initial_position_variables[aux_name] + 
                        LLEB[LLEA[xnot][0]][0]);
                }
                else if (LLEB[LLEA[xnot][0]][1] == 1){
                    aux_name = "TotalStorage("+to_string(xnt)+")";
                    columns.push_back(initial_position_variables[aux_name] + 
                        LLEB[LLEA[xnot][0]][0]);
                }
            }
        }
        aux_name = "treeaggregation("+to_string(xnt)+")";
        add_constraints(aux_name, nodes_tree - 1);
    }
    for (int xnt = 0; xnt < number_trees; xnt++)
    {
        for (int xnot = 1; xnot < nodes_tree; xnot++)
        {
            rowLower.push_back(0.0);
            rowUpper.push_back(0.0);
        }
    }
}

void models_cpp::run_energy_tree(){
    /* This function calls all functions to create the model and run the
    energy tree model */
    create_energy_tree_model();

    // matrix data
    CoinBigIndex num_elements = rows.size();
    
    CoinPackedMatrix matrix(true, rows.data(), columns.data(), elements.data(), 
        num_elements);

    objective.resize(number_variables, 0.0);
    objective[1] = 1.0;
    // load problem
    model.loadProblem(matrix, colLower.data(), colUpper.data(), objective.data(),
                       rowLower.data(), rowUpper.data());
    
    /* +1 to minimize, -1 to maximize, and 0 to ignore */
    model.setOptimizationDirection(1);
    /*
    Amount of print out:
        0 - none
        1 - just final
        2 - just factorizations
        3 - as 2 plus a bit more
        4 - verbose
    */
    model.setLogLevel(0);

    // Solve
    model.primal();

    solution = model.primalColumnSolution();
    objective_function = model.objectiveValue();
}

void models_cpp::get_energy_tree_solution(vector<double> &PartialStorage, 
    vector<double> &TotalStorage, vector<double> &InputsTree,
    vector<double> &OutputsTree)
{
    string aux_name;
    for (int xtr = 0; xtr < number_trees; xtr++)
    {
        for (int xtn = 0; xtn < nodes_tree; xtn++)
        {
            aux_name = "PartialStorage("+to_string(xtr)+")";
            PartialStorage.push_back(
                solution[initial_position_variables[aux_name] + xtn]);
            aux_name = "TotalStorage("+to_string(xtr)+")";
            TotalStorage.push_back(
                solution[initial_position_variables[aux_name] + xtn]);
            aux_name = "InputsTree("+to_string(xtr)+")";
            InputsTree.push_back(
                solution[initial_position_variables[aux_name] + xtn]);
            aux_name = "OutputsTree("+to_string(xtr)+")";
            OutputsTree.push_back(
                solution[initial_position_variables[aux_name] + xtn]);
        }
    }
}

double models_cpp::get_objective_function_em()
{
    return objective_function;
}


// Reduced OPF

/**** Functions to add components ****/

void models_cpp::add_bus(const vector<double> &P, int n, string t){
    /*
        bus stores information of buses in the system

        string t specifies the type of element that is added. This structure 
        allows generalising the addition of any bus in the system, e.g. ac or
        dc bus.
    */
    bus aux;
    aux.active_power_demand = P;
    aux.number = n;
    aux.type = t;
    buses.push_back(aux);
}

void models_cpp::add_branch(const vector<double> &r, const vector<double> &x, 
    const vector<double> &Pmax, int from, int to, int n, string t){
    /*
        branch stores information of components that connect two buses

        string t specifies the type of element that is added. This structure 
        allows generalising the addition of any element that connects two buses,
        e.g. transmission lines, transformers.
    */
    branch aux;
    aux.resistance = r;
    aux.reactance = x;
    aux.maximum_P_flow = Pmax;
    aux.from_bus = from;
    aux.to_bus = to;
    aux.number = n;
    aux.type = t;
    branches.push_back(aux);
}

void models_cpp::add_generator(const vector<double> &Pmax, 
    const vector<double> &Pmin, int bn, int n, string t,
    double fc, double vc, const vector<double> &a_pwl, 
    const vector<double> &b_pwl, bool is_active){
    /*
        generator stores information of any type of generator

        string t specifies the type of element that is added. This structure 
        allows generalising the addition of any generator,
        e.g. thermal, hydro.
    */
    generator aux;
    aux.maximum_generation = Pmax;
    aux.minimum_generation = Pmin;
    aux.bus_number = bn;
    aux.number = n;
    aux.type = t;
    aux.fixed_cost = fc;
    aux.variable_cost = vc;
    aux.piecewise = make_pair(a_pwl, b_pwl);
    aux.is_active = is_active;
    generators.push_back(aux);
}

void models_cpp::set_integer_data_power_system(string name, int value){
    /*
    This function set any general integer data of the power system.
    The current available options are:
    - "number periods"
    - "number representative days"
    - "slack bus"
    The desired option should be passes as a string and the value as an int
    */
    integer_powersystem_data[name] = value;
}

void models_cpp::set_continuous_data_power_system(string name, double value){
    /*
    This function set any general continuous data of the power system.
    The current available options are:
    - "base power"
    The desired option should be passes as a string and the value as a double
    */
    continuous_powersystem_data[name] = value;
}

/**** Functions to create the reduced network model ****/

void models_cpp::create_graph_database(){
    /* 
    This method creates a structure in the for of a graph to store the 
    information of a power system. Each node in the graph corresponds to an 
    element of the vectors:
    - buses
    - branches
    - generators
    */

    // Creating the pair number of element(bus, branch, generator) and graph
    // node number

    // Reserving memory for vector
    buses_g.resize(buses.size());
    branches_g.resize(branches.size());
    generators_g.resize(generators.size());

    // Storing the information
    // The initial 'n' nodes of the graph correspond to the nodes
    nodes_graph = 0;
    for (size_t i = 0; i < buses.size(); i++)
    {
        buses_g[i].first = buses[i].number;
        buses_g[i].second = nodes_graph;
        nodes_graph++;
    }
    
    // The next nodes in the graph correspond to branches
    for (size_t i = 0; i < branches.size(); i++)
    {
        branches_g[i].first = branches[i].number;
        branches_g[i].second = nodes_graph;
        nodes_graph++;
    }
    
    // The next nodes in the graph correspond to generators
    for (size_t i = 0; i < generators.size(); i++)
    {
        generators_g[i].first = generators[i].number;
        generators_g[i].second = nodes_graph;
        nodes_graph++;
    }

    // Creating the graph
    for (size_t i = 0; i < buses.size(); i++)
    {
        // Adding connections between nodes and branches
        for (size_t j = 0; j < branches.size(); j++)
        {
            if (buses[i].number == branches[j].from_bus)
                boost::add_edge(buses_g[i].second, branches_g[j].second, 
                    power_system_datastructure);
            if (buses[i].number == branches[j].to_bus)
                boost::add_edge(branches_g[j].second, buses_g[i].second,
                    power_system_datastructure);
        }
        // Adding connections between nodes and generators
        for (size_t j = 0; j < generators.size(); j++)
        {
            if (buses[i].number == generators[j].bus_number)
            {
                boost::add_edge(buses_g[i].second, generators_g[j].second, 
                    power_system_datastructure);
            }
        }
    }

    // Adding the information of each structure to the graph
    int counter_nodes = 0;
    for (size_t i = 0; i < buses.size(); i++)
    {
        power_system_datastructure[counter_nodes].info_bus = buses[i];
        power_system_datastructure[counter_nodes].type = "bus";
        counter_nodes++;
    }
    for (size_t i = 0; i < branches.size(); i++)
    {
        power_system_datastructure[counter_nodes].info_branch = branches[i];
        power_system_datastructure[counter_nodes].type = "branch";
        counter_nodes++;
    }
    for (size_t i = 0; i < generators.size(); i++)
    {
        power_system_datastructure[counter_nodes].infor_generator = 
            generators[i];
        power_system_datastructure[counter_nodes].type = "generator";
        counter_nodes++;
    }    
}

void models_cpp::create_susceptance_matrix(){
    /*
    This function creates the susceptance matrix for the DC OPF
    */
    vector<double> diag_vals(buses_g.size(), 0.0);
    susceptance_matrix.resize(
        buses_g.size(),vector<double> (buses_g.size(),0.0));
    int pos1 = -1;
    int pos2 = -1;
    for (size_t i = 0; i < branches_g.size(); i++)
    {
        pos1 = -1;
        pos2 = -1;
        for (size_t j = 0; j < buses_g.size(); j++)
        {
            if (power_system_datastructure[branches_g[i].second].
                info_branch.from_bus == buses_g[j].first)
                pos1 = j;
            else if (power_system_datastructure[branches_g[i].second].
                info_branch.to_bus == buses_g[j].first)
                pos2 = j;
            if(pos1 != -1 && pos2 != -1) break;
        }
        susceptance_matrix[pos1][pos2] += 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        susceptance_matrix[pos2][pos1] += 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        diag_vals[pos1] -= 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        diag_vals[pos2] -= 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0]; 
    }
    for (size_t i = 0; i < buses_g.size(); i++)
        susceptance_matrix[i][i] = diag_vals[i];
}

void models_cpp::create_reduced_dc_opf_model(){
    number_variables = 0;
    number_constraints = 0;
    declaration_variables_dc_opf();
    active_power_balance_ac();
    active_power_flow_limit_ac();
    active_power_generation_cost();
    objective_function_nm();    
}

void models_cpp::declaration_variables_dc_opf(){

    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, generators_g.size());   // generation
            aux_name = "theta("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, buses_g.size());    // voltage angle
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, buses_g.size()); // Load curtailment
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, buses_g.size()); // Generation curtailment
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, generators_g.size());   // Cost generation
        }
    }
    // Declaration limit variables

    int aux_count = integer_powersystem_data["number periods"];
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            // maximum and minimum active power generation
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                if(power_system_datastructure[generators_g[xgen].second].
                    infor_generator.is_active)
                {
                    colLower.push_back(
                        power_system_datastructure[generators_g[xgen].second].
                        infor_generator.minimum_generation[xrd*aux_count + xp]);
                    colUpper.push_back
                        (power_system_datastructure[generators_g[xgen].second].
                        infor_generator.maximum_generation[xrd*aux_count + xp]);
                }
                else
                {
                    colLower.push_back(0.0);
                    colUpper.push_back(0.0);
                }                
            }

            // maximum and minimum voltage phase angle
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                if (buses_g[xnode].first != 
                    integer_powersystem_data["slack bus"])
                {
                    colLower.push_back(-COIN_DBL_MAX);
                    colUpper.push_back(COIN_DBL_MAX);
                }
                else{
                    colLower.push_back(0.0);
                    colUpper.push_back(0.0);
                }
            }

            // maximum and minimum load curtailment
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                colLower.push_back(0.0);
                if (power_system_datastructure[buses_g[xnode].second].info_bus.
                    active_power_demand[xrd*aux_count + xp] > 0)
                    colUpper.push_back(COIN_DBL_MAX);
                else
                    colUpper.push_back(0.0);
            }

            // maximum and minimum generation curtailment
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                colLower.push_back(0.0);
                bool flag_gen = false;
                AdjacencyIterator ai, a_end;
                boost::tie(ai, a_end) = boost::adjacent_vertices(buses_g[xnode].second, 
                    power_system_datastructure);
                for (; ai != a_end; ai++) {
                    if(power_system_datastructure[*ai].type == "generator")
                    {
                        colUpper.push_back(COIN_DBL_MAX);
                        flag_gen = true;
                        break;
                    }
                }
                if (!flag_gen)
                    colUpper.push_back(0.0);
            }

            // maximum and minimum generation cost
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                colLower.push_back(0.0);
                colUpper.push_back(COIN_DBL_MAX);
            }
        }
    }
}

void models_cpp::active_power_balance_ac(){
    /*
    This file constructs the active power balance constraint
    */
    // Determining generators per node
    vector< vector<int> > pos_gen(buses_g.size());
    for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
    {
        AdjacencyIterator ai, a_end;
        boost::tie(ai, a_end) = boost::adjacent_vertices(buses_g[xnode].second, 
            power_system_datastructure);
        for (; ai != a_end; ai++) {
            if(power_system_datastructure[*ai].type == "generator")
            {
                for (size_t xgen = 0; xgen < generators_g.size(); 
                    xgen++)
                {
                    if (power_system_datastructure[*ai].
                        infor_generator.number == generators_g[xgen].
                        first && power_system_datastructure[*ai].
                        infor_generator.is_active)
                    {
                        pos_gen[xnode].push_back(xgen);
                        break;
                    }
                }
            }
        }
    }
    
    // definition of constraint
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            string aux_name;
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                // Generation
                aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
                for (size_t xgen = 0; xgen < pos_gen[xnode].size(); xgen++)
                {
                    rows.push_back(number_constraints + xnode);
                    columns.push_back(initial_position_variables[aux_name] + 
                        pos_gen[xnode][xgen]);
                    elements.push_back(1.0);
                }
                // Angles
                aux_name = "theta("+to_string(xrd)+","+to_string(xp)+")";
                for (size_t xang = 0; xang < buses_g.size(); xang++)
                {
                    if (abs(susceptance_matrix[xnode][xang]) > 1e-8)
                    {
                        rows.push_back(number_constraints + xnode);
                        columns.push_back(initial_position_variables[aux_name] +
                            xang);
                        elements.push_back(susceptance_matrix[xnode][xang]);
                    }
                }
                // Load Curtailment
                aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
                rows.push_back(number_constraints + xnode);
                columns.push_back(initial_position_variables[aux_name] + xnode);
                elements.push_back(1.0);
                // Generation curtailment
                aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
                rows.push_back(number_constraints + xnode);
                columns.push_back(initial_position_variables[aux_name] +
                    xnode);
                elements.push_back(-1.0);
            }            
            aux_name = "PB("+to_string(xrd)+","+to_string(xp)+")";
            add_constraints(aux_name, buses_g.size());
        }
    }

    // definition of limits for constraints
    int aux_count = integer_powersystem_data["number periods"];
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            string aux_name = "PB("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                rowLower.push_back(
                    power_system_datastructure[buses_g[xnode].second].info_bus.
                    active_power_demand[xrd*aux_count + xp]);
                rowUpper.push_back(
                    power_system_datastructure[buses_g[xnode].second].info_bus.
                    active_power_demand[xrd*aux_count + xp]);
            }
        }
    }
}

void models_cpp::active_power_flow_limit_ac(){
    /*
    This function constructs the active power flow limit constraint
    */
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            string aux_name;
            for (size_t xbranch = 0; xbranch < branches_g.size(); xbranch++)
            {
                int pos1 = -1, pos2 = -1;
                for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                {
                    if (power_system_datastructure[branches_g[xbranch].second].
                        info_branch.from_bus == buses_g[xnode].first)
                        pos1 = xnode;
                    else if (power_system_datastructure[branches_g[xbranch].second].
                        info_branch.to_bus == buses_g[xnode].first)
                        pos2 = xnode;
                    if (pos1 != -1 && pos2 != -1) break;
                }
                aux_name = "theta("+to_string(xrd)+","+to_string(xp)+")";
                rows.push_back(number_constraints + xbranch);
                columns.push_back(initial_position_variables[aux_name] + pos1);
                elements.push_back(1.0/power_system_datastructure[
                    branches_g[xbranch].second].info_branch.reactance[0]);
                rows.push_back(number_constraints + xbranch);
                columns.push_back(initial_position_variables[aux_name] + pos2);
                elements.push_back(-(1.0/power_system_datastructure[
                    branches_g[xbranch].second].info_branch.reactance[0]));
            }
            aux_name = "TC("+to_string(xrd)+","+to_string(xp)+")";
            add_constraints(aux_name, branches_g.size());
        }
    }

    // Limits constraint
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            string aux_name = "TC("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xbranch = 0; xbranch < branches_g.size(); xbranch++)
            {
                rowLower.push_back(
                    -power_system_datastructure[branches_g[xbranch].second].
                    info_branch.maximum_P_flow[0]);
                rowUpper.push_back(
                    power_system_datastructure[branches_g[xbranch].second].
                    info_branch.maximum_P_flow[0]);
            }
        }
    }
}

void models_cpp::active_power_generation_cost(){
    /*
    This function constructs the active power generation cost
    */
    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                generator aux_gen = power_system_datastructure[generators_g[xgen].second].
                    infor_generator;
                for (size_t xpieces = 0; xpieces < aux_gen.piecewise.
                    first.size(); xpieces++)
                {
                    aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
                    rows.push_back(number_constraints + xpieces);
                    columns.push_back(initial_position_variables[aux_name] + xgen);
                    elements.push_back(1.0);
                    aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
                    rows.push_back(number_constraints + xpieces);
                    columns.push_back(initial_position_variables[aux_name] + xgen);
                    elements.push_back(-aux_gen.piecewise.first[xpieces] * 
                    continuous_powersystem_data["total hours period"]);
                }
                aux_name = "GC("+to_string(xrd)+","+to_string(xp)+","+
                    to_string(xgen)+")";
                add_constraints(aux_name, aux_gen.piecewise.first.size());
            }
        }
    }
    // Limits constraint
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                generator aux_gen = power_system_datastructure[generators_g[xgen].second].
                    infor_generator;
                aux_name = "GC("+to_string(xrd)+","+to_string(xp)+","+
                    to_string(xgen)+")";
                for (size_t xpieces = 0; xpieces < aux_gen.piecewise.
                    first.size(); xpieces++)
                {
                    rowLower.push_back(aux_gen.piecewise.second[xpieces] * 
                    continuous_powersystem_data["total hours period"]);
                    rowUpper.push_back(COIN_DBL_MAX);
                }
            }
        }
    }
}

void models_cpp::objective_function_nm(){
    /*
        This function constructs the objective function
    */
    objective.resize(number_variables, 0.0);
    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            // Penalise load curtailment
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                objective[initial_position_variables[aux_name] + xnode] = 10000000;
            // Penalise generation curtailment
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                objective[initial_position_variables[aux_name] + xnode] = 10000000;
            // maximum and minimum generation cost
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
                objective[initial_position_variables[aux_name] + xgen] = 1;
        }
    }
}

void models_cpp::run_reduced_dc_opf(){
    /* This function calls all functions to create the model and run the DC
    OPF with the reduced model */

    type_OPF = 1;

    create_graph_database();
    create_susceptance_matrix();
    create_reduced_dc_opf_model();

    // matrix data
    CoinBigIndex num_elements = rows.size();
    
    CoinPackedMatrix matrix(true, rows.data(), columns.data(), elements.data(), 
        num_elements);

    // load problem
    model.loadProblem(matrix, colLower.data(), colUpper.data(), objective.data(),
                       rowLower.data(), rowUpper.data());
    
    /* +1 to minimize, -1 to maximize, and 0 to ignore */
    model.setOptimizationDirection(1);
    /*
    Amount of print out:
        0 - none
        1 - just final
        2 - just factorizations
        3 - as 2 plus a bit more
        4 - verbose
    */
    model.setLogLevel(0);

    // Solve
    model.primal();

    solution = model.primalColumnSolution();
    objective_function = model.objectiveValue();
}

// Iterative functions

/*** Functions to create iterative models for the optimal power flow
 and the combined energy tree - OPF ***/

void models_cpp::solve_iterative_models(){
/*  This function performs the following actions:
    1. Determine which constraints are unfeasible with the 
    current solution
    2. Update the model with the unfeasible constraints
    3. Solve the updated model
    4. Repeat 1- 3 until the solution is feasible for all constraints
*/

    // matrix data
    CoinBigIndex num_elements = rows.size();
    
    CoinPackedMatrix matrix(true, rows.data(), columns.data(), elements.data(), 
        num_elements);
    
    model.loadProblem(matrix, colLower.data(), colUpper.data(), objective.data(),
                       rowLower.data(), rowUpper.data());

    vector< pair<double, int> > weight(number_constraints);
    vector<int> active_rows(number_constraints, numeric_limits<int>::max());
    int numberSort = 0;

    int iRow, iColumn;
    // Set up initial list
    numberSort = 0;
    for (iRow = 0; iRow < number_constraints; iRow++)
        if (rowLower[iRow] == rowUpper[iRow])
            active_rows[numberSort++] = iRow;
    
    // load the biggest piece of the piecewise linearisation of the generation cost
    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                generator aux_gen = power_system_datastructure[
                    generators_g[xgen].second].infor_generator;
                string aux_name = "GC("+to_string(xrd)+","+to_string(xp)+","+
                    to_string(xgen)+")";
                active_rows[numberSort++] = initial_position_constraints[
                    aux_name] + aux_gen.piecewise.first.size() - 1;
            }
        }
    }
                
    vector<int> whichColumns(number_variables);
    for (iColumn = 0; iColumn < number_variables; iColumn++)
         whichColumns[iColumn] = iColumn;

    vector<double> solution_rows(number_constraints, 0.0);

    // Flag that indicates if the solution is feasible for all constraints or not
    bool infeasible_solution = true;
    iterations_opt = 0; // Number of iterations in the iterative process
    int max_rows_iter = max((int)floor(number_constraints/40), 1);
    while (infeasible_solution)
    {
        // Order in descending order the list of rows
        sort(active_rows.begin(), active_rows.end());

        ClpSimplex small_dc_model(&model, numberSort, active_rows.data(), 
            number_variables, whichColumns.data());
        if (iterations_opt == 0)
        {
            small_dc_model.primal();
        }
        else {
            ClpSolve solveOptions;
            solveOptions.setSolveType(ClpSolve::useDual);
            small_dc_model.setLogLevel(0);
            small_dc_model.initialSolve(solveOptions);
        }

        iterations_opt++;

        std::fill(solution_rows.begin(), solution_rows.end(), 0.0);

        model.times(1.0, small_dc_model.primalColumnSolution(), solution_rows.data());

        for (iRow = 0; iRow < number_constraints; iRow++){
            weight[iRow].first = -max(max(solution_rows[iRow] - rowUpper[iRow],
                                    rowLower[iRow] - solution_rows[iRow]), 0.0);
            weight[iRow].second = iRow;
            if(abs(weight[iRow].first) < 1e-6) weight[iRow].first = 0.0;
        }
        
        infeasible_solution = false;
        for (iRow = 0; iRow < number_constraints; iRow++){
            if(weight[iRow].first < 0.0){
                infeasible_solution = true;
                break;
            }
        }

        if (infeasible_solution)
        {
            sort(weight.begin(), weight.end());
            for (iRow = 0; iRow < max_rows_iter; iRow++)
                active_rows[numberSort++] = weight[iRow].second;            
        }
        else{
            double *solution_copy = small_dc_model.primalColumnSolution();
            solution = new double[number_variables];
            for (int sol = 0; sol < number_variables; sol++)
                solution[sol] = solution_copy[sol];
            objective_function = small_dc_model.objectiveValue();
        }
    }
}

// Reduced OPF - Iterative

/**** Functions to create the iterative network model ****/

void models_cpp::run_iterative_reduced_dc_opf(){
    /* This function calls all functions to create the model and run the DC
    OPF with the reduced model and in an iterative way. The non-binding constraints
    (power flow limits) are introduced iteratively in CLP and the problem is 
    reoptimised using the dual simplex method */

    type_OPF = 2;

    create_graph_database();
    create_susceptance_matrix();
    create_reduced_dc_opf_model();

    solve_iterative_models();
}

// Reduced OPF version 2

/**** Functions to create the reduced network model ****/

void models_cpp::create_inverse_susceptance_matrix(){
    /*
    This function creates the susceptance matrix for the DC OPF
    */
    mat mat_susceptance(buses_g.size(), buses_g.size(), arma::fill::zeros);
    vector<double> diag_vals(buses_g.size(), 0.0);
    susceptance_matrix.resize(
        buses_g.size(),vector<double> (buses_g.size(),0.0));
    int pos1 = -1;
    int pos2 = -1;
    for (size_t i = 0; i < branches_g.size(); i++)
    {
        pos1 = -1;
        pos2 = -1;
        for (size_t j = 0; j < buses_g.size(); j++)
        {
            if (power_system_datastructure[branches_g[i].second].
                info_branch.from_bus == buses_g[j].first)
                pos1 = j;
            else if (power_system_datastructure[branches_g[i].second].
                info_branch.to_bus == buses_g[j].first)
                pos2 = j;
            if(pos1 != -1 && pos2 != -1) break;
        }
        mat_susceptance(pos1, pos2) += 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        mat_susceptance(pos2, pos1) += 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        mat_susceptance(pos1, pos1) -= 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        mat_susceptance(pos2, pos2) -= 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0]; 
    }
    
    int slack_pos = 0;
    for (size_t i = 0; i < buses_g.size(); i++)
    {
        if (buses_g[i].first == integer_powersystem_data["slack bus"])
        {
            slack_pos = i;
            break;
        }
    }

    mat_susceptance.shed_row(slack_pos);
    mat_susceptance.shed_col(slack_pos);

    mat inverse;
    inverse_susceptance_completed = true;
    inverse_susceptance_completed = arma::inv(inverse, mat_susceptance);
    
    if (inverse_susceptance_completed)
        inverse_mat_sustance = sp_mat(inverse);
}

void models_cpp::create_dc_opf_model_v2(){
    number_variables = 0;
    number_constraints = 0;
    declaration_variables_dc_opf_v2();
    active_power_balance_ac_system();
    active_power_flow_limit_ac_v2();
    active_power_generation_cost();
    objective_function_nm();  
}

void models_cpp::declaration_variables_dc_opf_v2(){

    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, generators_g.size());   // generation
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, buses_g.size()); // Load curtailment
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, buses_g.size()); // Generation curtailment
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, generators_g.size());   // Cost generation
        }
    }
    // Declaration limit variables

    int aux_count = integer_powersystem_data["number periods"];
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            // maximum and minimum active power generation
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                if(power_system_datastructure[generators_g[xgen].second].
                    infor_generator.is_active)
                {
                    colLower.push_back(
                        power_system_datastructure[generators_g[xgen].second].
                        infor_generator.minimum_generation[xrd*aux_count + xp]);
                    colUpper.push_back
                        (power_system_datastructure[generators_g[xgen].second].
                        infor_generator.maximum_generation[xrd*aux_count + xp]);
                }
                else
                {
                    colLower.push_back(0.0);
                    colUpper.push_back(0.0);
                }                
            }

            // maximum and minimum load curtailment
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                colLower.push_back(0.0);
                if (power_system_datastructure[buses_g[xnode].second].info_bus.
                    active_power_demand[xrd*aux_count + xp] > 0)
                    colUpper.push_back(COIN_DBL_MAX);
                else
                    colUpper.push_back(0.0);
            }

            // maximum and minimum generation curtailment
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                colLower.push_back(0.0);
                bool flag_gen = false;
                AdjacencyIterator ai, a_end;
                boost::tie(ai, a_end) = boost::adjacent_vertices(buses_g[xnode].second, 
                    power_system_datastructure);
                for (; ai != a_end; ai++) {
                    if(power_system_datastructure[*ai].type == "generator")
                    {
                        colUpper.push_back(COIN_DBL_MAX);
                        flag_gen = true;
                        break;
                    }
                }
                if (!flag_gen)
                    colUpper.push_back(0.0);
            }

            // maximum and minimum generation cost
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                colLower.push_back(0.0);
                colUpper.push_back(COIN_DBL_MAX);
            }
        }
    }
}

void models_cpp::active_power_balance_ac_system(){
    /*
    This file constructs the active power balance constraint
    */

    // definition of constraint
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            string aux_name;
            // Generation
            aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); 
                xgen++)
            {
                rows.push_back(number_constraints);
                columns.push_back(initial_position_variables[aux_name] + 
                    xgen);
                elements.push_back(1.0);
            }
            
            // Load Curtailment
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                rows.push_back(number_constraints);
                columns.push_back(initial_position_variables[aux_name] + xnode);
                elements.push_back(1.0);
            }

            // Generation curtailment
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                rows.push_back(number_constraints);
                columns.push_back(initial_position_variables[aux_name] +
                    xnode);
                elements.push_back(-1.0);
            }
            aux_name = "PB("+to_string(xrd)+","+to_string(xp)+")";
            add_constraints(aux_name, 1);
        }
    }

    // definition of limits for constraints
    int aux_count = integer_powersystem_data["number periods"];
    double total_demand;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            total_demand = 0;
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                total_demand += power_system_datastructure[buses_g[xnode].second].info_bus.
                    active_power_demand[xrd*aux_count + xp];
            }
            rowLower.push_back(total_demand);
            rowUpper.push_back(total_demand);
        }
    }
}

void models_cpp::active_power_flow_limit_ac_v2(){
    /*
    This function constructs the active power flow limit constraint
    */
    // Determining generators per node
    vector< vector<int> > pos_gen(buses_g.size());
    for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
    {
        AdjacencyIterator ai, a_end;
        boost::tie(ai, a_end) = boost::adjacent_vertices(buses_g[xnode].second, 
            power_system_datastructure);
        for (; ai != a_end; ai++) {
            if(power_system_datastructure[*ai].type == "generator")
            {
                for (size_t xgen = 0; xgen < generators_g.size(); 
                    xgen++)
                {
                    if (power_system_datastructure[*ai].
                        infor_generator.number == generators_g[xgen].
                        first && power_system_datastructure[*ai].
                        infor_generator.is_active)
                    {
                        pos_gen[xnode].push_back(xgen);
                        break;
                    }
                }
            }
        }
    }
    // Finding position slack bus
    int slack_pos;
    for (size_t i = 0; i < buses_g.size(); i++)
    {
        if (buses_g[i].first == integer_powersystem_data["slack bus"])
        {
            slack_pos = i;
            break;
        }
    }
    int aux_count = integer_powersystem_data["number periods"];
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            string aux_name;
            for (size_t xbranch = 0; xbranch < branches_g.size(); xbranch++)
            {
                int pos1 = -1, pos2 = -1;
                for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                {
                    if (power_system_datastructure[branches_g[xbranch].second].
                        info_branch.from_bus == buses_g[xnode].first)
                        pos1 = xnode;
                    else if (power_system_datastructure[branches_g[xbranch].second].
                        info_branch.to_bus == buses_g[xnode].first)
                        pos2 = xnode;
                    if (pos1 != -1 && pos2 != -1) break;
                }
                int counter = 0;
                double value_ope = 0.0;
                // Substracting the elements of the matrix
                for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                {
                    if (buses_g[xnode].first != 
                        integer_powersystem_data["slack bus"])
                    {
                        if (pos1 != slack_pos && pos2 != slack_pos){
                            if (pos1 > slack_pos && pos2 > slack_pos)
                                value_ope = 
                                    inverse_mat_sustance(pos1 - 1, counter) - 
                                    inverse_mat_sustance(pos2 - 1, counter);
                            else if (pos1 < slack_pos && pos2 > slack_pos)
                                value_ope = 
                                    inverse_mat_sustance(pos1, counter) - 
                                    inverse_mat_sustance(pos2 - 1, counter);
                            else if (pos1 > slack_pos && pos2 < slack_pos)
                                value_ope = 
                                    inverse_mat_sustance(pos1 - 1, counter) - 
                                    inverse_mat_sustance(pos2, counter);
                            else if (pos1 < slack_pos && pos2 < slack_pos)
                                value_ope = 
                                    inverse_mat_sustance(pos1, counter) - 
                                    inverse_mat_sustance(pos2, counter);
                        }
                        if (pos1 == slack_pos && pos2 != slack_pos){
                            if (pos2 > slack_pos)
                                value_ope = 
                                    -inverse_mat_sustance(pos2 - 1, counter);
                            else if (pos2 < slack_pos)
                                value_ope = 
                                    -inverse_mat_sustance(pos2, counter);
                        }
                        if (pos1 != slack_pos && pos2 == slack_pos){
                            if (pos1 > slack_pos)
                                value_ope = 
                                    inverse_mat_sustance(pos1 - 1, counter);
                            else if (pos1 < slack_pos)
                                value_ope = 
                                    inverse_mat_sustance(pos1, counter);
                        }
                        counter++;
                    }
                    if (value_ope != 0.0)
                    {
                        // Generation
                        aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
                        for (size_t xgen = 0; xgen < pos_gen[xnode].size(); xgen++)
                        {
                            rows.push_back(number_constraints + xbranch);
                            columns.push_back(initial_position_variables[aux_name] + 
                                pos_gen[xnode][xgen]);
                            elements.push_back(-value_ope * (
                                1.0/power_system_datastructure[
                                branches_g[xbranch].second].info_branch.
                                reactance[0]));
                        }
                        // Load Curtailment
                        aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
                        rows.push_back(number_constraints + xbranch);
                        columns.push_back(initial_position_variables[aux_name] + xnode);
                        elements.push_back(-value_ope * (
                            1.0/power_system_datastructure[
                            branches_g[xbranch].second].info_branch.
                            reactance[0]));
                        // Generation curtailment
                        aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
                        rows.push_back(number_constraints + xbranch);
                        columns.push_back(initial_position_variables[aux_name] +
                            xnode);
                        elements.push_back(value_ope * (
                            1.0/power_system_datastructure[
                            branches_g[xbranch].second].info_branch.
                            reactance[0]));
                    }
                    value_ope = 0.0;
                }
            }
            aux_name = "TC("+to_string(xrd)+","+to_string(xp)+")";
            add_constraints(aux_name, branches_g.size());
        }
    }

    // Limits constraint
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            vec active_dem(buses_g.size() - 1);
            int counter = 0;
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                if (buses_g[xnode].first != integer_powersystem_data["slack bus"])
                {
                    active_dem(counter) = 
                        power_system_datastructure[buses_g[xnode].second].
                        info_bus.active_power_demand[xrd*aux_count + xp];
                    counter++;
                }
            }
            vec constant_product = inverse_mat_sustance * active_dem;
            string aux_name = "TC("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xbranch = 0; xbranch < branches_g.size(); xbranch++)
            {
                int pos1 = -1, pos2 = -1;
                for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                {
                    if (power_system_datastructure[branches_g[xbranch].second].
                        info_branch.from_bus == buses_g[xnode].first)
                        pos1 = xnode;
                    else if (power_system_datastructure[branches_g[xbranch].second].
                        info_branch.to_bus == buses_g[xnode].first)
                        pos2 = xnode;
                    if (pos1 != -1 && pos2 != -1) break;
                }
                double value_ope = 0.0;
                if (pos1 != slack_pos && pos2 != slack_pos){
                    if (pos1 > slack_pos && pos2 > slack_pos)
                        value_ope = 
                            constant_product(pos1 - 1) - 
                            constant_product(pos2 - 1);
                    else if (pos1 < slack_pos && pos2 > slack_pos)
                        value_ope = 
                            constant_product(pos1) - 
                            constant_product(pos2 - 1);
                    else if (pos1 > slack_pos && pos2 < slack_pos)
                        value_ope = 
                            constant_product(pos1 - 1) - 
                            constant_product(pos2);
                    else if (pos1 < slack_pos && pos2 < slack_pos)
                        value_ope = 
                            constant_product(pos1) - 
                            constant_product(pos2);
                }
                if (pos1 == slack_pos && pos2 != slack_pos){
                    if (pos2 > slack_pos)
                        value_ope = 
                            -constant_product(pos2 - 1);
                    else if (pos2 < slack_pos)
                        value_ope = 
                            -constant_product(pos2);
                }
                if (pos1 != slack_pos && pos2 == slack_pos){
                    if (pos1 > slack_pos)
                        value_ope = 
                            constant_product(pos1 - 1);
                    else if (pos1 < slack_pos)
                        value_ope = 
                            constant_product(pos1);
                }

                rowLower.push_back(
                    -power_system_datastructure[branches_g[xbranch].second].
                    info_branch.maximum_P_flow[0] - (value_ope * (
                    1.0/power_system_datastructure[branches_g[xbranch].second].
                    info_branch.reactance[0])));
                rowUpper.push_back(
                    power_system_datastructure[branches_g[xbranch].second].
                    info_branch.maximum_P_flow[0] - (value_ope * (
                    1.0/power_system_datastructure[branches_g[xbranch].second].
                    info_branch.reactance[0])));
            }
        }
    }
}

void models_cpp::run_iterative_reduced_dc_opf_v2(){
    /* This function calls all functions to create the model and run the DC
    OPF with the reduced model and in an iterative way. The non-binding constraints
    (power flow limits) are introduced iteratively in CLP and the problem is 
    reoptimised using the dual simplex method */

    type_OPF = 3;

    create_graph_database();
    create_inverse_susceptance_matrix();
    if (inverse_susceptance_completed)
    {
        create_dc_opf_model_v2();
        solve_iterative_models();
    }   
}

vector<double> models_cpp::calculate_angles(){
    /* This function calculate the value of the voltage angles */

    vector<double> generation;
    vector<double> generation_curtailment;
    vector<double> load_curtailment;
    // Determining generators per node
    vector< vector<int> > pos_gen(buses_g.size());
    for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
    {
        AdjacencyIterator ai, a_end;
        boost::tie(ai, a_end) = boost::adjacent_vertices(buses_g[xnode].second, 
            power_system_datastructure);
        for (; ai != a_end; ai++) {
            if(power_system_datastructure[*ai].type == "generator")
            {
                for (size_t xgen = 0; xgen < generators_g.size(); 
                    xgen++)
                {
                    if (power_system_datastructure[*ai].
                        infor_generator.number == generators_g[xgen].
                        first && power_system_datastructure[*ai].
                        infor_generator.is_active)
                    {
                        pos_gen[xnode].push_back(xgen);
                        break;
                    }
                }
            }
        }
    }
    // Extracting generation
    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
                generation.push_back(solution[
                    initial_position_variables[aux_name]+xgen]);
        }
    }

    // Calculating net generation
    vector<double> net_generation;
    double net_gen_node;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            string aux_name;
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                // Generation
                net_gen_node = 0.0;
                aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
                for (size_t xgen = 0; xgen < pos_gen[xnode].size(); xgen++)
                {
                    net_gen_node += generation[
                        xrd * integer_powersystem_data["number representative days"] + 
                        xp * integer_powersystem_data["number periods"] + 
                        pos_gen[xnode][xgen]];
                }
                net_generation.push_back(net_gen_node);
            }
        }
    }

    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                load_curtailment.push_back(solution[
                    initial_position_variables[aux_name]+xnode]);
        }
    }

    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                generation_curtailment.push_back(solution[
                    initial_position_variables[aux_name]+xnode]);
        }
    }
    // Finding position slack bus
    int slack_pos;
    for (size_t i = 0; i < buses_g.size(); i++)
    {
        if (buses_g[i].first == integer_powersystem_data["slack bus"])
        {
            slack_pos = i;
            break;
        }
    }

    vector<double> angle;
    int aux_count = integer_powersystem_data["number periods"];
    int aux_count1 = integer_powersystem_data["number periods"];
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            vec angle_partial(buses_g.size() - 1);
            vec balance(buses_g.size() - 1);
            int counter = 0;
            string aux_name;
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                if (xnode != slack_pos)
                {
                    balance(counter) = power_system_datastructure[buses_g[
                        xnode].second].info_bus.active_power_demand[
                        xrd*aux_count + xp] - net_generation[
                        xrd*aux_count + xp*aux_count1 + xnode] - 
                        load_curtailment[xrd*aux_count + xp*aux_count1 + xnode] +
                        generation_curtailment[
                        xrd*aux_count + xp*aux_count1 + xnode];
                        counter++;
                }
            }
            counter = 0;
            angle_partial = inverse_mat_sustance * balance;
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                if (xnode != slack_pos)
                {
                    angle.push_back(angle_partial(counter));
                    counter++;
                }
                else angle.push_back(0.0);
            }
        }
    }
    return angle;
}

void models_cpp::get_generation_solution(vector<double> &generation,
    vector<double> &generation_cost)
{
    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
                generation.push_back(solution[
                    initial_position_variables[aux_name]+xgen] * 
                    continuous_powersystem_data["base power"]);
        }
    }
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
                generation_cost.push_back(solution[
                    initial_position_variables[aux_name]+xgen]);
        }
    }    
}

void models_cpp::get_branch_solution(vector<double> &power_flow)
{
    string aux_name;
    if (type_OPF == 1 || type_OPF == 2)
    {
        for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
        {
            for (int xp = 0; xp < 
                integer_powersystem_data["number periods"]; xp++)
            {
                for (size_t xbranch = 0; xbranch < branches_g.size(); xbranch++)
                {
                    int pos1 = -1, pos2 = -1;
                    for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                    {
                        if (power_system_datastructure[branches_g[xbranch].second].
                            info_branch.from_bus == buses_g[xnode].first)
                            pos1 = xnode;
                        else if (power_system_datastructure[branches_g[xbranch].second].
                            info_branch.to_bus == buses_g[xnode].first)
                            pos2 = xnode;
                        if (pos1 != -1 && pos2 != -1) break;
                    }
                    aux_name = "theta("+to_string(xrd)+","+to_string(xp)+")";
                    power_flow.push_back((solution[
                        initial_position_variables[aux_name]+pos1] - solution[
                        initial_position_variables[aux_name]+pos2])*
                        (1.0/power_system_datastructure[
                        branches_g[xbranch].second].info_branch.reactance[0]) * 
                        continuous_powersystem_data["base power"]);
                }
            }
        }
    }
    else if (type_OPF == 3)
    {
        vector<double> angle = calculate_angles();
        int aux_count = integer_powersystem_data["number periods"];
        int aux_count1 = integer_powersystem_data["number periods"];
        for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
        {
            for (int xp = 0; xp < 
                integer_powersystem_data["number periods"]; xp++)
            {
                for (size_t xbranch = 0; xbranch < branches_g.size(); xbranch++)
                {
                    int pos1 = -1, pos2 = -1;
                    for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                    {
                        if (power_system_datastructure[branches_g[xbranch].second].
                            info_branch.from_bus == buses_g[xnode].first)
                            pos1 = xnode;
                        else if (power_system_datastructure[branches_g[xbranch].second].
                            info_branch.to_bus == buses_g[xnode].first)
                            pos2 = xnode;
                        if (pos1 != -1 && pos2 != -1) break;
                    }
                    power_flow.push_back((angle[
                        xrd*aux_count + xp*aux_count1 + pos1] - angle[
                        xrd*aux_count + xp*aux_count1 + pos2]) *
                        (1.0/power_system_datastructure[
                        branches_g[xbranch].second].info_branch.reactance[0]) * 
                        continuous_powersystem_data["base power"]);
                }
            }
        }
    } 
    
}

void models_cpp::get_node_solution(vector<double> &angle, 
    vector<double> &generation_curtailment, vector<double> &load_curtailment)
{
    
    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                load_curtailment.push_back(solution[
                    initial_position_variables[aux_name]+xnode] * 
                    continuous_powersystem_data["base power"]);
        }
    }
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                generation_curtailment.push_back(solution[
                    initial_position_variables[aux_name]+xnode] * 
                    continuous_powersystem_data["base power"]);
        }
    }
    if (type_OPF == 1 || type_OPF == 2)
    {
        for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
        {
            for (int xp = 0; xp < 
                integer_powersystem_data["number periods"]; xp++)
            {
                aux_name = "theta("+to_string(xrd)+","+to_string(xp)+")";
                for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                    angle.push_back(solution[initial_position_variables[aux_name]+
                        xnode]);
            }
        }
    }
    else if (type_OPF == 3)
    {
        angle = calculate_angles();
    }    
}

double models_cpp::get_objective_function_nm()
{
    return objective_function;
}

// Combined Energy Tree and reduced DC OPF

void models_cpp::load_combined_energy_dc_opf_information(
    const vector< vector<int> > &LLNA, const vector<int> &CTG){
    LLNodesAfter = LLNA;
    ConnectionTreeGen = CTG;
}

void models_cpp::create_combined_energy_dc_opf_model(){
    number_variables = 0;
    number_constraints = 0;
    create_graph_database();
    create_susceptance_matrix();
    declaration_variables_em();
    declaration_variables_dc_opf();
    energy_balance();
    energy_aggregation();
    active_power_balance_ac();
    active_power_flow_limit_ac();
    active_power_generation_cost();
    energy_and_network_relation();
    release_limits_energy_tree();
    objective_function_combined_energy_dc_opf();
}

void models_cpp::create_combined_energy_dc_opf_model_v2(){
    number_variables = 0;
    number_constraints = 0;
    create_graph_database();
    create_inverse_susceptance_matrix();
    if (inverse_susceptance_completed)
    {
        declaration_variables_em();
        declaration_variables_dc_opf_v2();
        energy_balance();
        energy_aggregation();
        active_power_balance_ac_system();
        active_power_flow_limit_ac_v2();
        active_power_generation_cost();
        energy_and_network_relation();
        release_limits_energy_tree();
        objective_function_combined_energy_dc_opf();
    }
}

void models_cpp::release_limits_energy_tree(){
    string aux_name;
    for (int xnt = 0; xnt < number_trees; xnt++)
    {
        aux_name = "OutputsTree("+to_string(xnt)+")";
        for (int xrd = 0; xrd < 
            integer_powersystem_data["number representative days"]; xrd++)
        {
            colLower[initial_position_variables[aux_name] + 
                ConnectionTreeGen[xrd]] = 0;
            colUpper[initial_position_variables[aux_name] + 
                ConnectionTreeGen[xrd]] = COIN_DBL_MAX;
        }
    }
}

void models_cpp::energy_and_network_relation(){
    string aux_name;
    int counter_cons = 0;
    int pos_hydro = -1;
    for (size_t i = 0; i < generators.size(); i++)
    {
        if (generators[i].type == "hydro")
        {
            pos_hydro = i;
            break;
        }
    }
    for (int xnt = 0; xnt < number_trees; xnt++)
    {
        counter_cons = 0;
        for (int xrd = 0; xrd < 
            integer_powersystem_data["number representative days"]; xrd++)
        {
            aux_name = "OutputsTree("+to_string(xnt)+")";
            columns.push_back(initial_position_variables[aux_name] + 
                ConnectionTreeGen[xrd]);
            rows.push_back(number_constraints + counter_cons);
            elements.push_back(1.0);
            for (int xnh = 0; xnh < 
                integer_powersystem_data["number periods"]; xnh++)
            {
                aux_name = "P_g("+to_string(xrd)+","+to_string(xnh)+")";
                columns.push_back(initial_position_variables[aux_name] + 
                pos_hydro);
                rows.push_back(number_constraints + counter_cons);
                elements.push_back(-continuous_powersystem_data["base power"]*
                    continuous_powersystem_data["total hours period"]);
            }
            counter_cons++;
        }
        pos_hydro++;
        aux_name = "ENR("+to_string(xnt)+")";
        add_constraints(aux_name, 
            integer_powersystem_data["number representative days"]);
    }

    // Limits constraint
    for (int xnt = 0; xnt < number_trees; xnt++)
    {
        for (int xrd = 0; xrd < 
            integer_powersystem_data["number representative days"]; xrd++)
        {
            rowLower.push_back(0.0);
            rowUpper.push_back(0.0);
        }
    }
}

void models_cpp::objective_function_combined_energy_dc_opf(){
    /*
        This function constructs the objective function
    */

    vector<double> weight_agg = weight_nodes;
    vector<double> OFaux (
        integer_powersystem_data["number representative days"], 1.0);
    int xp = 0;
    for (int xnt = 0; xnt < nodes_tree; xnt++)
    {
        int aux = LLNodesAfter[xnt][0];
        if (aux != 0)
            for (int xna = aux; xna <= LLNodesAfter[xnt][1]; xna++)
                weight_agg[xna] *= weight_agg[xnt];
        else{
            OFaux[xp] = weight_agg[xnt] * 
                continuous_powersystem_data["total hours period"];
            xp++;
        }
    }
    
    objective.resize(number_variables, 0.0);
    string aux_name;
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            // Penalise load curtailment
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                objective[initial_position_variables[aux_name] + xnode] = 
                    10000000*OFaux[xrd];
            // Penalise generation curtailment
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
                objective[initial_position_variables[aux_name] + xnode] = 
                    10000000*OFaux[xrd];
            // maximum and minimum generation cost
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
                objective[initial_position_variables[aux_name] + xgen] = 
                    OFaux[xrd];
        }
    }
}

void models_cpp::run_combined_energy_dc_opf_r1(){

    type_OPF = 1;

    create_combined_energy_dc_opf_model();

    // matrix data
    CoinBigIndex num_elements = rows.size();

    CoinPackedMatrix matrix(true, rows.data(), columns.data(), elements.data(), 
        num_elements);
    
    // load problem
    model.loadProblem(matrix, colLower.data(), colUpper.data(), objective.data(),
                       rowLower.data(), rowUpper.data());
        
    /* +1 to minimize, -1 to maximize, and 0 to ignore */
    model.setOptimizationDirection(1);
    /*
    Amount of print out:
        0 - none
        1 - just final
        2 - just factorizations
        3 - as 2 plus a bit more
        4 - verbose
    */
    model.setLogLevel(1);

    // Solve
    model.primal();

    solution = model.primalColumnSolution();
    objective_function = model.objectiveValue();
}

void models_cpp::run_iterative_combined_energy_dc_opf(){
    /* This function calls all functions to create the model and run the DC
    OPF with the reduced model and in an iterative way. The non-binding constraints
    (power flow limits) are introduced iteratively in CLP and the problem is 
    reoptimised using the dual simplex method */
    
    type_OPF = 2;

    create_combined_energy_dc_opf_model();
    solve_iterative_models();
}

void models_cpp::run_iterative_combined_energy_dc_opf_v2(){
    /* This function calls all functions to create the model and run the DC
    OPF with the second reduced model and in an iterative way. The non-binding constraints
    (power flow limits) are introduced iteratively in CLP and the problem is 
    reoptimised using the dual simplex method */

    type_OPF = 3;

    create_combined_energy_dc_opf_model_v2();
    if (inverse_susceptance_completed)
        solve_iterative_models();
}

double models_cpp::get_objective_function_combined_energy_dc_opf_r1()
{
    return objective_function;
}
