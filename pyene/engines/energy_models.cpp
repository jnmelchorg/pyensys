#include "energy_models.h"
#include <vector>
#include <string>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <stdlib.h> 

using namespace std;

reduced_dc_opf::reduced_dc_opf() {
    // Number of periods in a 24h period
    // e.g. 24 periods (1h per period)
    //      3 periods (8h per period)
    integer_powersystem_data["number periods"] = 1;
    // Total number of representative days defined by the user
    // e.g. January-weekdays
    //      January-weekends
    //      February-weekdays
    //      February-weekends
    // The total number of representative days is 4
    integer_powersystem_data["number representative days"] = 1;
    // Slack bus in the power system
    integer_powersystem_data["slack bus"] = 0;
};

reduced_dc_opf::~reduced_dc_opf() { };

/**** Functions to add components ****/

void reduced_dc_opf::add_bus(const vector<double> &P, int n, string t){
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

void reduced_dc_opf::add_branch(const vector<double> &r, const vector<double> &x, 
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

void reduced_dc_opf::add_generator(const vector<double> &Pmax, 
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

void reduced_dc_opf::set_integer_data_power_system(string name, int value){
    /*
    This function set any general integer data of the power system.
    The current available options are:
    - "number periods"
    - "number representative days"
    The desired option should be passes as a string and the value as an int
    */
    integer_powersystem_data[name] = value;
}

void reduced_dc_opf::create_graph_database(){
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
                boost::add_edge(buses_g[i].second, branches_g[i].second, 
                    power_system_datastructure);
            if (buses[i].number == branches[j].to_bus)
                boost::add_edge(branches_g[i].second, buses_g[i].second,
                    power_system_datastructure);
        }
        // Adding connections between nodes and generators
        for (size_t j = 0; j < generators.size(); j++)
        {
            if (buses[i].number == generators[j].bus_number)
            {
                boost::add_edge(buses_g[i].second, generators_g[i].second, 
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

void reduced_dc_opf::create_susceptance_matrix(){
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
        susceptance_matrix[pos1][pos2] = 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        susceptance_matrix[pos2][pos1] = 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        diag_vals[pos1] -= 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0];
        diag_vals[pos2] -= 1.0/power_system_datastructure[
            branches_g[i].second].info_branch.reactance[0]; 
    }
    for (size_t i = 0; i < buses_g.size(); i++)
        susceptance_matrix[i][i] = diag_vals[i];
}

void reduced_dc_opf::create_reduced_dc_opf_model(){
    number_variables_nm = 0;
    number_constraints_nm = 0;
    declaration_variables();
    active_power_balance_ac();
    active_power_flow_limit_ac();
    active_power_generation_cost();
    objective_function_nm();
}

void reduced_dc_opf::declaration_variables(){

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
            add_variables(aux_name, generators_g.size()); // Generation curtailment
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            add_variables(aux_name, generators_g.size());   // Cost generation
        }
    }
    // Declaration limit variables
    colLower.resize(number_variables_nm);
    colUpper.resize(number_variables_nm);

    int aux_count = integer_powersystem_data["number periods"];
    for (int xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (int xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            // maximum and minimum active power generation
            aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                colLower[initial_position_variables[aux_name] + xgen] = 
                    power_system_datastructure[generators_g[xgen].second].
                    infor_generator.minimum_generation[xrd*aux_count + xp];
                colUpper[initial_position_variables[aux_name] + xgen] = 
                    power_system_datastructure[generators_g[xgen].second].
                    infor_generator.maximum_generation[xrd*aux_count + xp];
            }
            // maximum and minimum voltage phase angle
            aux_name = "theta("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                if (buses_g[xnode].first != 
                    integer_powersystem_data["slack bus"])
                {
                    colLower[initial_position_variables[aux_name] + xnode] = 
                        -COIN_DBL_MAX;
                    colUpper[initial_position_variables[aux_name] + xnode] = 
                        COIN_DBL_MAX;
                }
                else{
                    colLower[initial_position_variables[aux_name] + xnode] = 0;
                    colUpper[initial_position_variables[aux_name] + xnode] = 0;
                }
            }
            // maximum and minimum load curtailment
            aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                colLower[initial_position_variables[aux_name] + xnode] = 0;
                colUpper[initial_position_variables[aux_name] + xnode] = 
                    power_system_datastructure[buses_g[xnode].second].info_bus.
                    active_power_demand[xrd*aux_count + xp];
            }
            // maximum and minimum generation curtailment
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                colLower[initial_position_variables[aux_name] + xgen] = 0;
                if (power_system_datastructure[generators_g[xgen].second].
                    infor_generator.minimum_generation[xrd*aux_count + xp] > 0)
                    colUpper[initial_position_variables[aux_name] + xgen] = 
                        power_system_datastructure[generators_g[xgen].second].
                        infor_generator.minimum_generation[xrd*aux_count + xp];
                else
                    colUpper[initial_position_variables[aux_name] + xgen] = 0;
            }
            // maximum and minimum generation cost
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
            {
                colLower[initial_position_variables[aux_name] + xgen] = 0;
                colUpper[initial_position_variables[aux_name] + xgen] = 
                        COIN_DBL_MAX;
            }
        }
    }
}

void reduced_dc_opf::active_power_balance_ac(){
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
                    rows.push_back(number_constraints_nm + xnode);
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
                        rows.push_back(number_constraints_nm + xnode);
                        columns.push_back(initial_position_variables[aux_name] +
                            xang);
                        elements.push_back(susceptance_matrix[xnode][xang]);
                    }
                }
                // Load Curtailment
                aux_name = "P_lc("+to_string(xrd)+","+to_string(xp)+")";
                rows.push_back(number_constraints_nm + xnode);
                columns.push_back(initial_position_variables[aux_name] + xnode);
                elements.push_back(1.0);
                // Generation curtailment
                aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
                for (size_t xgen = 0; xgen < pos_gen[xnode].size(); xgen++)
                {
                    rows.push_back(number_constraints_nm + xnode);
                    columns.push_back(initial_position_variables[aux_name] +
                        pos_gen[xnode][xgen]);
                    elements.push_back(-1.0);
                }
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

void reduced_dc_opf::active_power_flow_limit_ac(){
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
                rows.push_back(number_constraints_nm + xbranch);
                columns.push_back(initial_position_variables[aux_name] + pos1);
                elements.push_back(1.0);
                rows.push_back(number_constraints_nm + xbranch);
                columns.push_back(initial_position_variables[aux_name] + pos2);
                elements.push_back(-1.0);
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

void reduced_dc_opf::active_power_generation_cost(){
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
                    rows.push_back(number_constraints_nm + xpieces);
                    columns.push_back(initial_position_variables[aux_name] + xgen);
                    elements.push_back(1.0);
                    aux_name = "P_g("+to_string(xrd)+","+to_string(xp)+")";
                    rows.push_back(number_constraints_nm + xpieces);
                    columns.push_back(initial_position_variables[aux_name] + xgen);
                    elements.push_back(-aux_gen.piecewise.first[xpieces] * (
                        24.0/integer_powersystem_data["number periods"]));
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
                    rowLower.push_back(aux_gen.piecewise.second[xpieces] * (
                        24.0/integer_powersystem_data["number periods"]));
                    rowUpper.push_back(COIN_DBL_MAX);
                }
            }
        }
    }
}

void reduced_dc_opf::objective_function_nm(){
    /*
        This function constructs the objective function
    */
    objective.resize(number_variables_nm, 0.0);
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
                objective[initial_position_variables[aux_name] + xnode] = 100000;
            // Penalise generation curtailment
            aux_name = "P_gc("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
                objective[initial_position_variables[aux_name] + xgen] = 100000;
            // maximum and minimum generation cost
            aux_name = "C_g("+to_string(xrd)+","+to_string(xp)+")";
            for (size_t xgen = 0; xgen < generators_g.size(); xgen++)
                objective[initial_position_variables[aux_name] + xgen] = 1;
        }
    }
}

void reduced_dc_opf::add_variables(string name, int number)
{
    initial_position_variables.insert(pair<string, int>(name, 
        number_variables_nm));
    number_variables_nm += number;
}

void reduced_dc_opf::add_constraints(string name, int number)
{
    initial_position_constraints.insert(pair<string, int>(name, 
        number_constraints_nm));
    number_constraints_nm += number;
}

void reduced_dc_opf::run_reduced_dc_opf(){
    /* This function calls all functions to create the model and run the DC
    OPF with the reduced model */
    create_graph_database();
    create_susceptance_matrix();
    create_reduced_dc_opf_model();

    // matrix data
    CoinBigIndex num_elements = rows.size();
    
    CoinPackedMatrix matrix(true, rows.data(), columns.data(), elements.data(), 
        num_elements);

    // load problem
    model_nm.loadProblem(matrix, colLower.data(), colUpper.data(), objective.data(),
                       rowLower.data(), rowUpper.data());
    
    /* +1 to minimize, -1 to maximize, and 0 to ignore */
    model_nm.setOptimizationDirection(1);
    /*
    Amount of print out:
        0 - none
        1 - just final
        2 - just factorizations
        3 - as 2 plus a bit more
        4 - verbose
    */
    model_nm.setLogLevel(0);

    // Solve
    model_nm.primal();
}
