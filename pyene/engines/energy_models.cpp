#include "energy_models.h"
#include <vector>
#include <string>
#include <string.h>
#include <iostream>
#include <assert.h>

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
    active_power_balance();
}

void reduced_dc_opf::active_power_balance(){
    /*
    This file constructs the active power balance constraint
    */

    vector< vector<int> > pos_gen(buses_g.size());
    int num_LC_var = 0; // Calculate the number of load curtailment variables
    for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
    {
        AdjacencyIterator ai, a_end;
        boost::tie(ai, a_end) = boost::adjacent_vertices(buses_g[xnode].second, 
            power_system_datastructure);
        int counter_var_row = 0;
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
    for (size_t xrd = 0; xrd < 
        integer_powersystem_data["number representative days"]; xrd++)
    {
        for (size_t xp = 0; xp < 
            integer_powersystem_data["number periods"]; xp++)
        {
            for (size_t xnode = 0; xnode < buses_g.size(); xnode++)
            {
                
            }
        }
    }
        
}


void reduced_dc_opf::run_reduced_dc_opf(){
    /* This function calls all functions to create the model and run the DC
    OPF with the reduced model */
    create_graph_database();
    create_susceptance_matrix();
}
