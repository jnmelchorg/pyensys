// This class header declares all energy models
#ifndef REDUCED_DC_OPF_H
#define REDUCED_DC_OPF_H


#include <vector>
#include <string>
#include <utility>
#include <map>
#include <iomanip> 
#include "external files/include/graph/adjacency_list.hpp"
#include "ClpSimplex.hpp"
#include "CoinHelperFunctions.hpp"

using namespace std;


// Data structures - power system information
struct bus
{
    vector<double> active_power_demand;
    int number;
    string type;
};

struct branch
{
    vector<double> reactance;
    vector<double> resistance;
    vector<double> maximum_P_flow;
    int from_bus;
    int to_bus;
    int number;
    string type;
};

struct generator
{
    vector<double> minimum_generation;
    vector<double> maximum_generation;
    int bus_number;
    int number; // Generator number
    double fixed_cost;      // cost of equipment, land, financing, project 
                            // management, grid connection, and construction 
                            // of the power plant.
    double variable_cost;   // consist of fuel cost, operation and maintenance 
                            // expenses and carbon dioxide emission charges, 
                            // if applicable.
    string type;            // Indicates the type of generator that is stored
                            // conv, hydro, RES
    pair< vector<double>, vector<double> > piecewise;
    bool is_active;
};

// Data structures to relate graph's vertices with structures of the
// power system

struct vertex
{
    string type;    // Indicates the type of information stored. The available
                    // types are: bus, branch, generator
    bus info_bus;
    branch info_branch;
    generator infor_generator;
};

struct edges
{
    int any_info;
};

// Definition of c++ types
typedef boost::adjacency_list<  boost::vecS, boost::vecS, 
                                boost::undirectedS,
                                vertex,
                                edges
                                > GraphType;

typedef boost::graph_traits<GraphType>::adjacency_iterator AdjacencyIterator;

class models_cpp{

    protected:

        // Variables for all models
        vector<int> columns;
        vector<int> rows;
        vector<double> elements;
        vector<double> objective;
        vector<double> rowLower;
        vector<double> rowUpper;
        vector<double> colLower;
        vector<double> colUpper;
        int number_variables;
        int number_constraints;
        ClpSimplex  model;
        map<string, int> initial_position_variables;
        map<string, int> initial_position_constraints;

        // Functions for all models
        void add_variables(string name, int number);
        void add_constraints(string name, int number);

        // Energy tree formulation

        int nodes_tree;
        int number_trees;
        vector<vector<int> > LLEB;   // Link list energy balance
        vector<vector<int> > LLEA;   // Link list energy aggregation
        vector<double> energy_intake;   // entry of energy at each node of the tree
        vector<double> energy_output;   // output of energy at each node of the tree
        vector<double> weight_nodes;   // Weight of node (number of days, weeks, etc.)

        void create_energy_tree_model();

        void declaration_variables_em();

        void energy_balance();

        void energy_aggregation();

        void objective_function_em();

        // Reduced DC OPF v1
        // All power system components are treated as nodes of a graph

        vector< pair<int, int> > buses_g;       // * Nodes in the graph that  
                                                // contain the buses
                                                // first is the bus number
                                                // second is the node number in the graph
        vector< pair<int, int> > branches_g;    // * Nodes in the graph that 
                                                // contain the branches
                                                // first is the branch number
                                                // second is the branch number in the graph
        vector< pair<int, int> > generators_g;  // * Nodes in the graph that 
                                                //contain the generators
                                                // first is the generator number
                                                // second is the generator number in the graph
        
        vector<branch> equivalent_branches;  

        vector< vector<double> > susceptance_matrix;
        GraphType power_system_datastructure;

        int nodes_graph; // Number of nodes in graph

        // General data power system
        map<string, int> integer_powersystem_data;

        map<string, double> continuous_powersystem_data;

        void create_graph_database();

        void create_susceptance_matrix();

        void create_reduced_dc_opf_model();

        void declaration_variables_dc_opf();

        void active_power_balance_ac();

        void active_power_flow_limit_ac();

        void active_power_generation_cost();

        void objective_function_nm();

        // Combined energy tree and reduced DC OPF v1
        vector< vector<int> > LLNodesAfter;
        vector<int> ConnectionTreeGen;  // Connections
                        // between the energy model and the network
                        // model. This parameters connects the inputs 
                        // of each tree with the outputs of its 
                        // related hydro generator

        void create_combined_energy_dc_opf_model();

        void release_limits_energy_tree();

        void energy_and_network_relation();

        void objective_function_combined_energy_dc_opf();
    
    public:

        models_cpp();
        ~models_cpp();

        // Energy tree formulation

        void load_energy_tree_information(int number_nodes, int number_tree, 
            const vector<vector<int> > &LLEB,
            const vector<vector<int> > &LLEA,
            const vector<double> &energy_intake,
            const vector<double> &energy_output,
            const vector<double> &weight_nodes);
        
        void run_energy_tree();
        void get_energy_tree_solution(vector<double> &PartialStorage, 
            vector<double> &TotalStorage, vector<double> &InputsTree,
            vector<double> &OutputsTree);
        double get_objective_function_em();

        // Reduced DC OPF v1

        vector<bus> buses;
        vector<branch> branches;        
        vector<generator> generators;

        void add_bus(const vector<double> &P, int n, string t);
        void add_branch(const vector<double> &r, const vector<double> &x, 
            const vector<double> &Pmax, int from, int to, int n, string t);
        void add_generator(const vector<double> &Pmax, 
            const vector<double> &Pmin, int bn, int n, string t,
            double fc, double vc, const vector<double> &a_pwl, 
            const vector<double> &b_pwl, bool is_active);
        void set_integer_data_power_system(string name, int value);
        void set_continuous_data_power_system(string name, double value);
        void run_reduced_dc_opf();
        void get_generation_solution(vector<double> &generation,
            vector<double> &generation_cost);
        void get_branch_solution(vector<double> &power_flow);
        void get_node_solution(vector<double> &angle, 
            vector<double> &generation_curtailment,
            vector<double> &load_curtailment);
        double get_objective_function_nm();

        // Combined energy tree and reduced DC OPF v1
        void load_combined_energy_dc_opf_information(
            const vector< vector<int> > &LLNA, const vector<int> &CTG);

        void run_combined_energy_dc_opf_r1();
        double get_objective_function_combined_energy_dc_opf_r1();
};

#endif