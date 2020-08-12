// This class header declares all energy models
#ifndef REDUCED_DC_OPF_H
#define REDUCED_DC_OPF_H


#include <vector>
#include <string>
#include <utility>
#include <map>
#include <boost/graph/adjacency_list.hpp>
#include "/home/tesla/coinbrew/dist/include/coin/ClpSimplex.hpp"

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

class reduced_dc_opf{

    private:
        // All power system components are treated as nodes of a graph

        vector< pair<int, int> > buses_g;       // * Nodes in the graph that  
                                                // contain the buses
        vector< pair<int, int> > branches_g;    // * Nodes in the graph that 
                                                // contain the branches
        vector< pair<int, int> > generators_g;  // * Nodes in the graph that 
                                                //contain the generators

        vector< vector<double> > susceptance_matrix;
        GraphType power_system_datastructure;

        int nodes_graph; // Number of nodes in graph

        map<string, int> integer_powersystem_data;

        // Elements Clp model
        vector<CoinBigIndex> start;
        vector<int> length;
        vector<int> rows;
        vector<double> elements;

        ClpSimplex  model;

        void create_graph_database();

        void create_susceptance_matrix();

        void create_reduced_dc_opf_model();

        void active_power_balance();


    public:

        reduced_dc_opf();
        ~reduced_dc_opf();

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
        void run_reduced_dc_opf();
};

#endif