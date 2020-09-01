# Function method to interface the energy model in c++ with python

# Created by Dr. Jose Nicolas Melchor Gutierrez

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "energy_models.h":
    cdef cppclass reduced_dc_opf:
        reduced_dc_opf() except +
        void add_bus(const vector[double] &P, int n, string t)
        void add_branch(const vector[double] &r, const vector[double] &x, 
            const vector[double] &Pmax, int fr, int to, int n, string t)
        void add_generator(const vector[double] &Pmax, 
            const vector[double] &Pmin, int bn, int n, string t,
            double fc, double vc, const vector[double] &a_pwl, 
            const vector[double] &b_pwl, bool is_active)
        void set_integer_data_power_system(string name, int value)
        void set_continuous_data_power_system(string name, double value)
        void run_reduced_dc_opf()
        void get_generation_solution(vector[double] &generation,
            vector[double] &generation_curtailment, 
            vector[double] &generation_cost)
        void get_branch_solution(vector[double] &power_flow)
        void get_node_solution(vector[double] &angle, 
            vector[double] &load_curtailment)
        double get_objective_function_nm()
    
    cdef cppclass energy_tree:
        energy_tree() except +
        void load_energy_tree_information(int number_nodes, int number_tree, 
            const vector[vector[int] ] &LLEB,
            const vector[vector[int] ] &LLEA,
            const vector[double] &energy_intake,
            const vector[double] &energy_output,
            const vector[double] &weight_nodes)
        void run_energy_tree()
        void get_energy_tree_solution(vector[double] &PartialStorage, 
            vector[double] &TotalStorage, vector[double] &InputsTree,
            vector[double] &OutputsTree)
        double get_objective_function_em()
    
    cdef cppclass combined_energy_dc_opf_r1(energy_tree, reduced_dc_opf):
        combined_energy_dc_opf_r1() except +
        void load_combined_energy_dc_opf_information(
            const vector[ vector[int] ] &LLNA, const vector[int] &CTG)
        void run_combined_energy_dc_opf_r1()
        double get_objective_function_combined_energy_dc_opf_r1()

cdef extern from "energy_models.cpp":
    pass