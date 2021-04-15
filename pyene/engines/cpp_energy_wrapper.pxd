# Function method to interface the energy model in c++ with python

# Created by Dr. Jose Nicolas Melchor Gutierrez

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

#cdef extern from "energy_models.h":
#    cdef cppclass models_cpp:
#        models_cpp() except +
#        void add_bus(vector[double] &P, int n, string t)
#        void add_branch(vector[double] &r, vector[double] &x, 
#            vector[double] &Pmax, int fr, int to, int n, string t)
#        void add_generator(vector[double] &Pmax, 
#            vector[double] &Pmin, int bn, int n, string t,
#            double fc, double vc, vector[double] &a_pwl, 
#            vector[double] &b_pwl, bool is_active)
#        void set_integer_data_power_system(string name, int value)
#        void set_continuous_data_power_system(string name, double value)
#        void run_reduced_dc_opf()
#        void get_generation_solution(vector[double] &generation,
#            vector[double] &generation_cost)
#        void get_branch_solution(vector[double] &power_flow)
#        void get_node_solution(vector[double] &angle, 
#            vector[double] &generation_curtailment, 
#            vector[double] &load_curtailment)
#        double get_objective_function_nm()
#
#        void load_energy_tree_information(int number_nodes, int number_tree, 
#            vector[vector[int] ] &LLEB,
#            vector[vector[int] ] &LLEA,
#            vector[double] &energy_intake,
#            vector[double] &energy_output,
#            vector[double] &weight_nodes)
#        void run_energy_tree()
#        void get_energy_tree_solution(vector[double] &PartialStorage, 
#            vector[double] &TotalStorage, vector[double] &InputsTree,
#            vector[double] &OutputsTree)
#        double get_objective_function_em()
#
#        void load_combined_energy_dc_opf_information(
#            vector[ vector[int] ] &LLNA, vector[int] &CTG)
#        void run_combined_energy_dc_opf_r1()
#        double get_objective_function_combined_energy_dc_opf_r1()
#
#        void run_iterative_reduced_dc_opf()
#
#        void run_iterative_reduced_dc_opf_v2()
#
#        void run_iterative_combined_energy_dc_opf()
#
#        void run_iterative_combined_energy_dc_opf_v2()
#        
#
#cdef extern from "energy_models.cpp":
#    pass

cdef extern from "energy_models2.h":
    cdef cppclass models:
        models_cpp() except +
        void create_parameter()
        void load_double(const string& na, const double& val, const bool& is_vector);
        void load_integer(const string& na, const int& val, const bool& is_vector);
        void load_bool(const string& na, const bool& val, const bool& is_vector);
        void load_string(const string& na, const string& val, const bool& is_vector);
        void set_parameter(const string& typ);
        int update_parameter();

        void initialise();
        void evaluate();

        void return_outputs(vector[double]& values, vector[int]& starts, vector[ vector[ vector[ string] ] ]& characteristics);

cdef extern from "energy_models2.cpp":
    pass