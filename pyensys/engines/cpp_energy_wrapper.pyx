# distutils: language = c++
# cython: profile=True

from pyensys.engines.cpp_energy_wrapper cimport models

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

import copy
    
#cdef class models_cpp_clp():
#    cdef models_cpp* cpp_mod
#
#    def __cinit__(self):
#        self.cpp_mod = new models_cpp()
#    
#    cpdef load_combined_energy_dc_opf_information_cpp(self, LLNAfter, CTGen):
#        cdef int aux_size = len(LLNAfter)
#        cdef vector[vector[int] ] LLNA_cpp
#        cdef vector[int] aux_vec
#        for aux in range(aux_size):
#            aux_vec = LLNAfter[aux, :]
#            LLNA_cpp.push_back(aux_vec)
#        cdef vector[int] CTG_cpp = CTGen
#        self.cpp_mod.load_combined_energy_dc_opf_information(LLNA_cpp, CTG_cpp)
#    
#
#    cpdef add_bus_cpp(self, demand, number, str type_bus):
#        cdef vector[double] P = demand
#        cdef int n = number
#        cdef string t = type_bus.encode('utf-8')
#        self.cpp_mod.add_bus(P, n, t)
#    
#    cpdef add_branch_cpp(self, reactance, resistance, Pmax, \
#        fr, to, number_branch, type_branch):
#        cdef vector[double] r = resistance
#        cdef vector[double] x = reactance
#        cdef vector[double] Pm = Pmax
#        cdef int f = fr
#        cdef int t = to
#        cdef int n = number_branch
#        cdef string s = type_branch.encode('utf-8')
#        self.cpp_mod.add_branch(r, x, Pm, f, t, n, s)
#    
#    cpdef add_generator_cpp(self, P_max, P_min, bus_number, number, type_gen,\
#        fixed_cost, variable_cost, a_pwl, b_pwl, active):
#        cdef vector[double] Pmax = P_max
#        cdef vector[double] Pmin = P_min
#        cdef int bn = bus_number
#        cdef int n = number
#        cdef string t = type_gen.encode('utf-8')
#        cdef double fc = fixed_cost
#        cdef double vc = variable_cost
#        cdef vector[double] a = a_pwl
#        cdef vector[double] b = b_pwl
#        cdef bool is_active = active
#        self.cpp_mod.add_generator(Pmax, Pmin, bn, n, t, fc, vc, a, b, is_active)
#    
#    cpdef set_integer_data_power_system_cpp(self, str name, value):
#        cdef string n = name.encode('utf-8')
#        cdef int val = value
#        self.cpp_mod.set_integer_data_power_system(n, val)
#    
#    cpdef set_continuous_data_power_system_cpp(self, str name, value):
#        cdef string n = name.encode('utf-8')
#        cdef double val = value
#        self.cpp_mod.set_continuous_data_power_system(n, val)
#    
#
#    cpdef load_energy_tree_information_cpp(self, n_nodes, n_trees, LLEB, LLEA, \
#        intake, output, weight):
#        cdef int number_nodes = n_nodes
#        cdef int number_tree = n_trees
#        cdef int aux_size = len(LLEB)
#        cdef vector[vector[int] ] LLEB_cpp
#        cdef vector[int] aux_vec
#        for aux in range(aux_size):
#            aux_vec = LLEB[aux, :]
#            LLEB_cpp.push_back(aux_vec)
#        aux_size = len(LLEA)
#        cdef vector[vector[int] ] LLEA_cpp
#        for aux in range(aux_size):
#            aux_vec = LLEA[aux, :]
#            LLEA_cpp.push_back(aux_vec)
#        cdef vector[double] energy_intake = intake
#        cdef vector[double] energy_output = output
#        cdef vector[double] weight_nodes = weight
#        self.cpp_mod.load_energy_tree_information(number_nodes, number_tree, \
#            LLEB_cpp, LLEA_cpp, energy_intake, energy_output, weight_nodes)
#
#
#    cpdef run_reduced_dc_opf_cpp(self):
#        self.cpp_mod.run_reduced_dc_opf()
#    
#    cpdef run_iterative_reduced_dc_opf_cpp(self):
#        self.cpp_mod.run_iterative_reduced_dc_opf()
#    
#    cpdef run_iterative_reduced_dc_opf_v2_cpp(self):
#        self.cpp_mod.run_iterative_reduced_dc_opf_v2()
#
#    cpdef run_energy_tree_cpp(self):
#        self.cpp_mod.run_energy_tree()
#    
#    cpdef run_combined_energy_dc_opf_r1_cpp(self):
#        self.cpp_mod.run_combined_energy_dc_opf_r1()
#    
#    cpdef run_iterative_combined_energy_dc_opf_cpp(self):
#        self.cpp_mod.run_iterative_combined_energy_dc_opf()
#    
#    cpdef run_iterative_combined_energy_dc_opf_v2_cpp(self):
#        self.cpp_mod.run_iterative_combined_energy_dc_opf_v2()
#
#
#
#    cpdef get_objective_function_cpp(self):
#        return self.cpp_mod.get_objective_function_nm()
#
#    cpdef get_objective_function_em_cpp(self):
#        return self.cpp_mod.get_objective_function_em()
#
#    cpdef get_objective_function_combined_energy_dc_opf_r1_cpp(self):
#        return self.cpp_mod.get_objective_function_combined_energy_dc_opf_r1()
#
#
#
#    cpdef get_energy_tree_solution_cpp(self):
#        cdef vector[double] PartialStorage
#        cdef vector[double] TotalStorage
#        cdef vector[double] InputsTree
#        cdef vector[double] OutputsTree
#        self.cpp_mod.get_energy_tree_solution(PartialStorage, TotalStorage, 
#            InputsTree, OutputsTree)
#        sol_partial = PartialStorage
#        sol_total = TotalStorage
#        sol_inputs = InputsTree
#        sol_outputs = OutputsTree
#        return sol_partial, sol_total, sol_inputs, sol_outputs
#
#    cpdef get_generation_solution_cpp(self):
#        cdef vector[double] generation
#        cdef vector[double] generation_cost
#        self.cpp_mod.get_generation_solution(generation, generation_cost)
#        sol_gen = generation
#        sol_gen_cost = generation_cost
#        return sol_gen, sol_gen_cost
#
#    cpdef get_branch_solution_cpp(self):
#        cdef vector[double] power_flow
#        self.cpp_mod.get_branch_solution(power_flow)
#        sol_power_flows = power_flow
#        return sol_power_flows
#    
#    cpdef get_node_solution_cpp(self):
#        cdef vector[double] angle
#        cdef vector[double] generation_curtailment
#        cdef vector[double] load_curtailment
#        self.cpp_mod.get_node_solution(angle, generation_curtailment, load_curtailment)
#        sol_angle = angle
#        sol_gen_cur = generation_curtailment
#        sol_load_cur = load_curtailment
#        return sol_angle, sol_load_cur, sol_gen_cur


cdef class models_cpp():
    cdef models* cpp_mod

    def __cinit__(self):
        self.cpp_mod = new models()
    
    cpdef create_parameter(self):
        self.cpp_mod.create_parameter()
    
    cpdef load_value(self, type_value, na, val):
        cdef string name = na
        cdef int i_value
        cdef double d_value
        cdef bool b_value
        cdef string s_value
        if type_value == "integer":
            i_value = val
            self.cpp_mod.load_integer(name, i_value, False)
        elif type_value == "double":
            d_value = val
            self.cpp_mod.load_double(name, d_value, False)
        elif type_value == "bool":
            b_value = val
            self.cpp_mod.load_bool(name, b_value, False)
        elif type_value == "string":
            s_value = val.encode('utf-8')
            self.cpp_mod.load_string(name, s_value, False)
        elif type_value == "v_integer":
            for v in val:
                i_value = v
                self.cpp_mod.load_integer(name, i_value, True)
        elif type_value == "v_double":
            for v in val:
                d_value = v
                self.cpp_mod.load_double(name, d_value, True)
        elif type_value == "v_bool":
            for v in val:
                b_value = v
                self.cpp_mod.load_bool(name, b_value, True)
        elif type_value == "v_string":
            for v in val:
                s_value = v.encode('utf-8')
                self.cpp_mod.load_string(name, s_value, True)
        else:
            print("data type *{}* is not valid".format(type_value))
    
    cpdef set_parameter(self, type_var):
        cdef string s_value = type_var
        self.cpp_mod.set_parameter(s_value)

    cpdef initialise(self):
        self.cpp_mod.initialise()
    
    cpdef evaluate(self):
        self.cpp_mod.evaluate()
    
    cpdef return_outputs(self):
        cdef vector[double] values
        cdef vector[int] starts
        cdef vector[ vector[ vector[ string] ] ] characteristics
        self.cpp_mod.return_outputs(values, starts, characteristics)
        return values, starts, characteristics
    
    cpdef update_parameter(self):
        cdef int code
        code = self.cpp_mod.update_parameter()
        return code

    cpdef get_MOEA_variables(self):
        cdef vector[string] IDs
        cdef vector[string] names
        cdef vector[double] min_bnd
        cdef vector[double] max_bnd
        self.cpp_mod.get_MOEA_variables(IDs, names, min_bnd, max_bnd)
        return IDs, names, min_bnd, max_bnd

    cpdef get_moea_objectives(self):
        cdef vector[string] names
        self.cpp_mod.get_moea_objectives(names)
        return names

