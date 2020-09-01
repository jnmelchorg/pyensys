# distutils: language = c++

from cpp_energy_wrapper cimport reduced_dc_opf, energy_tree, combined_energy_dc_opf_r1

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef class network_models_cpp:

    cdef reduced_dc_opf* cpp_opf

    def __cinit__(self):
        self.cpp_opf = new reduced_dc_opf()

    cpdef add_bus_cpp(self, demand, number, str type_bus):
        cdef vector[double] P = demand
        cdef int n = number
        cdef string t = type_bus.encode('utf-8')
        self.cpp_opf.add_bus(P, n, t)
    
    cpdef add_branch_cpp(self, reactance, resistance, Pmax, \
        fr, to, number_branch, type_branch):
        cdef vector[double] r = resistance
        cdef vector[double] x = reactance
        cdef vector[double] Pm = Pmax
        cdef int f = fr
        cdef int t = to
        cdef int n = number_branch
        cdef string s = type_branch.encode('utf-8')
        self.cpp_opf.add_branch(r, x, Pm, f, t, n, s)
    
    cpdef add_generator_cpp(self, P_max, P_min, bus_number, number, type_gen,\
        fixed_cost, variable_cost, a_pwl, b_pwl, active):
        cdef vector[double] Pmax = P_max
        cdef vector[double] Pmin = P_min
        cdef int bn = bus_number
        cdef int n = number
        cdef string t = type_gen.encode('utf-8')
        cdef double fc = fixed_cost
        cdef double vc = variable_cost
        cdef vector[double] a = a_pwl
        cdef vector[double] b = b_pwl
        cdef bool is_active = active
        self.cpp_opf.add_generator(Pmax, Pmin, bn, n, t, fc, vc, a, b, is_active)
    
    cpdef set_integer_data_power_system_cpp(self, str name, value):
        cdef string n = name.encode('utf-8')
        cdef int val = value
        self.cpp_opf.set_integer_data_power_system(n, val)
    
    cpdef set_continuous_data_power_system_cpp(self, str name, value):
        cdef string n = name.encode('utf-8')
        cdef double val = value
        self.cpp_opf.set_continuous_data_power_system(n, val)
    
    cpdef run_reduced_dc_opf_cpp(self):
        self.cpp_opf.run_reduced_dc_opf()
    
    cpdef get_generation_solution_cpp(self):
        cdef vector[double] generation
        cdef vector[double] generation_curtailment
        cdef vector[double] generation_cost
        self.cpp_opf.get_generation_solution(generation, generation_curtailment,
            generation_cost)
        sol_gen = generation
        sol_gen_cur = generation_curtailment
        sol_gen_cost = generation_cost
        return sol_gen, sol_gen_cur, sol_gen_cost

    cpdef get_branch_solution_cpp(self):
        cdef vector[double] power_flow
        self.cpp_opf.get_branch_solution(power_flow)
        sol_power_flows = power_flow
        return sol_power_flows
    
    cpdef get_node_solution_cpp(self):
        cdef vector[double] angle
        cdef vector[double] load_curtailment
        self.cpp_opf.get_node_solution(angle, load_curtailment)
        sol_angle = angle
        sol_load_cur = load_curtailment
        return sol_angle, sol_load_cur
    
    cpdef get_objective_function_cpp(self):
        return self.cpp_opf.get_objective_function_nm()

cdef class energy_model_cpp:
    cdef energy_tree* cpp_em

    def __cinit__(self):
        self.cpp_em = new energy_tree()

    cpdef load_energy_tree_information_cpp(self, n_nodes, n_trees, LLEB, LLEA, \
        intake, output, weight):
        cdef int number_nodes = n_nodes
        cdef int number_tree = n_trees
        cdef int aux_size = len(LLEB)
        cdef vector[vector[int] ] LLEB_cpp
        cdef vector[int] aux_vec
        for aux in range(aux_size):
            aux_vec = LLEB[aux, :]
            LLEB_cpp.push_back(aux_vec)
        aux_size = len(LLEA)
        cdef vector[vector[int] ] LLEA_cpp
        for aux in range(aux_size):
            aux_vec = LLEA[aux, :]
            LLEA_cpp.push_back(aux_vec)
        cdef vector[double] energy_intake = intake
        cdef vector[double] energy_output = output
        cdef vector[double] weight_nodes = weight
        self.cpp_em.load_energy_tree_information(number_nodes, number_tree, \
            LLEB_cpp, LLEA_cpp, energy_intake, energy_output, weight_nodes)
    
    cpdef run_energy_tree_cpp(self):
        self.cpp_em.run_energy_tree()
    
    cpdef get_energy_tree_solution_cpp(self):
        cdef vector[double] PartialStorage
        cdef vector[double] TotalStorage
        cdef vector[double] InputsTree
        cdef vector[double] OutputsTree
        self.cpp_em.get_energy_tree_solution(PartialStorage, TotalStorage, 
            InputsTree, OutputsTree)
        sol_partial = PartialStorage
        sol_total = TotalStorage
        sol_inputs = InputsTree
        sol_outputs = OutputsTree
        return sol_partial, sol_total, sol_inputs, sol_outputs
    
    cpdef get_objective_function_em_cpp(self):
        return self.cpp_em.get_objective_function_em()

cdef class combined_energy_dc_opf_r1_cpp():
    cdef combined_energy_dc_opf_r1* cpp_enm

    def __cinit__(self):
        self.cpp_enm = new combined_energy_dc_opf_r1()
    
    cpdef load_combined_energy_dc_opf_information_cpp(self, LLNAfter, CTGen):
        cdef int aux_size = len(LLNAfter)
        cdef vector[vector[int] ] LLNA_cpp
        cdef vector[int] aux_vec
        for aux in range(aux_size):
            aux_vec = LLNAfter[aux, :]
            LLNA_cpp.push_back(aux_vec)
        cdef vector[int] CTG_cpp = CTGen
        self.cpp_enm.load_combined_energy_dc_opf_information(LLNA_cpp, CTG_cpp)
    

    cpdef add_bus_cpp(self, demand, number, str type_bus):
        cdef vector[double] P = demand
        cdef int n = number
        cdef string t = type_bus.encode('utf-8')
        self.cpp_enm.add_bus(P, n, t)
    
    cpdef add_branch_cpp(self, reactance, resistance, Pmax, \
        fr, to, number_branch, type_branch):
        cdef vector[double] r = resistance
        cdef vector[double] x = reactance
        cdef vector[double] Pm = Pmax
        cdef int f = fr
        cdef int t = to
        cdef int n = number_branch
        cdef string s = type_branch.encode('utf-8')
        self.cpp_enm.add_branch(r, x, Pm, f, t, n, s)
    
    cpdef add_generator_cpp(self, P_max, P_min, bus_number, number, type_gen,\
        fixed_cost, variable_cost, a_pwl, b_pwl, active):
        cdef vector[double] Pmax = P_max
        cdef vector[double] Pmin = P_min
        cdef int bn = bus_number
        cdef int n = number
        cdef string t = type_gen.encode('utf-8')
        cdef double fc = fixed_cost
        cdef double vc = variable_cost
        cdef vector[double] a = a_pwl
        cdef vector[double] b = b_pwl
        cdef bool is_active = active
        self.cpp_enm.add_generator(Pmax, Pmin, bn, n, t, fc, vc, a, b, is_active)
    
    cpdef set_integer_data_power_system_cpp(self, str name, value):
        cdef string n = name.encode('utf-8')
        cdef int val = value
        self.cpp_enm.set_integer_data_power_system(n, val)
    
    cpdef set_continuous_data_power_system_cpp(self, str name, value):
        cdef string n = name.encode('utf-8')
        cdef double val = value
        self.cpp_opf.set_continuous_data_power_system(n, val)
    

    cpdef load_energy_tree_information_cpp(self, n_nodes, n_trees, LLEB, LLEA, \
        intake, output, weight):
        cdef int number_nodes = n_nodes
        cdef int number_tree = n_trees
        cdef int aux_size = len(LLEB)
        cdef vector[vector[int] ] LLEB_cpp
        cdef vector[int] aux_vec
        for aux in range(aux_size):
            aux_vec = LLEB[aux, :]
            LLEB_cpp.push_back(aux_vec)
        aux_size = len(LLEA)
        cdef vector[vector[int] ] LLEA_cpp
        for aux in range(aux_size):
            aux_vec = LLEA[aux, :]
            LLEA_cpp.push_back(aux_vec)
        cdef vector[double] energy_intake = intake
        cdef vector[double] energy_output = output
        cdef vector[double] weight_nodes = weight
        self.cpp_enm.load_energy_tree_information(number_nodes, number_tree, \
            LLEB_cpp, LLEA_cpp, energy_intake, energy_output, weight_nodes)

    
    cpdef run_combined_energy_dc_opf_r1_cpp(self):
        self.cpp_enm.run_combined_energy_dc_opf_r1()

    cpdef get_objective_function_combined_energy_dc_opf_r1_cpp(self):
        return self.cpp_enm.get_objective_function_combined_energy_dc_opf_r1()
    
    cpdef get_energy_tree_solution_cpp(self):
        cdef vector[double] PartialStorage
        cdef vector[double] TotalStorage
        cdef vector[double] InputsTree
        cdef vector[double] OutputsTree
        self.cpp_enm.get_energy_tree_solution(PartialStorage, TotalStorage, 
            InputsTree, OutputsTree)
        sol_partial = PartialStorage
        sol_total = TotalStorage
        sol_inputs = InputsTree
        sol_outputs = OutputsTree
        return sol_partial, sol_total, sol_inputs, sol_outputs

    cpdef get_generation_solution_cpp(self):
        cdef vector[double] generation
        cdef vector[double] generation_curtailment
        cdef vector[double] generation_cost
        self.cpp_enm.get_generation_solution(generation, generation_curtailment,
            generation_cost)
        sol_gen = generation
        sol_gen_cur = generation_curtailment
        sol_gen_cost = generation_cost
        return sol_gen, sol_gen_cur, sol_gen_cost

    cpdef get_branch_solution_cpp(self):
        cdef vector[double] power_flow
        self.cpp_enm.get_branch_solution(power_flow)
        sol_power_flows = power_flow
        return sol_power_flows
    
    cpdef get_node_solution_cpp(self):
        cdef vector[double] angle
        cdef vector[double] load_curtailment
        self.cpp_enm.get_node_solution(angle, load_curtailment)
        sol_angle = angle
        sol_load_cur = load_curtailment
        return sol_angle, sol_load_cur
