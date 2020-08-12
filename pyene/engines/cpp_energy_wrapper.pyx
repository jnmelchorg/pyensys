# distutils: language = c++

from cpp_energy_wrapper cimport reduced_dc_opf

from libcpp.string cimport string
from libcpp.vector cimport vector

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
        fixed_cost, variable_cost, a_pwl, b_pwl):
        cdef vector[double] Pmax = P_max
        cdef vector[double] Pmin = P_min
        cdef int bn = bus_number
        cdef int n = number
        cdef string t = type_gen.encode('utf-8')
        cdef double fc = fixed_cost
        cdef double vc = variable_cost
        cdef vector[double] a = a_pwl
        cdef vector[double] b = b_pwl
        self.cpp_opf.add_generator(Pmax, Pmin, bn, n, t, fc, vc, a, b)
    
    cpdef set_integer_data_power_system_cpp(self, str name, value):
        cdef string n = name.encode('utf-8')
        cdef int val = value
        self.cpp_opf.set_integer_data_power_system(n, val)
    
    cpdef run_reduced_dc_opf_cpp(self):
        self.cpp_opf.run_reduced_dc_opf();