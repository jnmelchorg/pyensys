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
            const vector[double] &b_pwl, bool is_active);
        void set_integer_data_power_system(string name, int value);
        void run_reduced_dc_opf();

cdef extern from "energy_models.cpp":
    pass