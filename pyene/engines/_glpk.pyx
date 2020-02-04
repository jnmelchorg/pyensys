from libc.stdlib cimport malloc, free
include "glpk.pxi"
inf = float('inf')


GLP_BND_TYPES = {
        'free': GLP_FR,  # free (unbounded) variable
        'lower': GLP_LO,  # variable with lower bound
        'upper': GLP_UP,  # variable with upper bound
        'bounded': GLP_DB,  # double-bounded variable
        'fixed': GLP_FX,  # fixed variable
}

GLP_DIR = {
    'min': GLP_MIN, 'minimise': GLP_MIN,
    'max': GLP_MAX, 'maximise': GLP_MAX,
}

cdef class GLPKSolver:
    cdef glp_prob* prob
    cdef glp_smcp smcp

    cdef readonly dict row_ids
    cdef readonly dict col_ids
    cdef readonly  dict stats

    def __cinit__(self):
        # create a new problem
        self.prob = glp_create_prob()
        self.row_ids = {}
        self.col_ids = {}

    def __init__(self, time_limit=None, iteration_limit=None, message_level='error'):
        self.stats = None

        # Set solver options
        glp_init_smcp(&self.smcp)
        self.smcp.msg_lev = message_levels[message_level]
        if time_limit is not None:
            self.smcp.tm_lim = time_limit  # 5 second limit
        if iteration_limit is not None:
            self.smcp.it_lim = iteration_limit

        glp_term_hook(term_hook, NULL)

    def __dealloc__(self):
        # free the problem
        glp_delete_prob(self.prob)
        
    cpdef set_dir(self, direction):
        glp_set_obj_dir(self.prob, GLP_DIR[direction])

    cpdef int add_rows(self, str name, int num):
        """Add new rows to the linear programme."""
        if name in self.row_ids:
            raise KeyError(f'Row name "{name}" already exists.')
        cdef int idx = glp_add_rows(self.prob, num)
        self.row_ids[name] = idx
        return idx

    cpdef int add_cols(self, str name, int num):
        """Add new columns to the linear programme."""
        if name in self.col_ids:
            raise KeyError(f'Column name "{name}" already exists.')
        cdef int idx = glp_add_cols(self.prob, num)
        self.col_ids[name] = idx

        for i in range(num):
            self.set_col_bnds(name, i, 'lower', 0.0, DBL_MAX)
        return idx

    cpdef set_mat_row(self, str name, int row_offset, cols, values):
        cdef int* ind
        cdef double* val

        if len(cols) != len(values):
            raise ValueError(
                f'The length of the `cols` ({len(cols)}) and `values` ({len(values)}) arguments mut be the same.')

        # Calculate the actual row in the LP to update
        cdef int row = self.row_ids[name] + row_offset

        ind = <int*>malloc((1 + len(cols)) * sizeof(int))
        val = <double*>malloc((1 + len(cols)) * sizeof(double))
        for n, c in enumerate(cols):
            ind[1+n] = 1+c
            # TODO check for finite values
            val[1+n] = values[n]
            print(row, c, n, values[n])
        glp_set_mat_row(self.prob, row, len(cols), ind, val)
        #glp_set_row_bnds(self.prob, row, GLP_FX, 0.0, 0.0)

        free(ind)
        free(val)

    cpdef load_matrix(self, int ne, iapy, japy, arpy):
        cdef int* ia
        cdef int* ja
        cdef double* ar

        ia = <int*> malloc((ne + 1) * sizeof(int))
        ja = <int*> malloc((ne + 1) * sizeof(int))
        ar = <double*> malloc((ne + 1) * sizeof(double))

        for aux1 in range(ne):
            ia[aux1 + 1] = iapy[aux1]
            ja[aux1 + 1] = japy[aux1]
            ar[aux1 + 1] = arpy[aux1]
        
        glp_load_matrix(self.prob, ne, ia, ja, ar)
        
        free(ia)
        free(ja)
        free(ar)

    cpdef set_row_bnds(self, str name, int row_offset, type, double lb, double ub):
        # TODO check for finite values here
        # Calculate the actual row in the LP to update
        cdef int row = self.row_ids[name] + row_offset
        glp_set_row_bnds(self.prob, row, GLP_BND_TYPES[type], lb, ub)

    cpdef set_col_bnds(self, str name, int col_offset, type, double lb, double ub):
        # TODO check for finite values here
        # Calculate the actual col in the LP to update
        cdef int col = self.col_ids[name] + col_offset
        glp_set_col_bnds(self.prob, col, GLP_BND_TYPES[type], lb, ub)

    cpdef set_obj_coef(self, str name, int col_offset, double coef):
        # TODO check for finite vaues here
        # Calculate the actual col in the LP to update
        cdef int col = self.col_ids[name] + col_offset
        glp_set_obj_coef(self.prob, col, coef)

    cpdef int simplex(self):
        return glp_simplex(self.prob, &self.smcp)

    cpdef int status(self):
        status = glp_get_status(self.prob)

    cpdef double get_col_prim(self, str name, int col_offset):
        # Calculate the actual col in the LP to update
        cdef int col = self.col_ids[name] + col_offset
        return glp_get_col_prim(self.prob, col)

    cpdef double get_obj_val(self):
        # retrieve objective value
        return glp_get_obj_val(self.prob)


