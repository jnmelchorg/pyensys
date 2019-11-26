from ._glpk import GLPKSolver
import numpy as np


class Energymodel():
    """ This class builds and solve the energy model using the gplk wrapper.

    The information of the pyeneClass is passed to this class,
    which provides the parameters for the model. Furthermore,
    the GLPKSolver class that contains the GLPK wrapper is imported """

    number_variables = 0
    number_constraints = 0

    def __init__(self, obj=None):
        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def optimisation(self):
        # Creation of model instance
        self.solver = GLPKSolver(message_level='off')       
        # Definition of minimisation problem
        self.solver.set_dir('max')
        # Definition of the mathematical formulation
        self.modeldefinition()
        ret = self.solver.simplex()
        assert ret == 0

        for i in range(self.size['Vectors']):
            print("vector %d:" %(i))
            for j in range(self.LL['NosBal']+1):
                 print("%f %f" %(self.solver.get_col_prim('cols', \
                     i * (self.LL['NosBal']+1) + j), \
                         self.solver.get_col_prim('cols', \
                     i * (self.LL['NosBal']+1) + j + self.size['Vectors'] *\
                         (self.LL['NosBal']+1))))


    def modeldefinition(self):
        """ This class method build and solve the optimisation problem,
         to be expanded with a general optimisation problem """
        self.dnvariables()                                      # Function to determine de number of variables
        self.dnconstraints()                                    # Function to determine de number of constraints
        self.solver.add_cols('cols', self.number_variables)     # Number of rows (variables) of the matrix A

        # define matrix of coeficients (matrix A)
        self.Bounds_variables()
        self.coeffmatrix()
        self.Objective_function()

    def dnvariables(self):
        """ This class method determines the number of variables """
        self.number_variables += (self.LL['NosBal']+1) * 2 \
            * self.size['Vectors']

    def dnconstraints(self):
        """ This class method determines the number of constraints """
        # Number of constrains in the energy balance
        self.number_constraints += (self.LL['NosBal']+1) * self.size['Vectors']
        self.number_constraints += (self.LL['NosAgg']+1) * self.size['Vectors'] 
        if self.LL['NosUnc'] != 0:
            self.number_constraints += (self.LL['NosUnc']+1) * self.size['Vectors']


    def coeffmatrix(self):
        """ This class method contains the functions that allow building the coefficient
        matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) to be
        # later added to glpk
        self.ia = np.empty(self.number_constraints*self.number_variables\
            , dtype=int) # Position in rows
        self.ja = np.empty(self.number_constraints*self.number_variables\
            , dtype=int) # Position in columns
        self.ar = np.empty(self.number_constraints*self.number_variables\
            , dtype=int) # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A
        self.Energybalance()
        self.Aggregation()
        if self.LL['NosUnc'] != 0:
            self.AggregationStochastic()
        self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)
 
    def Energybalance(self):
        """ This class method writes the energy balance in glpk
        
        First, it is reserved space in memory to store the energy balance constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Reserving space in glpk for energy constraints
        self.EB_row_number = self.solver.add_rows('EB', self.LL['NosBal'] * \
            self.size['Vectors'])   # Number of columns (constraints) in matrix A
                                    # for energy balance

        # Generating the matrix A for the energy contraints
        for vectors in range(self.size['Vectors']):
            for nodes in range(2, self.LL['NosBal']+2):
                # Storing the Vin variables
                self.ia[self.ne] = self.EB_row_number + (vectors * \
                    self.LL['NosBal']) + nodes - 2
                self.ja[self.ne] = (vectors * \
                    (self.LL['NosBal'] + 1)) + nodes
                self.ar[self.ne] = 1
                # Storing the Vout variables
                self.ne += 1
                self.ia[self.ne] = self.EB_row_number + (vectors * \
                    self.LL['NosBal']) + nodes - 2
                self.ar[self.ne] = -1
                if(self.p['LLTS1'][nodes - 1, 1] == 0):
                    self.ja[self.ne] = (vectors * \
                        (self.LL['NosBal'] + 1)) + self.p['LLTS1'][nodes - 1, 0] + 1
                elif(self.p['LLTS1'][nodes - 1, 1] == 1):
                    self.ja[self.ne] = (vectors * \
                        (self.LL['NosBal'] + 1)) + (self.size['Vectors'] * \
                        (self.LL['NosBal'] + 1)) + self.p['LLTS1'][nodes - 1, 0] + 1
                self.ne += 1

        # Defining the limits for the energy constraints
        for vectors in range(self.size['Vectors']):
            for nodes in range(1, self.LL['NosBal']+1):
                self.solver.set_row_bnds('EB', (vectors * self.LL['NosBal']) + nodes - 1, 'fixed', \
                    self.Weight['In'][nodes, vectors] - self.Weight['Out'][nodes, vectors], \
                    self.Weight['In'][nodes, vectors] - self.Weight['Out'][nodes, vectors])

        # For verification
        # TODO: include it in pytest
        # for i in range(self.ne):
        #       print("%d %d %d" %(self.ia[i], self.ja[i], self.ar[i]))
        # for vectors in range(self.size['Vectors']):
        #     for nodes in range(1, self.LL['NosBal']+1):
        #         print("%f" %(self.Weight['In'][nodes, vectors] - self.Weight['Out'][nodes, vectors]))            
        # import sys
        # sys.exit('hasta aqui')

    def Aggregation(self):
        """ This class method writes the aggregation constraints in glpk
        
        First, it is reserved space in memory to store the aggregation constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Reserving space in glpk for aggregation constraints
        self.Agg_row_number = self.solver.add_rows('Agg', self.LL['NosAgg'] * \
            self.size['Vectors'])   # Number of columns (constraints) in matrix A
                                    # for aggregation        
        nep = self.ne # For verification, TODO remove in future versions
        # Generating the matrix A for the aggregation contraints
        for vectors in range(self.size['Vectors']):
            for nodes in range(2, self.LL['NosAgg']+2):
                # Storing the Vout variables
                self.ia[self.ne] = self.Agg_row_number + (vectors * \
                    self.LL['NosAgg']) + nodes - 2
                self.ja[self.ne] = (vectors * \
                    (self.LL['NosBal'] + 1)) + nodes + \
                    (self.size['Vectors'] * (self.LL['NosBal'] + 1))
                self.ar[self.ne] = 1
                # Storing Vin or Vout variables
                self.ne += 1
                self.ia[self.ne] = self.Agg_row_number + (vectors * \
                    self.LL['NosBal']) + nodes - 2
                self.ar[self.ne] = -self.p['WghtFull'][self.p['LLTS2']\
                    [nodes - 1, 0]]
                if(self.p['LLTS2'][nodes - 1, 2] == 0):
                    self.ja[self.ne] = (vectors * \
                        (self.LL['NosBal'] + 1)) + self.p['LLTS2'][nodes - 1, 1]\
                            + 1
                elif(self.p['LLTS2'][nodes - 1, 2] == 1):
                    self.ja[self.ne] = (vectors * \
                        (self.LL['NosBal'] + 1)) + (self.size['Vectors'] * \
                        (self.LL['NosBal'] + 1)) + self.p['LLTS2'][nodes - 1, 1] + 1
                # Storing Vin or Vout variables
                if(1 - self.p['WghtFull'][self.p['LLTS2'][nodes - 1, 0]] != 0):
                    self.ne += 1
                    self.ia[self.ne] = self.Agg_row_number + (vectors * \
                        self.LL['NosBal']) + nodes - 2
                    self.ar[self.ne] = -(1 - self.p['WghtFull'][self.p['LLTS2'][nodes - 1, 0]])
                    if(self.p['LLTS1'][self.p['LLTS2'][nodes - 1, 0], 1] == 0):
                        self.ja[self.ne] = (vectors * \
                            (self.LL['NosBal'] + 1)) + self.p['LLTS1'][self.p['LLTS2']\
                            [nodes - 1, 0], 0] + 1
                    elif(self.p['LLTS1'][self.p['LLTS2'][nodes - 1, 0], 1] == 1):
                        self.ja[self.ne] = (vectors * \
                            (self.LL['NosBal'] + 1)) + (self.size['Vectors'] * \
                            (self.LL['NosBal'] + 1)) + self.p['LLTS1']\
                                [self.p['LLTS2'][nodes - 1, 0], 0] + 1
                self.ne += 1

        # Defining the limits for the aggregation constraints
        for vectors in range(self.size['Vectors']):
            for nodes in range(1, self.LL['NosAgg']+1):
                self.solver.set_row_bnds('Agg', (vectors * self.LL['NosAgg']) + nodes - 1, 'fixed', \
                    0.0, 0.0)

        # For verification
        # TODO: include it in pytest
        # for i in range(self.ne):
        #       print("%d %d %d" %(self.ia[i], self.ja[i], self.ar[i]))
        # for vectors in range(self.size['Vectors']):
        #     for nodes in range(1, self.LL['NosBal']+1):
        #         print("%f" %(self.Weight['In'][nodes, vectors] - self.Weight['Out'][nodes, vectors]))            
        # import sys
        # sys.exit('hasta aqui')

    def AggregationStochastic(self):
        """ This class method writes the aggregation constraints for stochastic scenarios in glpk
        
        First, it is reserved space in memory to store the aggregation constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Reserving space in glpk for aggregation constraints
        self.Agg_Sto_row_number = self.solver.add_rows('AggStoch', self.LL['NosAgg'] * \
            self.size['Vectors'])   # Number of columns (constraints) in matrix A
                                    # for aggregation        
        nep = self.ne
        # Generating the matrix A for the aggregation contraints
        # TODO review this constraint
        for vectors in range(self.size['Vectors']):
            for nodes in range(2, self.LL['NosUnc']+2): # TODO, does it start from position 2??
                # Storing the first variable of each constraint
                self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                    self.LL['NosAgg']) + nodes - 2
                self.ja[self.ne] = (vectors * \
                    (self.LL['NosBal'] + 1)) + \
                    (self.size['Vectors'] * (self.LL['NosBal'] + 1)) + \
                    self.p['LLTS3'][nodes - 1, 0] + 1
                self.ar[self.ne] = 1
                # Storing the second variable of each constraint
                if(1-self.p['WghtFull'][self.p['LLTS3'][nodes - 1, 0]] != 0):
                    self.ne += 1
                    self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                        self.LL['NosBal']) + nodes - 2
                    self.ar[self.ne] = -(1-self.p['WghtFull'][self.p['LLTS3'][nodes - 1, 0]])
                    if(self.p['LLTS1'][self.p['LLTS3'][nodes - 1, 0], 1] == 0):
                        self.ja[self.ne] = (vectors * \
                            (self.LL['NosBal'] + 1)) + self.p['LLTS1'][self.p['LLTS3'][nodes - 1, 0], 0] + 1
                    elif(self.p['LLTS1'][self.p['LLTS3'][nodes - 1, 0], 1] == 1):
                        self.ja[self.ne] = (vectors * \
                            (self.LL['NosBal'] + 1)) + (self.size['Vectors'] * \
                            (self.LL['NosBal'] + 1)) + self.p['LLTS1'][self.p['LLTS3'][nodes - 1, 0], 0] + 1
                # Storing the third variable
                self.ne += 1
                self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                    self.LL['NosBal']) + nodes - 2
                self.ar[self.ne] = -(self.p['WghtFull'][self.p['LLTS3'][nodes - 1, 0]] * \
                    -self.p['LLTS3'][nodes - 1, 2])
                self.ja[self.ne] = (vectors * \
                    (self.LL['NosBal'] + 1)) + self.p['LLTS3'][nodes - 1, 0] + 1
                # Storing variables in the summation
                for aux1 in range(self.p['LLTS3'][nodes - 1, 2] + 1):
                    self.ne += 1
                    self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                        self.LL['NosBal']) + nodes - 2
                    self.ar[self.ne] = -(self.p['WghtFull'][self.p['LLTS3'][nodes - 1, 0]] * \
                        -self.p['LLTS3'][nodes - 1, 2])
                    self.ja[self.ne] = (vectors * \
                            (self.LL['NosBal'] + 1)) + (self.size['Vectors'] * \
                            (self.LL['NosBal'] + 1)) + self.p['LLTS3'][nodes, 1] + aux1 + 1
                self.ne += 1

                    

        # Defining the limits for the aggregation constraints
        for vectors in range(self.size['Vectors']):
            for nodes in range(1, self.LL['NosUnc']+1):
                self.solver.set_row_bnds('AggStoch', (vectors *  self.LL['NosBal']) + nodes - 1, 'fixed', \
                    0.0, 0.0)


    def Bounds_variables(self):
        """ This class method defines the bounds for the variables in glpk """
        counter = 0
        for aux1 in range(1, 3):
            for aux2 in range(1, self.size['Vectors'] + 1):
                for aux3 in range(1, self.LL['NosBal'] + 2):
                    if (aux3 == 1):
                        self.solver.set_col_bnds('cols', counter, 'fixed', 0.0, 0.0)
                    else:
                        self.solver.set_col_bnds('cols', counter, 'lower', 0.0, 100000)
                    counter += 1


    def Objective_function(self):
        """ This class method defines the cost coefficients for the
         objective function in glpk
         
        A dummy objective function is created for the problem """

        self.solver.set_obj_coef('cols', 1, 2)
        self.solver.set_obj_coef('cols', 2, -1)
        self.solver.set_obj_coef('cols', self.size['Vectors'] * \
                        (self.LL['NosBal'] + 1) + 2, -1)





        # For verification
        # TODO: include it in pytest
        # for i in range(nep, self.ne):
        #       print("%d %d %d" %(self.ia[i], self.ja[i], self.ar[i]))
        # for vectors in range(self.size['Vectors']):
        #     for nodes in range(1, self.LL['NosBal']+1):
        #         print("%f" %(self.Weight['In'][nodes, vectors] - self.Weight['Out'][nodes, vectors]))            
        # import sys
        # sys.exit('hasta aqui')


 
        # m.vEIn = self.Weight['In']
        # m.vEOut = self.Weight['Out']

        # ''' Adding pyomo constraints '''
        # # Initialisation conditions
        # m.cZSoC = Constraint(range(2), self.s['Vec'], rule=self.cZSoC_rule)

