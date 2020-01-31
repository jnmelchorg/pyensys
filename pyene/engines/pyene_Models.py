from ._glpk import GLPKSolver
import numpy as np
import sys
import math


class Energymodel():
    """ This class builds and solve the energy model using the gplk wrapper.

    The information of the pyeneClass is passed to this class,
    which provides the parameters for the model. Furthermore,
    the GLPKSolver class that contains the GLPK wrapper is imported """

    number_variablesEM = 0
    number_constraintsEM = 0

    def __init__(self, obj=None):
        """
        Parameters
        ----------
        obj : Energy object
            Information of the energy tree
        """
        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def optimisationEM(self):
        """ This class method solve the optimisation problem """
        # TODO to be expanded with a general optimisation problem       
        # Creation of model instance
        self.solver = GLPKSolver(message_level='off')       
        # Definition of minimisation problem
        self.solver.set_dir('max')
        # Definition of the mathematical formulation
        self.modeldefinitionEM()
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


    def modeldefinitionEM(self):
        """ This class method build and solve the optimisation problem,
         to be expanded with a general optimisation problem """
        # TODO: create functions such as posvariables and variables in a 
        # similar way than the network model
        self.dnvariablesEM()    # Function to determine de number of variables
        self.dnconstraintsEM()  # Function to determine de number of constraints
        # Number of columns (variables) of the matrix A
        self.solver.add_cols('cols', self.number_variablesEM)

        # define matrix of coeficients (matrix A)
        self.Bounds_variablesEM()
        self.coeffmatrixEM()
        self.Objective_functionEM()

    def dnvariablesEM(self):
        """ This class method determines the number of variables """
        self.number_variablesEM += (self.LL['NosBal']+1) * 2 \
            * self.size['Vectors']

    def dnconstraintsEM(self):
        """ This class method determines the number of constraints """
        # Number of constrains in the energy balance
        self.number_constraintsEM += (self.LL['NosBal']+1) * self.size['Vectors']
        self.number_constraintsEM += (self.LL['NosAgg']+1) * self.size['Vectors'] 
        if self.LL['NosUnc'] != 0:
            self.number_constraintsEM += (self.LL['NosUnc']+1) \
                * self.size['Vectors']


    def coeffmatrixEM(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) to be
        # later added to glpk
        self.ia = np.empty(self.number_constraintsEM*self.number_variablesEM\
            , dtype=int) # Position in rows
        self.ja = np.empty(self.number_constraintsEM*self.number_variablesEM\
            , dtype=int) # Position in columns
        self.ar = np.empty(self.number_constraintsEM*self.number_variablesEM\
            , dtype=int) # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A
        self.Energybalance()
        self.Aggregation()
        if self.LL['NosUnc'] != 0:
            self.AggregationStochastic()
        self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)
 
    def Energybalance(self):
        """ This class method writes the energy balance in glpk
        
        First, it is reserved space in memory to store the energy balance 
        constraints.
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


    def Bounds_variablesEM(self):
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


    def Objective_functionEM(self):
        """ This class method defines the cost coefficients for the
         objective function in glpk
         
        A dummy objective function is created for the problem """

        self.solver.set_obj_coef('cols', 1, 2)
        self.solver.set_obj_coef('cols', 2, -1)
        self.solver.set_obj_coef('cols', self.size['Vectors'] * \
                        (self.LL['NosBal'] + 1) + 2, -1)


class Networkmodel():
    """ This class builds and solve the network model(NM) using the gplk wrapper.

    The information of the pyeneClass is passed to this class,
    which provides the parameters for the model. Furthermore,
    the GLPKSolver class that contains the GLPK wrapper is imported """

    number_variablesED = 0
    number_constraintsED = 0

    def __init__(self, obj=None):
        """
        Parameters
        ----------
        obj : Network object
            Information of the power system
        """
        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def optimisationNM(self):
        """ This class method solve the optimisation problem """
        # Creation of model instance
        self.solver = GLPKSolver(message_level='all')       
        # Definition of minimisation problem
        self.solver.set_dir('min')
        # Definition of the mathematical formulation
        self.EconomicDispatchModel()
        ret = self.solver.simplex()
        assert ret == 0


        for i in self.NM.connections['set']:
            print('Case %d :' %(i))
            print('')
            print('Generation:')
            for k in range(len(self.NM.Gen.Conv)):
                for j in range(self.NM.settings['NoTime']):
                    print("%f" %(self.solver.get_col_prim(\
                        str(self.thermalgenerators[i, j][0]), k) * \
                            self.NM.ENetwork.get_Base()), end = ' ')
                print('')
            for k in range(len(self.NM.Gen.RES)):
                for j in range(self.NM.settings['NoTime']):                
                    print("%f" %(self.solver.get_col_prim(\
                        str(self.RESgenerators[i, j][0]), k) * \
                            self.NM.ENetwork.get_Base()), end = ' ')
                print('')
            for k in range(len(self.NM.Gen.Hydro)):
                for j in range(self.NM.settings['NoTime']):
                    print("%f" %(self.solver.get_col_prim(\
                        str(self.Hydrogenerators[i, j][0]), k) * \
                            self.NM.ENetwork.get_Base()), end = ' ')
                print('')
            print('')
            if self.NM.pumps['Number'] > 0:
                print('Pumps:')
                for k in range(self.NM.pumps['Number']):
                    for j in range(self.NM.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.pumps[i, j][0]), k) * \
                                self.NM.ENetwork.get_Base()), end = ' ')
                    print('')
                print('')
            print('LC:')
            for j in range(self.NM.settings['NoTime']):
                print("%f" %(self.solver.get_col_prim(\
                            str(self.loadcurtailmentsystem[i, j][0]), 0) * \
                                self.NM.ENetwork.get_Base()), end = ' ')
            print('\n\n')
            if len(self.NM.Gen.Conv) > 0:
                print('Thermal Generation cost:')
                for k in range(len(self.NM.Gen.Conv)):
                    for j in range(self.NM.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.thermalCG[i, j][0]), k)), end = ' ')
                    print('')
                print('')
            if len(self.NM.Gen.RES) > 0:
                print('RES Generation cost:')
                for k in range(len(self.NM.Gen.RES)):
                    for j in range(self.NM.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.RESCG[i, j][0]), k)), end = ' ')
                    print('')
                print('')
            if len(self.NM.Gen.Hydro) > 0:
                print('Hydro Generation cost:')
                for k in range(len(self.NM.Gen.Hydro)):
                    for j in range(self.NM.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.HydroCG[i, j][0]), k)), end = ' ')
                    print('')
                print('')
        print('')

        # for i in range(self.size['Vectors']):
        #     print("vector %d:" %(i))
        #     for j in range(self.LL['NosBal']+1):
        #          print("%f %f" %(self.solver.get_col_prim('cols', \
        #              i * (self.LL['NosBal']+1) + j), \
        #                  self.solver.get_col_prim('cols', \
        #              i * (self.LL['NosBal']+1) + j + self.size['Vectors'] *\
        #                  (self.LL['NosBal']+1))))


    def EconomicDispatchModel(self):
        """ This class method builds the optimisation model
        for the economic dispatch problem """
        self.dnvariablesED()    # Function to determine de number of variables
        self.dnconstraintsED()  # Function to determine de number of constraints

        # define matrix of coeficients (matrix A)
        self.variablesED()
        self.coeffmatrixED()
        self.Objective_functionED()

    def dnvariablesED(self):
        """ This class method determines the number of variables for the 
        economic dispatch problem """
        # TODO: Create a variable for size last tree nodes
        # len(self.connections['set'])
        # TODO: Further analysis of energy storage variables and constraints
        # Active power generation variables
        self.number_variablesED += len(self.NM.connections['set']) \
            * self.NM.Gen.get_NoGen() * self.NM.settings['NoTime']
        # Generation cost variables
        self.number_variablesED += len(self.NM.connections['set']) \
            * self.NM.Gen.get_NoGen() * self.NM.settings['NoTime']
        # Active power storage variables
        self.number_variablesED += len(self.NM.connections['set']) \
            * self.NM.Storage['Number'] * self.NM.settings['NoTime']
        # Pumps variables
        self.number_variablesED += len(self.NM.connections['set']) \
            * self.NM.pumps['Number'] * self.NM.settings['NoTime']
        # Load curtailment variables
        self.number_variablesED += len(self.NM.connections['set']) \
            * self.NM.settings['NoTime']

    def dnconstraintsED(self):
        """ This class method determines the number of constraints for the 
        economic dispatch problem """
        self.number_constraintsED += len(self.NM.connections['set']) * \
            self.NM.settings['NoTime']     # Constraints for power balance 
                                        # of whole power system
        self.number_constraintsED += len(self.NM.connections['set']) \
            * self.NM.Gen.get_NoGen() * self.NM.settings['NoTime'] * \
                self.NM.Gen.get_NoPieces() # Constraints 
                                        # for the piecewise linearisation
                                        # of the quadratic generation cost
        self.number_constraintsED += len(self.NM.connections['set']) \
            * self.NM.Gen.get_NoGen() * self.NM.settings['NoTime'] # Constraints 
                                    # for the generation ramps

    def coeffmatrixED(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) to be
        # later added to glpk
        self.ia = np.empty(math.ceil(self.number_constraintsED * \
            self.number_variablesED / 3), dtype=int) # Position in rows
        self.ja = np.empty(math.ceil(self.number_constraintsED * \
            self.number_variablesED / 3), dtype=int) # Position in columns
        self.ar = np.empty(math.ceil(self.number_constraintsED * \
            self.number_variablesED / 3), dtype=float) # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A
        
        self.constraintsED()
        self.activepowerbalancesystem()
        self.piecewiselinearisationcost()
        self.generationrampsconstraints()

        self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)

    # Variables ED

    def PosvariablesED(self):
        """ This class method creates the vector that stores the positions of 
        variables for the ED problem """

        if len(self.NM.Gen.Conv) > 0:
            self.thermalgenerators = np.empty(\
                (len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of thermal generators' variables in matrix A (rows)
                # for each period and each tree node
        if len(self.NM.Gen.RES) > 0:
            self.RESgenerators = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of RES generators' variables in matrix A (rows)
                # for each period and each tree node
        if len(self.NM.Gen.Hydro) > 0:        
            self.Hydrogenerators = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of Hydroelectric generators' variables in matrix A (rows)
                # for each period and each tree node
        # TODO: Change this with a flag for batteries
        if self.NM.Storage['Number'] > 0:
            self.ESS = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of Energy Storage Systems' variables in matrix A (rows)
                # for each period and each tree node
        # TODO: Change this with a flag for pumps
        if self.NM.pumps['Number'] > 0:
            self.pumps = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of pumps' variables in matrix A (rows)
                # for each period and each tree node
        if len(self.NM.Gen.Conv) > 0:
            self.thermalCG = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of thermal generation cost variables in matrix A (rows)
                # for each period and each tree node
        if len(self.NM.Gen.RES) > 0:
            self.RESCG = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of RES generation cost variables in matrix A (rows)
                # for each period and each tree node
        if len(self.NM.Gen.Hydro) > 0:
            self.HydroCG = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                # of Hydroelectric generation cost variables in matrix A (rows)
                # for each period and each tree node
        self.loadcurtailmentsystem = np.empty((len(self.NM.connections['set']),\
            self.NM.settings['NoTime']),\
            dtype=[('napos', 'U20'),('nupos', 'i4')]) # Start position
            # in matrix A (rows) of variables
            # for load curtailment in the system for each tree node

    def variablesED(self):
        """ This class method defines the variables and their limits for the
        economic dispatch problem """
        self.PosvariablesED()
        
        # Reserving space in glpk for ED variables
        for i in self.NM.connections['set']:
            for j in range(self.NM.settings['NoTime']):
                # Generation variables
                if len(self.NM.Gen.Conv) > 0:
                    self.thermalgenerators[i, j] = (\
                        'ThermalGen'+str(i)+str(j),\
                        self.solver.add_cols('ThermalGen'+str(i)+str(j),\
                        len(self.NM.Gen.Conv)))
                if len(self.NM.Gen.RES) > 0:
                    self.RESgenerators[i, j] = (\
                        'RESGen'+str(i)+str(j),\
                        self.solver.add_cols('RESGen'+str(i)+str(j),\
                        len(self.NM.Gen.RES)))
                if len(self.NM.Gen.Hydro) > 0:
                    self.Hydrogenerators[i, j] = (\
                        'HydroGen'+str(i)+str(j),\
                        self.solver.add_cols('HydroGen'+str(i)+str(j),\
                        len(self.NM.Gen.Hydro)))
                # Generation cost variables
                if len(self.NM.Gen.Conv) > 0:
                    self.thermalCG[i, j] = ('ThermalCG'+str(i)+str(j),\
                        self.solver.add_cols('ThermalCG'+str(i)+str(j),\
                        len(self.NM.Gen.Conv)))
                if len(self.NM.Gen.RES) > 0:
                    self.RESCG[i, j] = ('RESCG'+str(i)+str(j),\
                        self.solver.add_cols('RESCG'+str(i)+str(j),\
                        len(self.NM.Gen.RES)))
                if len(self.NM.Gen.Hydro) > 0:
                    self.HydroCG[i, j] = ('HydroCG'+str(i)+str(j),\
                        self.solver.add_cols('HydroCG'+str(i)+str(j),\
                        len(self.NM.Gen.Hydro)))
                # TODO: Change this with a flag for batteries
                if self.NM.Storage['Number'] > 0:
                    self.ESS[i, j] = ('ESS'+str(i)+str(j),\
                        self.solver.add_cols('ESS'+str(i)+str(j),\
                        self.NM.Storage['Number']))
                # TODO: Change this with a flag for pumps
                if self.NM.pumps['Number'] > 0:
                    self.pumps[i, j] = ('Pumps'+str(i)+str(j),\
                        self.solver.add_cols('Pumps'+str(i)+str(j),\
                        self.NM.pumps['Number']))
                self.loadcurtailmentsystem[i, j] = ('LCS'+str(i)+str(j),\
                    self.solver.add_cols('LCS'+str(i)+str(j), 1))

        # Defining the limits of the variables
        for i in self.NM.connections['set']:
            for j in range(self.NM.settings['NoTime']):
                # Limits for the thermal generators
                if len(self.NM.Gen.Conv) > 0:
                    for k in range(len(self.NM.Gen.Conv)):
                        self.solver.set_col_bnds(\
                            str(self.thermalgenerators[i, j][0]), k,\
                            'bounded', self.NM.Gen.Conv[k].get_Min(),\
                            self.NM.Gen.Conv[k].get_Max())
                # Limits for the RES generators
                if len(self.NM.Gen.RES) > 0:
                    for k in range(len(self.NM.Gen.RES)):
                        self.solver.set_col_bnds(\
                            str(self.RESgenerators[i, j][0]), k,\
                            'bounded', self.NM.Gen.RES[k].get_Min(),\
                            self.NM.scenarios['RES']\
                                [self.NM.resScenario[k][i]+j] * \
                                self.NM.RES['Max'][k])

                # Limits for the Hydroelectric generators
                if len(self.NM.Gen.Hydro) > 0:
                    for k in range(len(self.NM.Gen.Hydro)):
                        self.solver.set_col_bnds(\
                            str(self.Hydrogenerators[i, j][0]), k,\
                            'bounded', self.NM.Gen.Hydro[k].get_Min(),\
                            self.NM.Gen.Hydro[k].get_Max())
                # TODO: Modify information of storage, e.g. m.sNSto
                # if self.NM.Storage['Number'] > 0:
                if self.NM.pumps['Number'] > 0:
                    for k in range(self.NM.pumps['Number']):
                        self.solver.set_col_bnds(str(self.pumps[i, j][0]), k,\
                            'bounded', 0,\
                            self.NM.pumps['Max'][k]/self.NM.ENetwork.get_Base())

    # Constraints ED

    def posconstraintsED(self):
            """ This class method creates the vectors that store the positions of 
            contraints for the ED problem """
            # Creating the matrices to store the position of constraints in
            # matrix A
            self.powerbalance = np.empty((len(self.NM.connections['set']),\
                self.NM.settings['NoTime']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                        # of active power balance constraints (rows) 
                        # for each tree node
            if len(self.NM.Gen.Conv) > 0:
                self.thermalpiecewisecost = \
                    np.empty((len(self.NM.connections['set']),\
                    self.NM.settings['NoTime'], len(self.NM.Gen.Conv)),\
                        dtype=[('napos', 'U20'), ('nupos', 'i4')]) # Start 
                        # position of piecewise linearisation constraints 
                        # (rows) for each tree node, for each period and 
                        # for each thermal generator
            if len(self.NM.Gen.RES) > 0:
                self.RESpiecewisecost = \
                    np.empty((len(self.NM.connections['set']),\
                    self.NM.settings['NoTime'], len(self.NM.Gen.RES)),\
                        dtype=[('napos', 'U20'), ('nupos', 'i4')]) # Start 
                        # position of piecewise linearisation constraints 
                        # (rows) for each tree node, for each period and 
                        # for each RES generator
            if len(self.NM.Gen.Hydro) > 0:
                self.Hydropiecewisecost = \
                    np.empty((len(self.NM.connections['set']),\
                    self.NM.settings['NoTime'], len(self.NM.Gen.Hydro)),\
                        dtype=[('napos', 'U20'), ('nupos', 'i4')]) # Start 
                        # position of piecewise linearisation constraints 
                        # (rows) for each tree node, for each period and 
                        # for each Hydro generator
            if len(self.NM.Gen.Conv) > 0:
                self.thermalgenerationramps = \
                    np.empty((len(self.NM.connections['set']),\
                    self.NM.settings['NoTime'] - 1),\
                        dtype=[('napos', 'U20'), ('nupos', 'i4')]) # Start 
                        # position of thermal generation ramps constraints 
                        # (rows) for each tree node, for each period and for 
                        # each thermal generator
            if len(self.NM.Gen.Hydro) > 0:
                self.Hydrogenerationramps = \
                    np.empty((len(self.NM.connections['set']),\
                    self.NM.settings['NoTime'] - 1),\
                        dtype=[('napos', 'U20'), ('nupos', 'i4')]) # Start 
                        # position of Hydroelectrical generation ramps constraints
                        # (rows) for each tree node, for each period and for 
                        # each hydroelectrical generator
            
    def constraintsED(self):
        """ This class method reserves the space in glpk for the constraints of
        the economic dispatch problem """

        self.posconstraintsED()

        for i in self.NM.connections['set']:
            for j in range(self.NM.settings['NoTime']):
                self.powerbalance[i, j] = ('PB'+str(i)+str(j),\
                    self.solver.add_rows('PB'+str(i)+str(j), 1))  # Number of 
                        # columns (constraints) in matrix A for the active 
                        # power balance constraints fo each period and each 
                        # tree node
                if len(self.NM.Gen.Conv) > 0:
                    for k in range(len(self.NM.Gen.Conv)):
                        self.thermalpiecewisecost[i, j, k] =\
                            ('ThermalPWC'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                                'ThermalPWC'+str(i)+str(j)+str(k), \
                                self.NM.Gen.Conv[k].get_NoPieces()))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each thermal generator
                if len(self.NM.Gen.RES) > 0:
                    for k in range(len(self.NM.Gen.RES)):
                        self.RESpiecewisecost[i, j, k] =\
                            ('RESPWC'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                                'RESPWC'+str(i)+str(j)+str(k), \
                                self.NM.Gen.RES[k].get_NoPieces()))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each RES generator
                if len(self.NM.Gen.Hydro) > 0:
                    for k in range(len(self.NM.Gen.Hydro)):
                        self.Hydropiecewisecost[i, j, k] =\
                            ('HydroPWC'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                                'HydroPWC'+str(i)+str(j)+str(k), \
                                self.NM.Gen.Hydro[k].get_NoPieces()))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each Hydro generator
                if j > 0:
                    if len(self.NM.Gen.Conv) > 0:
                        self.thermalgenerationramps[i, j - 1] = \
                            ('ThermalGR'+str(i)+str(j),\
                            self.solver.add_rows('ThermalGR'+str(i)+str(j),\
                                len(self.NM.Gen.Conv)))  # Number of 
                                # columns (constraints) in matrix A for the 
                                # generation ramps constraints for each 
                                # period, for each tree node and for each 
                                # thermal generator
                    if len(self.NM.Gen.Hydro) > 0:
                        self.Hydrogenerationramps[i, j - 1] = \
                            ('HydroGR'+str(i)+str(j),\
                            self.solver.add_rows('HydroGR'+str(i)+str(j),\
                                len(self.NM.Gen.Hydro)))  # Number of 
                                # columns (constraints) in matrix A for the 
                                # generation ramps constraints for each 
                                # period, for each tree node and for each 
                                # thermal generator

    def activepowerbalancesystem(self):
        """ This class method writes the power balance constraint in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the active power balance constraints
        for i in self.NM.connections['set']:
            for j in range(self.NM.settings['NoTime']):
            # Storing the thermal generation variables
                if len(self.NM.Gen.Conv) > 0:
                    for k in range(len(self.NM.Gen.Conv)):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = \
                            self.thermalgenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
            # Storing the RES generation variables
                if len(self.NM.Gen.RES) > 0:
                    for k in range(len(self.NM.Gen.RES)):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = \
                            self.RESgenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
            # Storing the Hydroelectric generation variables
                if len(self.NM.Gen.Hydro) > 0:
                    for k in range(len(self.NM.Gen.Hydro)):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = \
                            self.Hydrogenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
            # Storing variables for ESS
            # TODO: Modify the constraint for the first period
                if self.NM.Storage['Number'] > 0:
                    if j > 0: # Start only after the first period
                        for k in range(self.NM.Storage['Number']):
                            self.ia[self.ne] = self.powerbalance[i, j][1]
                            self.ja[self.ne] = self.ESS[i, j][1] + k
                            self.ar[self.ne] = self.NM.Storage['Efficiency'][k] \
                                / self.NM.scenarios['Weights'][j - 1]
                            self.ne += 1
                        for k in range(self.NM.Storage['Number']):
                            self.ia[self.ne] = self.powerbalance[i, j][1]
                            self.ja[self.ne] = self.ESS[i, j - 1][1] + k
                            self.ar[self.ne] = \
                                -self.NM.Storage['Efficiency'][k] \
                                / self.NM.scenarios['Weights'][j - 1]
                            self.ne += 1
            # Storing the variables for load curtailment
                self.ia[self.ne] = self.powerbalance[i, j][1]
                self.ja[self.ne] = self.loadcurtailmentsystem[i, j][1]
                self.ar[self.ne] = 1.0
                self.ne += 1
            # Storing the variables for pumps
                if self.NM.pumps['Number'] > 0:
                    for k in range(self.NM.pumps['Number']):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = self.pumps[i, j][1] + k
                        self.ar[self.ne] = -1.0
                        self.ne += 1
            # Defining the resources (b) for the constraints
                totaldemand = 0                
                # TODO: Change the inputs of losses and demand scenarios
                # for parameters
                if self.NM.scenarios['NoDem'] == 0:
                    if self.NM.settings['Loss'] is None:
                        for k in range(self.NM.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.NM.busData[k] * \
                                self.NM.scenarios['Demand']\
                                    [self.NM.busScenario[k][i]]
                    else:
                        for k in range(self.NM.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.NM.busData[k] * \
                                self.NM.scenarios['Demand']\
                                    [self.NM.busScenario[k][i]] * \
                                (1 + self.NM.settings['Loss'])
                else:
                    if self.NM.settings['Loss'] is None:
                        for k in range(self.NM.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.NM.busData[k] * \
                                self.NM.scenarios['Demand']\
                                    [j+self.NM.busScenario[k][i]]
                    else:
                        for k in range(self.NM.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.NM.busData[k] * \
                                self.NM.scenarios['Demand'] \
                                    [j+self.NM.busScenario[k][i]] * \
                                (1 + self.NM.settings['Loss'])
                self.solver.set_row_bnds(str(self.powerbalance[i, j][0]), 0,\
                    'fixed', totaldemand, totaldemand)

    def piecewiselinearisationcost(self):
        """ This class method writes the piecewise linearisarion of
        the generation cost in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the piecewise linearisation constraints of
        # the generation cost
        for i in self.NM.connections['set']:
            for j in range(self.NM.settings['NoTime']):
                if len(self.NM.Gen.Conv) > 0:
                    for k in range(len(self.NM.Gen.Conv)):
                        for l in range(self.NM.Gen.Conv[k].get_NoPieces()):
                        # Storing the generation cost variables
                            self.ia[self.ne] = \
                                self.thermalpiecewisecost[i, j, k][1] + l
                            self.ja[self.ne] = self.thermalCG[i, j][1] + k
                            self.ar[self.ne] = 1.0
                            self.ne += 1
                        # Storing the generation variables
                            self.ia[self.ne] = \
                                self.thermalpiecewisecost[i, j, k][1] + l
                            self.ja[self.ne] = \
                                self.thermalgenerators[i, j][1] + k
                            self.ar[self.ne] = \
                                -self.NM.scenarios['Weights'][j] * \
                                self.NM.Gen.Conv[k].cost['LCost'][l][0]
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.thermalpiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.NM.scenarios['Weights'][j] * \
                                self.NM.Gen.Conv[k].cost['LCost'][l][1], 0)

                if len(self.NM.Gen.RES) > 0:
                    for k in range(len(self.NM.Gen.RES)):
                        for l in range(self.NM.Gen.RES[k].get_NoPieces()):
                        # Storing the generation cost variables
                            self.ia[self.ne] = \
                                self.RESpiecewisecost[i, j, k][1] + l
                            self.ja[self.ne] = self.RESCG[i, j][1] + k
                            self.ar[self.ne] = 1.0
                            self.ne += 1
                        # Storing the generation variables
                            self.ia[self.ne] = \
                                self.RESpiecewisecost[i, j, k][1] + l
                            self.ja[self.ne] = \
                                self.RESgenerators[i, j][1] + k
                            self.ar[self.ne] = \
                                -self.NM.scenarios['Weights'][j] * \
                                self.NM.Gen.RES[k].cost['LCost'][l][0]
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.RESpiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.NM.scenarios['Weights'][j] * \
                                self.NM.Gen.RES[k].cost['LCost'][l][1], 0)

                if len(self.NM.Gen.Hydro) > 0:
                    for k in range(len(self.NM.Gen.Hydro)):
                        for l in range(self.NM.Gen.Hydro[k].get_NoPieces()):
                        # Storing the generation cost variables
                            self.ia[self.ne] = \
                                self.Hydropiecewisecost[i, j, k][1] + l
                            self.ja[self.ne] = self.HydroCG[i, j][1] + k
                            self.ar[self.ne] = 1.0
                            self.ne += 1
                        # Storing the generation variables
                            self.ia[self.ne] = \
                                self.Hydropiecewisecost[i, j, k][1] + l
                            self.ja[self.ne] = \
                                self.Hydrogenerators[i, j][1] + k
                            self.ar[self.ne] = \
                                -self.NM.scenarios['Weights'][j] * \
                                self.NM.Gen.Hydro[k].cost['LCost'][l][0]
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.Hydropiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.NM.scenarios['Weights'][j] * \
                                self.NM.Gen.Hydro[k].cost['LCost'][l][1], 0)

    def generationrampsconstraints(self):
        """ This class method writes the constraints for the generation ramps
        in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the generation ramps constraints
        for i in self.NM.connections['set']:
            for j in range(1, self.NM.settings['NoTime']):
                if len(self.NM.Gen.Conv) > 0:
                    for k in range(len(self.NM.Gen.Conv)):
                    # Storing the generation variables for current period
                        self.ia[self.ne] = \
                            self.thermalgenerationramps[i, j - 1][1] + k
                        self.ja[self.ne] = \
                            self.thermalgenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
                    # Storing the generation variables for previous period
                        self.ia[self.ne] = \
                            self.thermalgenerationramps[i, j - 1][1] + k
                        self.ja[self.ne] = \
                            self.thermalgenerators[i, j - 1][1] + k
                        self.ar[self.ne] = -1.0
                        self.ne += 1
                    # Defining the resources (b) for the constraints
                        self.solver.set_row_bnds(\
                            str(self.thermalgenerationramps[i, j - 1][0]),\
                            k, 'bounded', -self.NM.Gen.Conv[k].data['Ramp'],\
                            self.NM.Gen.Conv[k].data['Ramp'])
                if len(self.NM.Gen.Hydro) > 0:
                    for k in range(len(self.NM.Gen.Hydro)):
                    # Storing the generation variables for current period
                        self.ia[self.ne] = \
                            self.Hydrogenerationramps[i, j - 1][1] + k
                        self.ja[self.ne] = \
                            self.Hydrogenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
                    # Storing the generation variables for previous period
                        self.ia[self.ne] = \
                            self.Hydrogenerationramps[i, j - 1][1] + k
                        self.ja[self.ne] = \
                            self.Hydrogenerators[i, j - 1][1] + k
                        self.ar[self.ne] = -1.0
                        self.ne += 1
                    # Defining the resources (b) for the constraints
                        self.solver.set_row_bnds(\
                            str(self.Hydrogenerationramps[i, j - 1][0]),\
                            k, 'bounded', -self.NM.Gen.Hydro[k].data['Ramp'],\
                            self.NM.Gen.Hydro[k].data['Ramp'])

    # Objective function ED

    def Objective_functionED(self):
        """ This class method defines the objective function of the economic
        dispatch in glpk """

        # Calculating the aggregated weights for the last nodes in the tree
        # TODO: explain the aggregated weights better
        WghtAgg = 0 + self.EM.p['WghtFull']
        OFaux = np.ones(len(self.NM.connections['set']), dtype=float)
        xp = 0
        for xn in range(self.EM.LL['NosBal']+1):
            aux = self.EM.tree['After'][xn][0]
            if aux != 0:
                for xb in range(aux, self.EM.tree['After'][xn][1] + 1):
                    WghtAgg[xb] *= WghtAgg[xn]
            else:
                OFaux[xp] = WghtAgg[xn]
                xp += 1

        for i in self.NM.connections['set']:
            for j in range(self.NM.settings['NoTime']):
            # Cost for conventional generation    
                if len(self.NM.Gen.Conv) > 0: 
                    for k in range(len(self.NM.Gen.Conv)):
                        self.solver.set_obj_coef(\
                            str(self.thermalCG[i, j][0]),\
                            k, OFaux[i] * self.NM.scenarios['Weights'][j])
            # Cost for RES generation    
                if len(self.NM.Gen.RES) > 0: 
                    for k in range(len(self.NM.Gen.RES)):
                        self.solver.set_obj_coef(\
                            str(self.RESCG[i, j][0]),\
                            k, OFaux[i] * self.NM.scenarios['Weights'][j])
            # Cost for Hydroelectric generation    
                if len(self.NM.Gen.Hydro) > 0: 
                    for k in range(len(self.NM.Gen.Hydro)):
                        self.solver.set_obj_coef(\
                            str(self.HydroCG[i, j][0]),\
                            k, OFaux[i] * self.NM.scenarios['Weights'][j])
            # Punitive cost for load curtailment
                self.solver.set_obj_coef(\
                            str(self.loadcurtailmentsystem[i, j][0]),\
                            0, OFaux[i] * self.NM.scenarios['Weights'][j] \
                                * self.Penalty)
            # Operation cost of pumps
                if self.NM.pumps['Number'] > 0:
                    for k in range(self.NM.pumps['Number']):
                        self.solver.set_obj_coef(\
                            str(self.pumps[i, j][0]),\
                            k, -OFaux[i] * self.NM.scenarios['Weights'][j] \
                                * self.NM.ENetwork.get_Base() \
                                    * self.NM.pumps['Value'][k])


class EnergyandNetwork(Energymodel, Networkmodel):
    """ This class builds and solve the energy and network models(NM) 
    using the gplk wrapper.

    The information of the pyeneClass is passed to this class,
    which provides the parameters for the model. Furthermore,
    the GLPKSolver class that contains the GLPK wrapper is imported """

    number_variablesENM = 0
    number_constraintsENM = 0

    def __init__(self, obj1=None, obj2=None):
        """
        Parameters
        ----------
        obj1 : Energy object
            Information of the energy tree
        obj2 : Network object
            Information of the power system
        """
        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))
        Energymodel.__init__(self, obj1)
        Networkmodel.__init__(self, obj2)

        def optimisationENM(self):
            """ This class method solve the optimisation problem """
            # Creation of model instance
            self.solver = GLPKSolver(message_level='all')       
            # Definition of minimisation problem
            self.solver.set_dir('min')
            # Definition of the mathematical formulation
            self.EnergyandEconomicDispatchModels()
            ret = self.solver.simplex()
            assert ret == 0

        def EnergyandEconomicDispatchModels(self):
            """ This class method builds the optimisation model
            for the energy and economic dispatch problem """
            # Function to determine de number of variables in the energy model
            self.dnvariablesEM()
            # Function to determine de number of constraints in the energy 
            # model
            self.dnconstraintsEM()            
            # Function to determine de number of variables in the economic 
            # dispatch
            self.dnvariablesED()
            # Function to determine de number of constraints in the economic 
            # dispatch
            self.dnconstraintsED()
            # Number of variables in the Energy and Network models
            self.number_variablesENM = self.number_variablesED + \
                self.number_variablesEM
            # Number of constraints in the Energy and Network models
            self.number_constraintsENM = self.number_constraintsED + \
                self.number_constraintsEM
            # Creation of variables for the energy model in 
            # glpk (matrix A)
            self.solver.add_cols('EMcols', self.number_variablesEM)
            # Creation of variables for the economic dispatch in 
            # glpk (matrix A)
            self.variablesED()

            # define matrix of coeficients (matrix A)
            self.Bounds_variablesEM()


        def coeffmatrixEEDM(self):
            """ This class method contains the functions that allow building 
            the coefficient matrix (matrix A) for the simplex method """
            # The coefficient matrix is stored in CSR format (sparse matrix) 
            # to be later added to glpk
            self.ia = np.empty(math.ceil(self.number_constraintsENM * \
                self.number_variablesENM / 3), dtype=int) # Position in rows
            self.ja = np.empty(math.ceil(self.number_constraintsENM * \
                self.number_variablesENM / 3), dtype=int) # Position in columns
            self.ar = np.empty(math.ceil(self.number_constraintsED * \
                self.number_variablesENM / 3), dtype=float) # Value
            self.ne = 0 # Number of non-zero coefficients in matrix A

            self.Energybalance()
            self.Aggregation()
            if self.LL['NosUnc'] != 0:
                self.AggregationStochastic()

            self.constraintsED()
            self.activepowerbalancesystem()
            self.piecewiselinearisationcost()
            self.generationrampsconstraints()



            self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)

    def posconstraintsEED(self):
            """ This class method creates the vectors that store the positions
            of contraints that links the energy and ED problems """
            # Creating the matrices to store the position of constraints in
            # matrix A
            self.connectionEDandEnergy = np.empty(\
                self.size['Vectors'],\
                len(self.NM.connections['set']), dtype=[('napos', 'U20'),\
                    ('nupos', 'i4')]) # Start position 
                        # of energy and economic dispatch constraints (rows)             
    
    def constraintsEED(self):
        """ This class method reserves the space in glpk for the constraints of
        that links the energy and economic dispatch problems """

        self.posconstraintsEED()

        for i in range(self.size['Vectors']):
            for j in self.NM.connections['set']:
                self.connectionEDandEnergy[i, j] = ('CEED'+str(i)+str(j),\
                    self.solver.add_rows('CEED'+str(i)+str(j), 1))  # Number of 
                        # columns (constraints) in matrix A for the 
                        # constraints that links the energy and economic 
                        # dispatch model


    def EnergyandNetworkRelation(self):
        """ This class method writes the constraint that links the energy
        model with the network model in glpk.
    
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """
        # Generating the matrix A for the active power balance constraints
        for i in self.NM.connections['set']:
            for j in range(self.NM.settings['NoTime']):
            # Storing the thermal generation variables
                if len(self.NM.Gen.Conv) > 0:
                    for k in range(len(self.NM.Gen.Conv)):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = \
                            self.thermalgenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1