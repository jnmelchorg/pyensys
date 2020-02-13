"""
Pyene Models provides a glpk implementation of different methods
for: 
1) Balancing multiple vectors at different time aggregation levels.
2) Optimal dispatch of diverse generation technologies without considering
the transmission system (Economic Dispatch)
3) Optimal dispatch of diverse generation technologies considering
the transmission system (Optimal Power Flow)

@author: Dr Jose Nicolas Melchor Gutierrez
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena

@contributor:
"""

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
        # Storing data input - Parameters
        assert obj!=None   # If the object is empty then raise an error
        self.TreeNodes = obj.LL['NosBal'] + 1  # Number of nodes in the 
                                                # temporal tree
        self.NumberTrees = obj.size['Vectors'] # Number of temporal trees
                                                # to be solved
        self.LLEB = obj.p['LLTS1'] # Link list for the energy balance
        self.LLEA = obj.p['LLTS2'] # Link List for the energy aggregation
        self.IntakeTree = obj.Weight['In'] # Inputs at each node of the
                                            # temporal tree
        self.OutputTree = obj.Weight['Out']    # Outputs at each node of the
                                                # temporal tree
        self.WeightNodes = obj.p['WghtFull']   # Weight of node (number of
                                                # days, weeks, etc.)


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

        for i in range(self.NumberTrees):
            print("vector %d:" %(i))
            for j in range(self.TreeNodes):
                 print("%f %f" %(self.solver.get_col_prim(str(\
                     self.Partialstorage[i][0]), j), \
                        self.solver.get_col_prim(str(self.Totalstorage[i][0]),\
                        j)))



    def modeldefinitionEM(self):
        """ This class method build and solve the optimisation problem,
         to be expanded with a general optimisation problem """
        # TODO: create functions such as posvariables and variables in a 
        # similar way than the network model
        self.dnvariablesEM()    # Function to determine de number 
                                # of variables
        self.dnconstraintsEM()  # Function to determine de number 
                                # of constraints

        # define matrix of coeficients (matrix A)
        self.variablesEM()
        self.coeffmatrixEM()
        self.Objective_functionEM()

    def dnvariablesEM(self):
        """ This class method determines the number of variables """
        self.number_variablesEM += (self.TreeNodes) * 4 \
            * self.NumberTrees

    def dnconstraintsEM(self):
        """ This class method determines the number of constraints """
        # Number of constrains in the energy balance
        self.number_constraintsEM += (self.TreeNodes - 1) * self.NumberTrees
        self.number_constraintsEM += (self.TreeNodes - 1) * self.NumberTrees
        # TODO: create a case for uncertainty
        # if self.LL['NosUnc'] != 0:
        #     self.number_constraintsEM += (self.LL['NosUnc']+1) \
        #         * self.NumberTrees

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

        self.constraintsEM()
        self.Energybalance()
        self.Aggregation()
        # if self.LL['NosUnc'] != 0:
        #     self.AggregationStochastic()
        self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)

    # Variables EM

    def PosvariablesEM(self):
        """ This class method creates the vector that stores the positions of 
        variables for the energy problem """

        self.Partialstorage = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
            # of variables in matrix A (rows) for partial storage of 
            # energy or water in the tree for each vector

        self.Totalstorage = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
            # of variables in matrix A (rows) for total storage of 
            # energy or water in the tree for each vector
        
        self.InputsTree = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
            # of variables in matrix A (rows) for inputs of 
            # energy or water in the tree for each vector

        self.OutputsTree = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
            # of variables in matrix A (rows) for inputs of 
            # energy or water in the tree for each vector

    def variablesEM(self):
        """ This class method defines the variables and their limits for the
        economic dispatch problem """
        self.PosvariablesEM()
        # Reserving space in glpk for energy model variables
        for i in range(self.NumberTrees):
            # Variables for storage of energy or water
            self.Partialstorage[i] = ('PartialStorage'+str(i),\
                self.solver.add_cols('PartialStorage'+str(i),\
                (self.TreeNodes)))
        for i in range(self.NumberTrees):
            # Variables for storage of energy or water
            self.Totalstorage[i] = ('TotalStorage'+str(i),\
                self.solver.add_cols('TotalStorage'+str(i),\
                (self.TreeNodes)))
        for i in range(self.NumberTrees):
            # Variables for storage of energy or water
            self.InputsTree[i] = ('InputsTree'+str(i),\
                self.solver.add_cols('InputsTree'+str(i),\
                (self.TreeNodes)))
        for i in range(self.NumberTrees):
            # Variables for storage of energy or water
            self.OutputsTree[i] = ('OutputsTree'+str(i),\
                self.solver.add_cols('OutputsTree'+str(i),\
                (self.TreeNodes)))
        

        # Defining the limits of the variables
        for i in range(self.NumberTrees):
            # Limits for initial nodes of the tree for storage of 
            # energy or water
            self.solver.set_col_bnds(\
                str(self.Partialstorage[i][0]), 0, 'fixed', 0.0, 0.0)
            self.solver.set_col_bnds(\
                str(self.Totalstorage[i][0]), 0, 'fixed', 0.0, 0.0)
            for j in range(self.TreeNodes):
                self.solver.set_col_bnds(\
                    str(self.InputsTree[i][0]), j, 'fixed', \
                        self.IntakeTree[j, i], self.IntakeTree[j, i])
                self.solver.set_col_bnds(\
                    str(self.OutputsTree[i][0]), j, 'fixed', \
                        self.OutputTree[j, i], self.OutputTree[j, i])

    # Constraints EM
    
    def posconstraintsEM(self):
        """ This class method creates the vectors that store the positions of 
        contraints for the energy problem """
        # Creating the matrices to store the position of constraints in
        # matrix A
        self.treebalance = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
                    # of the tree balance constraints (rows) 
                    # for vector
        self.treeaggregation = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
                    # of the tree aggregation constraints (rows) 
                    # for vector
        
    def constraintsEM(self):
        """ This class method reserves the space in glpk for the constraints of
        the energy problem """

        self.posconstraintsEM()

        for i in range(self.NumberTrees):
            self.treebalance[i] = ('TB'+str(i), \
                self.solver.add_rows('TB'+str(i), \
                    (self.TreeNodes - 1)))  # Number of 
                    # rows (constraints) in matrix A for the three balance
                    # for each vector
        for i in range(self.NumberTrees):
            self.treeaggregation[i] = ('TA'+str(i), \
                self.solver.add_rows('TA'+str(i), \
                    (self.TreeNodes - 1)))  # Number of 
                    # rows (constraints) in matrix A for the three aggregation
                    # for each vector
                
    def Energybalance(self):
        """ This class method writes the energy balance in glpk
        
        First, it is reserved space in memory to store the energy balance 
        constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """
        # Generating the matrix A for the energy contraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                # Storing the Vin variables
                self.ia[self.ne] = self.treebalance[vectors][1] + nodes - 1
                self.ja[self.ne] = self.Partialstorage[vectors][1] + nodes
                self.ar[self.ne] = 1
                # Storing the Vout variables
                self.ne += 1
                self.ia[self.ne] = self.treebalance[vectors][1] + nodes - 1
                self.ar[self.ne] = -1
                if(self.LLEB[nodes, 1] == 0):
                    self.ja[self.ne] = self.Partialstorage[vectors][1] + \
                            self.LLEB[nodes, 0]
                elif(self.LLEB[nodes, 1] == 1):
                    self.ja[self.ne] = self.Totalstorage[vectors][1] + \
                            self.LLEB[nodes, 0]
                # Storing the Inputs            
                self.ne += 1
                self.ia[self.ne] = self.treebalance[vectors][1] + nodes - 1
                self.ja[self.ne] = self.InputsTree[vectors][1] + nodes
                self.ar[self.ne] = -1
                # Storing the Outputs            
                self.ne += 1
                self.ia[self.ne] = self.treebalance[vectors][1] + nodes - 1
                self.ja[self.ne] = self.OutputsTree[vectors][1] + nodes
                self.ar[self.ne] = 1
                self.ne += 1

        # Defining the limits for the energy constraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                self.solver.set_row_bnds(str(self.treebalance[vectors][0]), \
                    nodes - 1, 'fixed', 0, 0)

        # For verification
        # TODO: include it in pytest
        # for i in range(self.ne):
        #       print("%d %d %d" %(self.ia[i], self.ja[i], self.ar[i]))
        # for vectors in range(self.NumberTrees):
        #     for nodes in range(1, self.TreeNodes):
        #         print("%f" %(self.IntakeTree[nodes, vectors] - self.OutputTree[nodes, vectors]))            
        # import sys
        # sys.exit('hasta aqui')

    def Aggregation(self):
        """ This class method writes the aggregation constraints in glpk
        
        First, it is reserved space in memory to store the aggregation constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the aggregation contraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                # Storing the Vout variables
                self.ia[self.ne] = self.treeaggregation[vectors][1] + nodes - 1
                self.ja[self.ne] = self.Totalstorage[vectors][1] + nodes
                self.ar[self.ne] = 1
                # Storing Vin or Vout variables
                self.ne += 1
                self.ia[self.ne] = self.treeaggregation[vectors][1] + nodes - 1
                self.ar[self.ne] = -self.WeightNodes[self.LLEA\
                    [nodes, 0]]
                if(self.LLEA[nodes, 2] == 0):
                    self.ja[self.ne] = self.Partialstorage[vectors][1]\
                        + self.LLEA[nodes, 1]
                elif(self.LLEA[nodes, 2] == 1):
                    self.ja[self.ne] = self.Totalstorage[vectors][1]\
                        + self.LLEA[nodes, 1]
                # Storing Vin or Vout variables
                if(1 - self.WeightNodes[self.LLEA[nodes, 0]] != 0):
                    self.ne += 1
                    self.ia[self.ne] = self.treeaggregation[vectors][1]\
                        + nodes - 1
                    self.ar[self.ne] = -(1 - self.WeightNodes\
                        [self.LLEA[nodes, 0]])
                    if(self.LLEB[self.LLEA[nodes, 0], 1] == 0):
                        self.ja[self.ne] = self.Partialstorage[vectors][1] + \
                            self.LLEB[self.LLEA[nodes, 0], 0]
                    elif(self.LLEB[self.LLEA[nodes, 0], 1] == 1):
                        self.ja[self.ne] = self.Totalstorage[vectors][1] \
                            + self.LLEB[self.LLEA[nodes, 0], 0]
                self.ne += 1

        # Defining the limits for the aggregation constraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                self.solver.set_row_bnds(str(self.treeaggregation[vectors][0]), \
                    nodes - 1, 'fixed', 0.0, 0.0)

        # For verification
        # TODO: include it in pytest
        # for i in range(self.ne):
        #       print("%d %d %d" %(self.ia[i], self.ja[i], self.ar[i]))
        # for vectors in range(self.NumberTrees):
        #     for nodes in range(1, self.TreeNodes):
        #         print("%f" %(self.IntakeTree[nodes, vectors] - self.OutputTree[nodes, vectors]))            
        # import sys
        # sys.exit('hasta aqui')


    # TODO: Modify Stochastic Aggregation constraint with new positions of 
    # variables and constraints
    def AggregationStochastic(self):
        """ This class method writes the aggregation constraints for stochastic scenarios in glpk
        
        First, it is reserved space in memory to store the aggregation constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Reserving space in glpk for aggregation constraints
        self.Agg_Sto_row_number = self.solver.add_rows('AggStoch', (self.NumberTrees - 1) * \
            self.TreeNodes)   # Number of columns (constraints) in matrix A
                                    # for aggregation        
        nep = self.ne
        # Generating the matrix A for the aggregation contraints
        # TODO review this constraint
        for vectors in range(self.NumberTrees):
            for nodes in range(2, self.LL['NosUnc']+2): # TODO, does it start from position 2??
                # Storing the first variable of each constraint
                self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                    (self.TreeNodes - 1)) + nodes - 2
                self.ja[self.ne] = (vectors * \
                    (self.TreeNodes)) + \
                    (self.NumberTrees * (self.TreeNodes)) + \
                    self.p['LLTS3'][nodes - 1, 0] + 1
                self.ar[self.ne] = 1
                # Storing the second variable of each constraint
                if(1-self.WeightNodes[self.p['LLTS3'][nodes - 1, 0]] != 0):
                    self.ne += 1
                    self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                        (self.TreeNodes - 1)) + nodes - 2
                    self.ar[self.ne] = -(1-self.WeightNodes[self.p['LLTS3'][nodes - 1, 0]])
                    if(self.LLEB[self.p['LLTS3'][nodes - 1, 0], 1] == 0):
                        self.ja[self.ne] = (vectors * \
                            (self.TreeNodes)) + self.LLEB[self.p['LLTS3'][nodes - 1, 0], 0] + 1
                    elif(self.LLEB[self.p['LLTS3'][nodes - 1, 0], 1] == 1):
                        self.ja[self.ne] = (vectors * \
                            (self.TreeNodes)) + (self.NumberTrees * \
                            (self.TreeNodes)) + self.LLEB[self.p['LLTS3'][nodes - 1, 0], 0] + 1
                # Storing the third variable
                self.ne += 1
                self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                    (self.TreeNodes - 1)) + nodes - 2
                self.ar[self.ne] = -(self.WeightNodes[self.p['LLTS3'][nodes - 1, 0]] * \
                    -self.p['LLTS3'][nodes - 1, 2])
                self.ja[self.ne] = (vectors * \
                    (self.TreeNodes)) + self.p['LLTS3'][nodes - 1, 0] + 1
                # Storing variables in the summation
                for aux1 in range(self.p['LLTS3'][nodes - 1, 2] + 1):
                    self.ne += 1
                    self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                        (self.TreeNodes - 1)) + nodes - 2
                    self.ar[self.ne] = -(self.WeightNodes[self.p['LLTS3'][nodes - 1, 0]] * \
                        -self.p['LLTS3'][nodes - 1, 2])
                    self.ja[self.ne] = (vectors * \
                            (self.TreeNodes)) + (self.NumberTrees * \
                            (self.TreeNodes)) + self.p['LLTS3'][nodes, 1] + aux1 + 1
                self.ne += 1

                    

        # Defining the limits for the aggregation constraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.LL['NosUnc']+1):
                self.solver.set_row_bnds('AggStoch', (vectors *  (self.TreeNodes - 1)) + nodes - 1, 'fixed', \
                    0.0, 0.0)

    # Objective function EM

    def Objective_functionEM(self):
        """ This class method defines the cost coefficients for the
         objective function in glpk
         
        A dummy objective function is created for the problem """

        self.solver.set_obj_coef(str(self.Partialstorage[0][0]), 1, 2)
        self.solver.set_obj_coef(str(self.Partialstorage[0][0]), 2, -1)
        self.solver.set_obj_coef(str(self.Totalstorage[1][0]), 2, -1)


class Networkmodel():
    """ This class builds and solve the network model(NM) using the gplk wrapper.

    The information of the pyeneClass is passed to this class,
    which provides the parameters for the model. Furthermore,
    the GLPKSolver class that contains the GLPK wrapper is imported """

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
        print(dir(self))
        # Definition of the mathematical formulation
        self.EconomicDispatchModel()
        ret = self.solver.simplex()
        assert ret == 0


        for i in self.connections['set']:
            print('Case %d :' %(i))
            print('')
            print('Generation:')
            for k in range(len(self.Gen.Conv)):
                for j in range(self.settings['NoTime']):
                    print("%f" %(self.solver.get_col_prim(\
                        str(self.thermalgenerators[i, j][0]), k) * \
                            self.ENetwork.get_Base()), end = ' ')
                print('')
            for k in range(len(self.Gen.RES)):
                for j in range(self.settings['NoTime']):                
                    print("%f" %(self.solver.get_col_prim(\
                        str(self.RESgenerators[i, j][0]), k) * \
                            self.ENetwork.get_Base()), end = ' ')
                print('')
            for k in range(len(self.Gen.Hydro)):
                for j in range(self.settings['NoTime']):
                    print("%f" %(self.solver.get_col_prim(\
                        str(self.Hydrogenerators[i, j][0]), k) * \
                            self.ENetwork.get_Base()), end = ' ')
                print('')
            print('')
            if self.pumps['Number'] > 0:
                print('Pumps:')
                for k in range(self.pumps['Number']):
                    for j in range(self.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.pumpsvar[i, j][0]), k) * \
                                self.ENetwork.get_Base()), end = ' ')
                    print('')
                print('')
            print('LC:')
            for j in range(self.settings['NoTime']):
                print("%f" %(self.solver.get_col_prim(\
                            str(self.loadcurtailmentsystem[i, j][0]), 0) * \
                                self.ENetwork.get_Base()), end = ' ')
            print('\n\n')
            if len(self.Gen.Conv) > 0:
                print('Thermal Generation cost:')
                for k in range(len(self.Gen.Conv)):
                    for j in range(self.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.thermalCG[i, j][0]), k)), end = ' ')
                    print('')
                print('')
            if len(self.Gen.RES) > 0:
                print('RES Generation cost:')
                for k in range(len(self.Gen.RES)):
                    for j in range(self.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.RESCG[i, j][0]), k)), end = ' ')
                    print('')
                print('')
            if len(self.Gen.Hydro) > 0:
                print('Hydro Generation cost:')
                for k in range(len(self.Gen.Hydro)):
                    for j in range(self.settings['NoTime']):
                        print("%f" %(self.solver.get_col_prim(\
                            str(self.HydroCG[i, j][0]), k)), end = ' ')
                    print('')
                print('')
        print('')


    ############################################
    ###   COMMON VARIABLES AND CONSTRAINTS   ###
    ###   FOR DIFFERENT MODELS               ###
    ############################################

    # Number of variables and constraints
    def dnvariablesCommon(self):
        """ This class method determines the number of variables that 
        are common for various problems """
        # TODO: Create a variable for size last tree nodes
        # len(self.connections['set'])
        # TODO: Further analysis of energy storage variables and constraints
        # Active power generation variables
        self.number_variablesCommon = 0
        self.number_variablesCommon += len(self.connections['set']) \
            * self.Gen.get_NoGen() * self.settings['NoTime']
        # Generation cost variables
        self.number_variablesCommon += len(self.connections['set']) \
            * self.Gen.get_NoGen() * self.settings['NoTime']
        # Active power storage variables
        self.number_variablesCommon += len(self.connections['set']) \
            * self.Storage['Number'] * self.settings['NoTime']
        # Pumps variables
        self.number_variablesCommon += len(self.connections['set']) \
            * self.pumps['Number'] * self.settings['NoTime']

    def dnconstraintsCommon(self):
        """ This class method determines the number of constraints that 
        are common for various problems """

        self.number_constraintsCommon = 0
        self.number_constraintsCommon += len(self.connections['set']) \
            * self.Gen.get_NoGen() * self.settings['NoTime'] * \
                self.Gen.get_NoPieces() # Constraints 
                                        # for the piecewise linearisation
                                        # of the quadratic generation cost
        self.number_constraintsCommon += len(self.connections['set']) \
            * self.Gen.get_NoGen() * self.settings['NoTime'] # Constraints 
                                    # for the generation ramps

    # Variables
    def PosvariablesCommon(self):
        """ This class method creates the vector that stores the positions of 
        variables that are common for various problems """

        if len(self.Gen.Conv) > 0:
            self.thermalgenerators = np.empty(\
                (len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of thermal generators' variables in matrix A (rows)
                # for each period and each tree node
        if len(self.Gen.RES) > 0:
            self.RESgenerators = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of RES generators' variables in matrix A (rows)
                # for each period and each tree node
        if len(self.Gen.Hydro) > 0:        
            self.Hydrogenerators = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of Hydroelectric generators' variables in matrix A (rows)
                # for each period and each tree node
        # TODO: Change this with a flag for batteries
        if self.Storage['Number'] > 0:
            self.ESS = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of Energy Storage Systems' variables in matrix A (rows)
                # for each period and each tree node
        # TODO: Change this with a flag for pumps
        if self.pumps['Number'] > 0:
            self.pumpsvar = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of pumps' variables in matrix A (rows)
                # for each period and each tree node
        if len(self.Gen.Conv) > 0:
            self.thermalCG = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of thermal generation cost variables in matrix A (rows)
                # for each period and each tree node
        if len(self.Gen.RES) > 0:
            self.RESCG = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of RES generation cost variables in matrix A (rows)
                # for each period and each tree node
        if len(self.Gen.Hydro) > 0:
            self.HydroCG = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of Hydroelectric generation cost variables in matrix A (rows)
                # for each period and each tree node

    def variablesCommon(self):
        """ This class method defines the variables and their limits that 
        are common for various problems """
        self.PosvariablesED()
        
        # Reserving space in glpk for ED variables
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                # Generation variables
                if len(self.Gen.Conv) > 0:
                    self.thermalgenerators[i, j] = (\
                        'ThermalGen'+str(i)+str(j),\
                        self.solver.add_cols('ThermalGen'+str(i)+str(j),\
                        len(self.Gen.Conv)))
                if len(self.Gen.RES) > 0:
                    self.RESgenerators[i, j] = (\
                        'RESGen'+str(i)+str(j),\
                        self.solver.add_cols('RESGen'+str(i)+str(j),\
                        len(self.Gen.RES)))
                if len(self.Gen.Hydro) > 0:
                    self.Hydrogenerators[i, j] = (\
                        'HydroGen'+str(i)+str(j),\
                        self.solver.add_cols('HydroGen'+str(i)+str(j),\
                        len(self.Gen.Hydro)))
                # Generation cost variables
                if len(self.Gen.Conv) > 0:
                    self.thermalCG[i, j] = ('ThermalCG'+str(i)+str(j),\
                        self.solver.add_cols('ThermalCG'+str(i)+str(j),\
                        len(self.Gen.Conv)))
                if len(self.Gen.RES) > 0:
                    self.RESCG[i, j] = ('RESCG'+str(i)+str(j),\
                        self.solver.add_cols('RESCG'+str(i)+str(j),\
                        len(self.Gen.RES)))
                if len(self.Gen.Hydro) > 0:
                    self.HydroCG[i, j] = ('HydroCG'+str(i)+str(j),\
                        self.solver.add_cols('HydroCG'+str(i)+str(j),\
                        len(self.Gen.Hydro)))
                # TODO: Change this with a flag for batteries
                if self.Storage['Number'] > 0:
                    self.ESS[i, j] = ('ESS'+str(i)+str(j),\
                        self.solver.add_cols('ESS'+str(i)+str(j),\
                        self.Storage['Number']))
                # TODO: Change this with a flag for pumps
                if self.pumps['Number'] > 0:
                    self.pumpsvar[i, j] = ('Pumps'+str(i)+str(j),\
                        self.solver.add_cols('Pumps'+str(i)+str(j),\
                        self.pumps['Number']))


        # Defining the limits of the variables
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                # Limits for the thermal generators
                if len(self.Gen.Conv) > 0:
                    for k in range(len(self.Gen.Conv)):
                        self.solver.set_col_bnds(\
                            str(self.thermalgenerators[i, j][0]), k,\
                            'bounded', self.Gen.Conv[k].get_Min(),\
                            self.Gen.Conv[k].get_Max())
                # Limits for the RES generators
                if len(self.Gen.RES) > 0:
                    for k in range(len(self.Gen.RES)):
                        self.solver.set_col_bnds(\
                            str(self.RESgenerators[i, j][0]), k,\
                            'bounded', self.Gen.RES[k].get_Min(),\
                            self.scenarios['RES']\
                                [self.resScenario[k][i]+j] * \
                                self.RES['Max'][k])

                # Limits for the Hydroelectric generators
                if len(self.Gen.Hydro) > 0:
                    for k in range(len(self.Gen.Hydro)):
                        self.solver.set_col_bnds(\
                            str(self.Hydrogenerators[i, j][0]), k,\
                            'bounded', self.Gen.Hydro[k].get_Min(),\
                            self.Gen.Hydro[k].get_Max())
                # TODO: Modify information of storage, e.g. m.sNSto
                # if self.Storage['Number'] > 0:
                if self.pumps['Number'] > 0:
                    for k in range(self.pumps['Number']):
                        self.solver.set_col_bnds(str(self.pumpsvar[i, j][0]), k,\
                            'bounded', 0,\
                            self.pumps['Max'][k]/self.ENetwork.get_Base())

    # Constraints
    def posconstraintsCommon(self):
            """ This class method creates the vectors that store the positions of 
            contraints that are common for various problems """
            # Creating the matrices to store the position of constraints in
            # matrix A
            if len(self.Gen.Conv) > 0:
                self.thermalpiecewisecost = \
                    np.empty((len(self.connections['set']),\
                    self.settings['NoTime'], len(self.Gen.Conv)),\
                        dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                        # position of piecewise linearisation constraints 
                        # (rows) for each tree node, for each period and 
                        # for each thermal generator
            if len(self.Gen.RES) > 0:
                self.RESpiecewisecost = \
                    np.empty((len(self.connections['set']),\
                    self.settings['NoTime'], len(self.Gen.RES)),\
                        dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                        # position of piecewise linearisation constraints 
                        # (rows) for each tree node, for each period and 
                        # for each RES generator
            if len(self.Gen.Hydro) > 0:
                self.Hydropiecewisecost = \
                    np.empty((len(self.connections['set']),\
                    self.settings['NoTime'], len(self.Gen.Hydro)),\
                        dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                        # position of piecewise linearisation constraints 
                        # (rows) for each tree node, for each period and 
                        # for each Hydro generator
            if len(self.Gen.Conv) > 0:
                self.thermalgenerationramps = \
                    np.empty((len(self.connections['set']),\
                    self.settings['NoTime'] - 1),\
                        dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                        # position of thermal generation ramps constraints 
                        # (rows) for each tree node, for each period and for 
                        # each thermal generator
            if len(self.Gen.Hydro) > 0:
                self.Hydrogenerationramps = \
                    np.empty((len(self.connections['set']),\
                    self.settings['NoTime'] - 1),\
                        dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                        # position of Hydroelectrical generation ramps constraints
                        # (rows) for each tree node, for each period and for 
                        # each hydroelectrical generator
            
    def constraintsCommon(self):
        """ This class method reserves the space in glpk for the constraints of
        the economic dispatch problem """

        self.posconstraintsED()

        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                if len(self.Gen.Conv) > 0:
                    for k in range(len(self.Gen.Conv)):
                        self.thermalpiecewisecost[i, j, k] =\
                            ('ThermalPWC'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                                'ThermalPWC'+str(i)+str(j)+str(k), \
                                self.Gen.Conv[k].get_NoPieces()))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each thermal generator
                if len(self.Gen.RES) > 0:
                    for k in range(len(self.Gen.RES)):
                        self.RESpiecewisecost[i, j, k] =\
                            ('RESPWC'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                                'RESPWC'+str(i)+str(j)+str(k), \
                                self.Gen.RES[k].get_NoPieces()))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each RES generator
                if len(self.Gen.Hydro) > 0:
                    for k in range(len(self.Gen.Hydro)):
                        self.Hydropiecewisecost[i, j, k] =\
                            ('HydroPWC'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                                'HydroPWC'+str(i)+str(j)+str(k), \
                                self.Gen.Hydro[k].get_NoPieces()))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each Hydro generator
                if j > 0:
                    if len(self.Gen.Conv) > 0:
                        self.thermalgenerationramps[i, j - 1] = \
                            ('ThermalGR'+str(i)+str(j),\
                            self.solver.add_rows('ThermalGR'+str(i)+str(j),\
                                len(self.Gen.Conv)))  # Number of 
                                # columns (constraints) in matrix A for the 
                                # generation ramps constraints for each 
                                # period, for each tree node and for each 
                                # thermal generator
                    if len(self.Gen.Hydro) > 0:
                        self.Hydrogenerationramps[i, j - 1] = \
                            ('HydroGR'+str(i)+str(j),\
                            self.solver.add_rows('HydroGR'+str(i)+str(j),\
                                len(self.Gen.Hydro)))  # Number of 
                                # columns (constraints) in matrix A for the 
                                # generation ramps constraints for each 
                                # period, for each tree node and for each 
                                # thermal generator

    def piecewiselinearisationcost(self):
        """ This class method writes the piecewise linearisarion of
        the generation cost in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the piecewise linearisation constraints of
        # the generation cost
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                if len(self.Gen.Conv) > 0:
                    for k in range(len(self.Gen.Conv)):
                        for l in range(self.Gen.Conv[k].get_NoPieces()):
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
                                -self.scenarios['Weights'][j] * \
                                self.Gen.Conv[k].cost['LCost'][l][0]
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.thermalpiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.scenarios['Weights'][j] * \
                                self.Gen.Conv[k].cost['LCost'][l][1], 0)

                if len(self.Gen.RES) > 0:
                    for k in range(len(self.Gen.RES)):
                        for l in range(self.Gen.RES[k].get_NoPieces()):
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
                                -self.scenarios['Weights'][j] * \
                                self.Gen.RES[k].cost['LCost'][l][0]
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.RESpiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.scenarios['Weights'][j] * \
                                self.Gen.RES[k].cost['LCost'][l][1], 0)

                if len(self.Gen.Hydro) > 0:
                    for k in range(len(self.Gen.Hydro)):
                        for l in range(self.Gen.Hydro[k].get_NoPieces()):
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
                                -self.scenarios['Weights'][j] * \
                                self.Gen.Hydro[k].cost['LCost'][l][0]
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.Hydropiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.scenarios['Weights'][j] * \
                                self.Gen.Hydro[k].cost['LCost'][l][1], 0)

    def generationrampsconstraints(self):
        """ This class method writes the constraints for the generation ramps
        in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the generation ramps constraints
        for i in self.connections['set']:
            for j in range(1, self.settings['NoTime']):
                if len(self.Gen.Conv) > 0:
                    for k in range(len(self.Gen.Conv)):
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
                            k, 'bounded', -self.Gen.Conv[k].data['Ramp'],\
                            self.Gen.Conv[k].data['Ramp'])
                if len(self.Gen.Hydro) > 0:
                    for k in range(len(self.Gen.Hydro)):
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
                            k, 'bounded', -self.Gen.Hydro[k].data['Ramp'],\
                            self.Gen.Hydro[k].data['Ramp'])

    # Objective function

    def Objective_functionCommon(self):
        """ This class method defines the objective function of the economic
        dispatch in glpk """

        # Calculating the aggregated weights for the last nodes in the tree
        # TODO: explain the aggregated weights better
        # WghtAgg = 0 + self.EM.p['WghtFull']
        OFaux = np.ones(len(self.connections['set']), dtype=float)
        # xp = 0
        # for xn in range(self.TreeNodes):
        #     aux = self.LLNodesAfter[xn][0]
        #     if aux != 0:
        #         for xb in range(aux, self.EM.self.LLNodesAfter[xn][1] + 1):
        #             WghtAgg[xb] *= WghtAgg[xn]
        #     else:
        #         OFaux[xp] = WghtAgg[xn]
        #         xp += 1

        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
            # Cost for conventional generation    
                if len(self.Gen.Conv) > 0: 
                    for k in range(len(self.Gen.Conv)):
                        self.solver.set_obj_coef(\
                            str(self.thermalCG[i, j][0]),\
                            k, OFaux[i] * self.scenarios['Weights'][j])
            # Cost for RES generation    
                if len(self.Gen.RES) > 0: 
                    for k in range(len(self.Gen.RES)):
                        self.solver.set_obj_coef(\
                            str(self.RESCG[i, j][0]),\
                            k, OFaux[i] * self.scenarios['Weights'][j])
            # Cost for Hydroelectric generation    
                if len(self.Gen.Hydro) > 0: 
                    for k in range(len(self.Gen.Hydro)):
                        self.solver.set_obj_coef(\
                            str(self.HydroCG[i, j][0]),\
                            k, OFaux[i] * self.scenarios['Weights'][j])
            # Operation cost of pumps
                if self.pumps['Number'] > 0:
                    for k in range(self.pumps['Number']):
                        self.solver.set_obj_coef(\
                            str(self.pumpsvar[i, j][0]),\
                            k, -OFaux[i] * self.scenarios['Weights'][j] \
                                * self.ENetwork.get_Base() \
                                    * self.pumps['Value'][k])
            # Punitive cost for load curtailment
                if self.settings['Flag']:
                # Optimal Power Flow
                    for k in range(len(self.settings['Security']) + 1):
                        for ii in range(self.ENetwork.get_NoBus()):
                            self.solver.set_obj_coef(\
                                str(self.LoadCurtailmentNode[i, j, k][0]),\
                                ii, OFaux[i] * self.scenarios['Weights'][j] \
                                    * 100000000)
                else:
                # Economic Dispatch
                # TODO: Set a parameter penalty in pyeneN
                    self.solver.set_obj_coef(\
                                str(self.loadcurtailmentsystem[i, j][0]),\
                                0, OFaux[i] * self.scenarios['Weights'][j] \
                                    * 100000000)

    #############################
    ###   ECONOMIC DISPATCH   ###
    #############################

    def EconomicDispatchModel(self):
        """ This class method builds the optimisation model
        for the economic dispatch problem """

        self.dnvariablesED()    # Function to determine de number of variables
        self.dnconstraintsED()  # Function to determine de number of constraints

        # define matrix of coeficients (matrix A)
        self.variablesED()
        self.coeffmatrixED()
        self.Objective_functionCommon()

    def dnvariablesED(self):
        """ This class method determines the number of variables for the 
        economic dispatch problem """

        self.number_variablesED = 0
        self.dnvariablesCommon()
        self.number_variablesED = self.number_variablesCommon
        # Load curtailment variables
        self.number_variablesED += len(self.connections['set']) \
            * self.settings['NoTime']

    def dnconstraintsED(self):
        """ This class method determines the number of constraints for the 
        economic dispatch problem """

        self.number_constraintsED = 0
        self.dnconstraintsCommon()
        self.number_constraintsED = self.number_constraintsCommon

        self.number_constraintsED += len(self.connections['set']) * \
            self.settings['NoTime']     # Constraints for power balance 
                                        # of whole power system

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

        self.PosvariablesCommon()

        self.loadcurtailmentsystem = np.empty((len(self.connections['set']),\
            self.settings['NoTime']),\
            dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
            # in matrix A (rows) of variables
            # for load curtailment in the system for each tree node

    def variablesED(self):
        """ This class method defines the variables and their limits for the
        economic dispatch problem """
        self.PosvariablesED()

        self.variablesCommon()
        
        # Reserving space in glpk for ED variables
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                self.loadcurtailmentsystem[i, j] = ('LCS'+str(i)+str(j),\
                    self.solver.add_cols('LCS'+str(i)+str(j), 1))

    # Constraints ED

    def posconstraintsED(self):
            """ This class method creates the vectors that store the positions of 
            contraints for the ED problem """
            # Creating the matrices to store the position of constraints in
            # matrix A

            self.posconstraintsCommon()

            self.powerbalance = np.empty((len(self.connections['set']),\
                self.settings['NoTime']), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                        # of active power balance constraints (rows) 
                        # for each tree node
            
    def constraintsED(self):
        """ This class method reserves the space in glpk for the constraints of
        the economic dispatch problem """

        self.posconstraintsED()

        self.constraintsCommon()

        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                self.powerbalance[i, j] = ('PB'+str(i)+str(j),\
                    self.solver.add_rows('PB'+str(i)+str(j), 1))  # Number of 
                        # columns (constraints) in matrix A for the active 
                        # power balance constraints fo each period and each 
                        # tree node

    def activepowerbalancesystem(self):
        """ This class method writes the power balance constraint in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the active power balance constraints
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
            # Storing the thermal generation variables
                if len(self.Gen.Conv) > 0:
                    for k in range(len(self.Gen.Conv)):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = \
                            self.thermalgenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
            # Storing the RES generation variables
                if len(self.Gen.RES) > 0:
                    for k in range(len(self.Gen.RES)):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = \
                            self.RESgenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
            # Storing the Hydroelectric generation variables
                if len(self.Gen.Hydro) > 0:
                    for k in range(len(self.Gen.Hydro)):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = \
                            self.Hydrogenerators[i, j][1] + k
                        self.ar[self.ne] = 1.0
                        self.ne += 1
            # Storing variables for ESS
            # TODO: Modify the constraint for the first period
                if self.Storage['Number'] > 0:
                    if j > 0: # Start only after the first period
                        for k in range(self.Storage['Number']):
                            self.ia[self.ne] = self.powerbalance[i, j][1]
                            self.ja[self.ne] = self.ESS[i, j][1] + k
                            self.ar[self.ne] = self.Storage['Efficiency'][k] \
                                / self.scenarios['Weights'][j - 1]
                            self.ne += 1
                        for k in range(self.Storage['Number']):
                            self.ia[self.ne] = self.powerbalance[i, j][1]
                            self.ja[self.ne] = self.ESS[i, j - 1][1] + k
                            self.ar[self.ne] = \
                                -self.Storage['Efficiency'][k] \
                                / self.scenarios['Weights'][j - 1]
                            self.ne += 1
            # Storing the variables for load curtailment
                self.ia[self.ne] = self.powerbalance[i, j][1]
                self.ja[self.ne] = self.loadcurtailmentsystem[i, j][1]
                self.ar[self.ne] = 1.0
                self.ne += 1
            # Storing the variables for pumps
                if self.pumps['Number'] > 0:
                    for k in range(self.pumps['Number']):
                        self.ia[self.ne] = self.powerbalance[i, j][1]
                        self.ja[self.ne] = self.pumpsvar[i, j][1] + k
                        self.ar[self.ne] = -1.0
                        self.ne += 1
            # Defining the resources (b) for the constraints
                totaldemand = 0                
                # TODO: Change the inputs of losses and demand scenarios
                # for parameters
                if self.scenarios['NoDem'] == 0:
                    if self.settings['Loss'] is None:
                        for k in range(self.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.busData[k] * \
                                self.scenarios['Demand']\
                                    [self.busScenario[k][i]]
                    else:
                        for k in range(self.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.busData[k] * \
                                self.scenarios['Demand']\
                                    [self.busScenario[k][i]] * \
                                (1 + self.settings['Loss'])
                else:
                    if self.settings['Loss'] is None:
                        for k in range(self.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.busData[k] * \
                                self.scenarios['Demand']\
                                    [j+self.busScenario[k][i]]
                    else:
                        for k in range(self.ENetwork.get_NoBus()):
                            totaldemand = totaldemand + self.busData[k] * \
                                self.scenarios['Demand'] \
                                    [j+self.busScenario[k][i]] * \
                                (1 + self.settings['Loss'])
                self.solver.set_row_bnds(str(self.powerbalance[i, j][0]), 0,\
                    'fixed', totaldemand, totaldemand)

    ##############################
    ###   OPTIMAL POWER FLOW   ###
    ##############################

    def OptimalPowerFlowModel(self):
        """ This class method builds the optimisation model
        for the economic dispatch problem """

        self.dnvariablesOPF()    # Function to determine de number of variables
        self.dnconstraintsOPF()  # Function to determine de number of constraints

        # define matrix of coeficients (matrix A)
        self.variablesOPF()
        self.coeffmatrixOPF()
        self.Objective_functionCommon()

    def dnvariablesOPF(self):
        """ This class method determines the number of variables for the 
        economic dispatch problem """

        self.number_variablesOPF = 0
        self.dnvariablesCommon()
        self.number_variablesOPF = self.number_variablesCommon
        # Active power flow variables
        self.number_variablesOPF += self.ENetwork.get_NoBra() * \
            len(self.connections['set']) * \
            (len(self.settings['Security']) + 1) \
            * self.settings['NoTime']
        if self.settings['Losses']:
            # Active power losses variables
            self.number_variablesOPF += self.ENetwork.get_NoBra() * \
                len(self.connections['set']) * \
                (len(self.settings['Security']) + 1) \
                * self.settings['NoTime']
        # load curtailment variables
        self.number_variablesOPF += self.ENetwork.get_NoBus() * \
            len(self.connections['set']) * \
            (len(self.settings['Security']) + 1) \
            * self.settings['NoTime']
        # Voltage angle variables
        self.number_variablesOPF += self.ENetwork.get_NoBus() * \
            len(self.connections['set']) * \
            (len(self.settings['Security']) + 1) \
            * self.settings['NoTime']

    def dnconstraintsOPF(self):
        """ This class method determines the number of constraints for the 
        economic dispatch problem """

        self.number_constraintsOPF = 0
        self.dnconstraintsCommon()
        self.number_constraintsOPF = self.number_constraintsCommon
        # Constraint that relates the active power flow and the voltage angle
        self.number_constraintsOPF += self.ENetwork.get_NoBra() * \
                len(self.connections['set']) * \
                (len(self.settings['Security']) + 1) \
                * self.settings['NoTime']
        if self.settings['Losses']:
            # Constraint for losses linearization
            self.number_constraintsOPF += self.ENetwork.get_NoBra() * \
                    len(self.connections['set']) * \
                    (len(self.settings['Security']) + 1) \
                    * self.settings['NoTime'] * \
                    len(self.connections['Loss_Param']) - 1
            # Constraint for losses linearization
            self.number_constraintsOPF += self.ENetwork.get_NoBra() * \
                    len(self.connections['set']) * \
                    (len(self.settings['Security']) + 1) \
                    * self.settings['NoTime'] * \
                    len(self.connections['Loss_Param']) - 1
        # Active power balance constraint
        self.number_constraintsOPF += self.ENetwork.get_NoBus() * \
                len(self.connections['set']) * \
                (len(self.settings['Security']) + 1) \
                * self.settings['NoTime']

    def coeffmatrixOPF(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) to be
        # later added to glpk
        self.ia = np.empty(math.ceil(self.number_constraintsOPF * \
            self.number_variablesOPF / 2), dtype=int) # Position in rows
        self.ja = np.empty(math.ceil(self.number_constraintsOPF * \
            self.number_variablesOPF / 2), dtype=int) # Position in columns
        self.ar = np.empty(math.ceil(self.number_constraintsOPF * \
            self.number_variablesOPF / 2), dtype=float) # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A
        
        self.constraintsOPF()

        self.piecewiselinearisationcost()
        self.generationrampsconstraints()

        self.activepowerbalancepernode()
        self.activepowerflowconstraints()
        if self.settings['Losses']:
            self.activepowerlosses1constraints()
            self.activepowerlosses2constraints()

        self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)

    # Variables ED

    def PosvariablesOPF(self):
        """ This class method creates the vector that stores the positions of 
        variables for the ED problem """

        self.PosvariablesCommon()

        self.ActivePowerFlow = np.empty((len(self.connections['set']),\
            self.settings['NoTime'], (len(self.settings['Security']) + 1)),\
            dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
            # in matrix A (rows) of variables for the active power flow
        if self.settings['Losses']:
            self.ActivePowerLosses = np.empty((len(self.connections['set']),\
                self.settings['NoTime'], (len(self.settings['Security']) + 1)),\
                dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
                # in matrix A (rows) of variables for the active power losses
        self.LoadCurtailmentNode = np.empty((len(self.connections['set']),\
            self.settings['NoTime'], (len(self.settings['Security']) + 1)),\
            dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
            # in matrix A (rows) of variables for the load curtailment per
            # node
        self.VoltageAngle = np.empty((len(self.connections['set']),\
            self.settings['NoTime'], (len(self.settings['Security']) + 1)),\
            dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
            # in matrix A (rows) of variables for the voltage angle per
            # node

    def variablesOPF(self):
        """ This class method defines the variables and their limits for the
        economic dispatch problem """
        self.PosvariablesOPF()

        self.variablesCommon()
        
        # Reserving space in glpk for OPF variables
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                for k in range(len(self.settings['Security']) + 1):
                    self.ActivePowerFlow[i, j, k] = \
                        ('ActivePowerFlow'+str(i)+str(j)+str(k),\
                        self.solver.add_cols(\
                        'ActivePowerFlow'+str(i)+str(j)+str(k),\
                        self.ENetwork.get_NoBra()))
                    if self.settings['Losses']:
                         self.ActivePowerLosses[i, j, k] = \
                            ('ActivePowerLosses'+str(i)+str(j)+str(k),\
                            self.solver.add_cols(\
                            'ActivePowerLosses'+str(i)+str(j)+str(k),\
                            self.ENetwork.get_NoBra()))
                    self.LoadCurtailmentNode[i, j, k] = \
                        ('LoadCurtailmentNode'+str(i)+str(j)+str(k),\
                        self.solver.add_cols(\
                        'LoadCurtailmentNode'+str(i)+str(j)+str(k),\
                        self.ENetwork.get_NoBus()))
                    self.VoltageAngle[i, j, k] = \
                        ('VoltageAngle'+str(i)+str(j)+str(k),\
                        self.solver.add_cols(\
                        'VoltageAngle'+str(i)+str(j)+str(k),\
                        self.ENetwork.get_NoBus()))
        
        # Defining the limits of the variables
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                for k in range(len(self.settings['Security']) + 1):
                    for ii in range(self.ENetwork.get_NoBra()):
                        # If the line is active in the current contingency then
                        # define the limits
                        if self.ENetwork.Branch[ii].is_active(k):
                            self.solver.set_col_bnds(\
                                str(self.ActivePowerFlow[i, j, k][0]), ii,\
                                'bounded', \
                                -self.ENetwork.Branch[ii].get_Rate(),\
                                self.ENetwork.Branch[ii].get_Rate())
                        # If the line is not active in the current contingency 
                        # then fix the active power flow to zero
                        else:
                            self.solver.set_col_bnds(\
                                str(self.ActivePowerFlow[i, j, k][0]), ii,\
                                'fixed', 0, 0)
                    if self.settings['Losses']:
                        for ii in range(self.ENetwork.get_NoBra()):
                        # If the line is not active in the current contingency 
                        # then fix the active power losses to zero
                            if not self.ENetwork.Branch[ii].is_active(k):
                                self.solver.set_col_bnds(\
                                    str(self.ActivePowerLosses[i, j, k][0]),\
                                        ii, 'fixed', 0, 0)
                    for ii in range(self.ENetwork.get_NoBus()):
                        # If the demand in the node is greater than zero then
                        # define the limits
                        if self.busData[ii] > 0:
                            if self.scenarios['NoDem'] == 0:
                                self.solver.set_col_bnds(\
                                    str(self.LoadCurtailmentNode[i, j, k][0]),\
                                    ii,'bounded', 0, self.busData[ii] * \
                                    self.scenarios['Demand'][0])
                            else:
                                self.solver.set_col_bnds(\
                                    str(self.LoadCurtailmentNode[i, j, k][0]),\
                                    ii,'bounded', 0, self.busData[ii] * \
                                    self.scenarios['Demand'][j])
                        # If the demand in the node is zero then
                        # fix the load curtailment to zero
                        else:
                            self.solver.set_col_bnds(\
                                str(self.LoadCurtailmentNode[i, j, k][0]), ii,\
                                'fixed', 0, 0)
                    for ii in range(self.ENetwork.get_NoBus()):
                        if self.ENetwork.Bus[ii].get_Type() is not 3:
                            self.solver.set_col_bnds(\
                                str(self.VoltageAngle[i, j, k][0]),\
                                    ii,'free', 0, 0)
                        else:
                            self.solver.set_col_bnds(\
                                str(self.VoltageAngle[i, j, k][0]),\
                                    ii,'fixed', 0, 0)
                    
    # Constraints OPF

    def posconstraintsOPF(self):
            """ This class method creates the vectors that store the positions of 
            contraints for the ED problem """
            # Creating the matrices to store the position of constraints in
            # matrix A

            self.posconstraintsCommon()

            self.activepowerbalancenode = np.empty(\
                (len(self.connections['set']),\
                self.settings['NoTime'], \
                (len(self.settings['Security']) + 1)), dtype=[('napos', 'U80'),\
                ('nupos', 'i4')]) # Start position of active power balance 
                                  # constraints (rows) per node
            self.activepowerflowconstraint = np.empty(\
                (len(self.connections['set']),\
                self.settings['NoTime'], \
                (len(self.settings['Security']) + 1)), dtype=[('napos', 'U80'),\
                ('nupos', 'i4')]) # Start position of active power flow 
                                  # constraints (rows) per line
            if self.settings['Losses']:
                self.activepowerlosses1 = np.empty(\
                    (len(self.connections['set']),\
                    self.settings['NoTime'], \
                    (len(self.settings['Security']) + 1), \
                    self.ENetwork.get_NoBra()), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position of active power losses 
                                      # constraints (rows) per line and per piece
                self.activepowerlosses2 = np.empty(\
                    (len(self.connections['set']),\
                    self.settings['NoTime'], \
                    (len(self.settings['Security']) + 1), \
                    self.ENetwork.get_NoBra()), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position of active power losses 
                                      # constraints (rows) per line and per piece
            
    def constraintsOPF(self):
        """ This class method reserves the space in glpk for the constraints of
        the economic dispatch problem """

        self.posconstraintsOPF()

        self.constraintsCommon()

        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                for k in range(len(self.settings['Security']) + 1):
                    self.activepowerbalancenode[i, j, k] = \
                        ('activepowerbalancenode'+str(i)+str(j)+str(k),\
                        self.solver.add_rows(\
                        'activepowerbalancenode'+str(i)+str(j)+str(k),\
                        self.ENetwork.get_NoBus()))  # Number of 
                            # rows (constraints) in matrix A for the active 
                            # power balance constraints per node
                    # Pre-contingency
                    if k == 0:
                        self.activepowerflowconstraint[i, j, k] = \
                            ('activepowerflowconstraint'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                            'activepowerflowconstraint'+str(i)+str(j)+str(k),\
                            self.ENetwork.get_NoBra()))  # Number of 
                                # rows (constraints) in matrix A for the active 
                                # power flow constraints per line
                    # Post-contingency
                    else:
                        self.activepowerflowconstraint[i, j, k] = \
                            ('activepowerflowconstraint'+str(i)+str(j)+str(k),\
                            self.solver.add_rows(\
                            'activepowerflowconstraint'+str(i)+str(j)+str(k),\
                            self.ENetwork.get_NoBra() - 1))  # Number of 
                                # rows (constraints) in matrix A for the active 
                                # power flow constraints per line
                    if self.settings['Losses']:
                        # Pre-contingency
                        if k == 0:
                            for ii in range(self.ENetwork.get_NoBra()):
                                self.activepowerlosses1[i, j, k, ii] = \
                                    ('activepowerlosses1'+str(i)+str(j)+str(k) \
                                    +str(ii), self.solver.add_rows(\
                                    'activepowerlosses1'+str(i)+str(j)+str(k) \
                                    +str(ii), self.Number_LossCon))
                                    # Number of rows (constraints) in matrix A 
                                    # for the active power losses constraints 
                                    # per line and per piece
                        # Post-contingency
                        else:
                            for ii in range(self.ENetwork.get_NoBra()):
                                # If the line is active in the current contingency
                                # then reserve the space
                                if self.ENetwork.Branch[ii].is_active(k):
                                    self.activepowerlosses1[i, j, k, ii] = \
                                        ('activepowerlosses1'+str(i)+str(j)+ \
                                        str(k) +str(ii), self.solver.add_rows(\
                                        'activepowerlosses1'+str(i)+str(j) \
                                        +str(k)+str(ii), \
                                        self.Number_LossCon)) # Number
                                            # of rows (constraints) in matrix A 
                                            # for the active power losses 
                                            # constraints per line 
                                            # and per piece
                                # If the line is not active in the current 
                                # contingency then do not reserve space
                                else:
                                     self.activepowerlosses1[i, j, k, ii] = \
                                        ('activepowerlosses1'+str(i)+str(j)+ \
                                        str(k) +str(ii), 0) # Number
                                            # of rows (constraints) in matrix A 
                                            # for the active power losses 
                                            # constraints per line 
                                            # and per piece
                        # Pre-contingency
                        if k == 0:
                            for ii in range(self.ENetwork.get_NoBra()):
                                self.activepowerlosses2[i, j, k, ii] = \
                                    ('activepowerlosses2'+str(i)+str(j)+str(k) \
                                    +str(ii), self.solver.add_rows(\
                                    'activepowerlosses2'+str(i)+str(j)+str(k) \
                                    +str(ii), self.Number_LossCon))
                            #         # Number of rows (constraints) in matrix A 
                            #         # for the active power losses constraints 
                            #         # per line and per piece
                        # Post-contingency
                        else:
                            for ii in range(self.ENetwork.get_NoBra()):
                                # If the line is active in the current contingency
                                # then reserve the space
                                if self.ENetwork.Branch[ii].is_active(k):
                                    self.activepowerlosses2[i, j, k, ii] = \
                                        ('activepowerlosses2'+str(i)+str(j)+ \
                                        str(k) +str(ii), self.solver.add_rows(\
                                        'activepowerlosses2'+str(i)+str(j) \
                                        +str(k)+str(ii), \
                                        self.Number_LossCon)) # Number
                                            # of rows (constraints) in matrix A 
                                            # for the active power losses 
                                            # constraints per line 
                                            # and per piece
                                # If the line is not active in the current 
                                # contingency then do not reserve space
                                else:
                                     self.activepowerlosses2[i, j, k, ii] = \
                                        ('activepowerlosses2'+str(i)+str(j)+ \
                                        str(k) +str(ii), 0) # Number
                                            # of rows (constraints) in matrix A 
                                            # for the active power losses 
                                            # constraints per line 
                                            # and per piece

    def activepowerbalancepernode(self):
        """ This class method writes the power balance constraint in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Generating the matrix A for the active power balance constraints
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                for k in range(len(self.settings['Security']) + 1):
                    for ii in range(self.ENetwork.get_NoBus()):
                    # Storing the thermal generation variables
                        if len(self.Gen.Conv) > 0:
                            for jj in range(len(self.Gen.Conv)):
                                if self.Gen.Conv[jj].get_Bus() == \
                                    self.ENetwork.Bus[ii].get_Number():
                                    self.ia[self.ne] = \
                                        self.activepowerbalancenode\
                                        [i, j, k][1] + ii
                                    self.ja[self.ne] = \
                                        self.thermalgenerators[i, j][1] + jj
                                    self.ar[self.ne] = 1.0
                                    self.ne += 1
                    # Storing the RES generation variables
                        if len(self.Gen.RES) > 0:
                            for jj in range(len(self.Gen.RES)):
                                if self.Gen.RES[jj].get_Bus() == \
                                    self.ENetwork.Bus[ii].get_Number():
                                    self.ia[self.ne] = \
                                        self.activepowerbalancenode\
                                        [i, j, k][1] + ii
                                    self.ja[self.ne] = \
                                        self.RESgenerators[i, j][1] + jj
                                    self.ar[self.ne] = 1.0
                                    self.ne += 1
                    # Storing the Hydroelectric generation variables
                        if len(self.Gen.Hydro) > 0:
                            for jj in range(len(self.Gen.Hydro)):
                                if self.Gen.Hydro[jj].get_Bus() == \
                                    self.ENetwork.Bus[ii].get_Number():
                                    self.ia[self.ne] = \
                                        self.activepowerbalancenode\
                                        [i, j, k][1] + ii
                                    self.ja[self.ne] = \
                                        self.Hydrogenerators[i, j][1] + jj
                                    self.ar[self.ne] = 1.0
                                    self.ne += 1
                    # Storing variables for ESS
                    # TODO: Modify the constraint for the first period
                    # TODO: create an input for storage without the 
                    # Link List
                        if self.Storage['Number'] > 0:
                            if j > 0: # Start only after the first period
                                for jj in range(self.Storage['Number']):
                                    self.ia[self.ne] = \
                                        self.activepowerbalancenode\
                                        [i, j, k][1] + ii
                                    self.ja[self.ne] = \
                                        self.ESS[i, j][1] + jj
                                    self.ar[self.ne] = \
                                        self.Storage['Efficiency'][jj] \
                                        / self.scenarios['Weights'][j - 1]
                                    self.ne += 1
                                for k in range(self.Storage['Number']):
                                    self.ia[self.ne] = \
                                        self.activepowerbalancenode\
                                        [i, j, k][1] + ii
                                    self.ja[self.ne] = \
                                        self.ESS[i, j - 1][1] + jj
                                    self.ar[self.ne] = \
                                        -self.Storage['Efficiency'][jj] \
                                        / self.scenarios['Weights'][j - 1]
                                    self.ne += 1
                    # Storing the variables for pumps
                    # TODO: create an input for storage without the 
                    # Link List
                        if self.pumps['Number'] > 0:
                            for jj in range(self.pumps['Number']):
                                if self.pumps['Max'][jj] > 0:
                                    self.ia[self.ne] = \
                                        self.activepowerbalancenode\
                                            [i, j, k][1] + ii
                                    self.ja[self.ne] = \
                                        self.pumpsvar[i, j][1] + jj
                                    self.ar[self.ne] = -1.0
                                    self.ne += 1
                    # Storing the variables for active power flows
                        for jj in range(self.ENetwork.get_NoBra()):
                            if self.ENetwork.Branch[jj].get_BusF() ==\
                                self.ENetwork.Bus[ii].get_Number():
                                self.ia[self.ne] = \
                                    self.activepowerbalancenode\
                                        [i, j, k][1] + ii
                                self.ja[self.ne] = \
                                    self.ActivePowerFlow[i, j, k][1] + jj
                                self.ar[self.ne] = -1.0
                                self.ne += 1
                            if self.ENetwork.Branch[jj].get_BusT() ==\
                                self.ENetwork.Bus[ii].get_Number():
                                self.ia[self.ne] = \
                                    self.activepowerbalancenode\
                                        [i, j, k][1] + ii
                                self.ja[self.ne] = \
                                    self.ActivePowerFlow[i, j, k][1] + jj
                                self.ar[self.ne] = 1.0
                                self.ne += 1
                    # Storing the variables for active power losses
                        if self.settings['Losses']:
                            for jj in range(self.ENetwork.get_NoBra()):
                                if self.ENetwork.Branch[jj].get_BusF() ==\
                                    self.ENetwork.Bus[ii].get_Number() or \
                                    self.ENetwork.Branch[jj].get_BusT() ==\
                                    self.ENetwork.Bus[ii].get_Number():
                                    self.ia[self.ne] = \
                                        self.activepowerbalancenode\
                                            [i, j, k][1] + ii
                                    self.ja[self.ne] = \
                                        self.ActivePowerLosses[i, j, k][1]\
                                            + jj
                                    self.ar[self.ne] = -0.5
                                    self.ne += 1
                    # Storing the variables for load curtailment
                        self.ia[self.ne] = \
                            self.activepowerbalancenode[i, j, k][1] + ii
                        self.ja[self.ne] = \
                            self.LoadCurtailmentNode[i, j, k][1] + ii
                        self.ar[self.ne] = 1.0
                        self.ne += 1

                    # Defining the resources (b) for the constraints
                        totaldemand = 0                
                        # TODO: Change the inputs of losses and demand scenarios
                        # for parameters
                        if self.scenarios['NoDem'] == 0:
                            totaldemand = totaldemand + \
                                self.busData[ii] * \
                                self.scenarios['Demand']\
                                    [self.busScenario[ii][i]]
                        else:
                            totaldemand = totaldemand + \
                                self.busData[ii] * \
                                self.scenarios['Demand']\
                                    [j+self.busScenario[ii][i]]

                        self.solver.set_row_bnds(\
                            str(self.activepowerbalancenode[i, j, k][0]), ii,\
                            'fixed', totaldemand, totaldemand)

    def activepowerflowconstraints(self):
        """ This class method writes the active power flow constraints in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """
        # Generating the matrix A for the active power flow constraints
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                for k in range(len(self.settings['Security']) + 1):
                    # Pre-contingency
                    if k == 0:
                        for ii in range(self.ENetwork.get_NoBra()):
                        # Storing the active power flow variables
                            self.ia[self.ne] = \
                                self.activepowerflowconstraint[i, j, k][1] + ii
                            self.ja[self.ne] = \
                                self.ActivePowerFlow[i, j, k][1] + ii
                            self.ar[self.ne] = 1.0
                            self.ne += 1
                        # Storing the voltage angle variables at end "from"
                            self.ia[self.ne] = \
                                self.activepowerflowconstraint[i, j, k][1] + ii
                            self.ja[self.ne] = \
                                self.VoltageAngle[i, j, k][1] + \
                                    self.ENetwork.Branch[ii].get_PosF()
                            self.ar[self.ne] = \
                                -1.0/self.ENetwork.Branch[ii].get_X()
                            self.ne += 1
                        # Storing the voltage angle variables at end "to"
                            self.ia[self.ne] = \
                                self.activepowerflowconstraint[i, j, k][1] + ii
                            self.ja[self.ne] = \
                                self.VoltageAngle[i, j, k][1] + \
                                    self.ENetwork.Branch[ii].get_PosT()
                            self.ar[self.ne] = \
                                1.0/self.ENetwork.Branch[ii].get_X()
                            self.ne += 1

                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.activepowerflowconstraint[i, j, k][0]),\
                                    ii, 'fixed', 0, 0)
                    # Post-contingency
                    else:
                        counter = 0
                        for ii in range(self.ENetwork.get_NoBra()):
                            if self.ENetwork.Branch[ii].is_active(k):
                            # Storing the active power flow variables
                                self.ia[self.ne] = \
                                    self.activepowerflowconstraint[i, j, k][1] \
                                        + counter
                                self.ja[self.ne] = \
                                    self.ActivePowerFlow[i, j, k][1] + ii
                                self.ar[self.ne] = 1.0
                                self.ne += 1
                            # Storing the voltage angle variables at end "from"
                                self.ia[self.ne] = \
                                    self.activepowerflowconstraint[i, j, k][1] \
                                        + counter
                                self.ja[self.ne] = \
                                    self.VoltageAngle[i, j, k][1] + \
                                        self.ENetwork.Branch[ii].get_PosF()
                                self.ar[self.ne] = \
                                    -1.0/self.ENetwork.Branch[ii].get_X()
                                self.ne += 1
                            # Storing the voltage angle variables at end "to"
                                self.ia[self.ne] = \
                                    self.activepowerflowconstraint[i, j, k][1] \
                                        + counter
                                self.ja[self.ne] = \
                                    self.VoltageAngle[i, j, k][1] + \
                                        self.ENetwork.Branch[ii].get_PosT()
                                self.ar[self.ne] = \
                                    1.0/self.ENetwork.Branch[ii].get_X()
                                self.ne += 1
                                
                            # Defining the resources (b) for the constraints
                                self.solver.set_row_bnds(\
                                    str(self.activepowerflowconstraint\
                                        [i, j, k][0]), counter, 'fixed', 0, 0)
                                counter += 1                            

    def activepowerlosses1constraints(self):
        """ This class method writes the active power losses constraints in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """
        # Generating the matrix A for the active power losses constraints
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                for k in range(len(self.settings['Security']) + 1):
                    # Pre-contingency
                    if k == 0:
                        for ii in range(self.ENetwork.get_NoBra()):
                            for jj in range(self.Number_LossCon):
                            # Storing the active power losses variables
                                self.ia[self.ne] = \
                                    self.activepowerlosses1[i, j, k, ii][1] \
                                        + jj
                                self.ja[self.ne] = \
                                    self.ActivePowerLosses[i, j, k][1] \
                                        + ii
                                self.ar[self.ne] = 1.0
                                self.ne += 1
                            # Storing the active power flow variables
                                self.ia[self.ne] = \
                                    self.activepowerlosses1[i, j, k, ii][1] \
                                        + jj
                                self.ja[self.ne] = \
                                    self.ActivePowerFlow[i, j, k][1] \
                                        + ii
                                self.ar[self.ne] = -self.ENetwork.loss['B'][jj]\
                                    * self.ENetwork.Branch[ii].get_R()
                                self.ne += 1

                            # Defining the resources (b) for the constraints
                                self.solver.set_row_bnds(\
                                    str(self.activepowerlosses1\
                                        [i, j, k, ii][0]), jj, 'lower', \
                                        self.ENetwork.loss['A'][jj]\
                                        * self.ENetwork.Branch[ii].get_R(), 0)
                                # print("{} ≤ {} - ({})*{} ≤ ∞".format(\
                                #     self.ENetwork.loss['A'][jj]\
                                #     * self.ENetwork.Branch[ii].get_R(), \
                                #     str(self.ActivePowerLosses[i, j, k][0]+\
                                #     ","+str(ii)),-self.ENetwork.loss['B'][jj]\
                                #     * self.ENetwork.Branch[ii].get_R(),\
                                #     str(self.ActivePowerFlow[i, j, k][0]+\
                                #     ","+str(ii))))
                    # Post-contingency
                    else:
                        for ii in range(self.ENetwork.get_NoBra()):
                            if self.ENetwork.Branch[ii].is_active(k):
                                for jj in range(self.Number_LossCon):
                                # Storing the active power losses variables
                                    self.ia[self.ne] = \
                                        self.activepowerlosses1\
                                            [i, j, k, ii][1] + jj
                                    self.ja[self.ne] = \
                                        self.ActivePowerLosses[i, j, k][1] \
                                            + ii
                                    self.ar[self.ne] = 1.0
                                    self.ne += 1
                                # Storing the active power flow variables
                                    self.ia[self.ne] = \
                                        self.activepowerlosses1\
                                            [i, j, k, ii][1] + jj
                                    self.ja[self.ne] = \
                                        self.ActivePowerFlow[i, j, k][1] \
                                            + ii
                                    self.ar[self.ne] = \
                                        -self.ENetwork.loss['B'][jj]\
                                        * self.ENetwork.Branch[ii].get_R()
                                    self.ne += 1

                                # Defining the resources (b) for the constraints
                                    self.solver.set_row_bnds(\
                                        str(self.activepowerlosses1\
                                            [i, j, k, ii][0]), jj, 'lower', \
                                            self.ENetwork.loss['A'][jj]\
                                            * self.ENetwork.Branch[ii].get_R(), 0)

    def activepowerlosses2constraints(self):
        """ This class method writes the active power losses constraints in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """
        # Generating the matrix A for the active power losses constraints
        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
                for k in range(len(self.settings['Security']) + 1):
                    # Pre-contingency
                    if k == 0:
                        for ii in range(self.ENetwork.get_NoBra()):
                            for jj in range(self.Number_LossCon):
                            # Storing the active power losses variables
                                self.ia[self.ne] = \
                                    self.activepowerlosses2[i, j, k, ii][1] \
                                        + jj
                                self.ja[self.ne] = \
                                    self.ActivePowerLosses[i, j, k][1] \
                                        + ii
                                self.ar[self.ne] = 1.0
                                self.ne += 1
                            # Storing the active power flow variables
                                self.ia[self.ne] = \
                                    self.activepowerlosses2[i, j, k, ii][1] \
                                        + jj
                                self.ja[self.ne] = \
                                    self.ActivePowerFlow[i, j, k][1] \
                                        + ii
                                self.ar[self.ne] = self.ENetwork.loss['B'][jj]\
                                    * self.ENetwork.Branch[ii].get_R()
                                self.ne += 1

                            # Defining the resources (b) for the constraints
                                self.solver.set_row_bnds(\
                                    str(self.activepowerlosses2\
                                        [i, j, k, ii][0]), jj, 'lower', \
                                        self.ENetwork.loss['A'][jj]\
                                        * self.ENetwork.Branch[ii].get_R(), 0)
                    # Post-contingency
                    else:
                        for ii in range(self.ENetwork.get_NoBra()):
                            if self.ENetwork.Branch[ii].is_active(k):
                                for jj in range(self.Number_LossCon):
                                # Storing the active power losses variables
                                    self.ia[self.ne] = \
                                        self.activepowerlosses2\
                                            [i, j, k, ii][1] + jj
                                    self.ja[self.ne] = \
                                        self.ActivePowerLosses[i, j, k][1] \
                                            + ii
                                    self.ar[self.ne] = 1.0
                                    self.ne += 1
                                # Storing the active power flow variables
                                    self.ia[self.ne] = \
                                        self.activepowerlosses2\
                                            [i, j, k, ii][1] + jj
                                    self.ja[self.ne] = \
                                        self.ActivePowerFlow[i, j, k][1] \
                                            + ii
                                    self.ar[self.ne] = \
                                        self.ENetwork.loss['B'][jj]\
                                        * self.ENetwork.Branch[ii].get_R()
                                    self.ne += 1

                                # Defining the resources (b) for the constraints
                                    self.solver.set_row_bnds(\
                                        str(self.activepowerlosses2\
                                            [i, j, k, ii][0]), jj, 'lower', \
                                            self.ENetwork.loss['A'][jj]\
                                            * self.ENetwork.Branch[ii].get_R(), 0)


class EnergyandNetwork(Energymodel, Networkmodel):
    """ This class builds and solve the energy and network models(NM) 
    using the gplk wrapper.

    The information of the pyeneClass is passed to this class,
    which provides the parameters for the model. Furthermore,
    the GLPKSolver class that contains the GLPK wrapper is imported """

    number_variablesENM = 0
    number_constraintsENM = 0

    def __init__(self, obj1=None, obj2=None, obj3=None):
        """
        Parameters
        ----------
        obj1 : Energy object
            Information of the energy tree
        obj2 : Network object
            Information of the power system
        """

        # Storing data input - Parameters
        self.LLNodesAfter = obj1.tree['After']

        # Storing the data of other objects
        Energymodel.__init__(self, obj1)
        Networkmodel.__init__(self, obj2)

                # Copy attributes
        for pars in obj3.__dict__.keys():
            setattr(self, pars, getattr(obj3, pars))


    def optimisationENM(self):
        """ This class method solve the optimisation problem """
        # Creation of model instance
        self.solver = GLPKSolver(message_level='off')       
        # Definition of minimisation problem
        self.solver.set_dir('min')
        # Definition of the mathematical formulation
        self.EnergyandEconomicDispatchModels()
        ret = self.solver.simplex()
        assert ret == 0

        print('Objective Function: %.10f' %(self.solver.get_obj_val()))

        # for i in self.connections['set']:
        #     print('Case %d :' %(i))
        #     print('')
        #     print('Generation:')
        #     for k in range(len(self.Gen.Conv)):
        #         for j in range(self.settings['NoTime']):
        #             print("%f" %(self.solver.get_col_prim(\
        #                 str(self.thermalgenerators[i, j][0]), k) * \
        #                     self.ENetwork.get_Base()), end = ' ')
        #         print('')
        #     for k in range(len(self.Gen.RES)):
        #         for j in range(self.settings['NoTime']):                
        #             print("%f" %(self.solver.get_col_prim(\
        #                 str(self.RESgenerators[i, j][0]), k) * \
        #                     self.ENetwork.get_Base()), end = ' ')
        #         print('')
        #     for k in range(len(self.Gen.Hydro)):
        #         for j in range(self.settings['NoTime']):
        #             print("%f" %(self.solver.get_col_prim(\
        #                 str(self.Hydrogenerators[i, j][0]), k) * \
        #                     self.ENetwork.get_Base()), end = ' ')
        #         print('')
        #     print('')
        #     if self.pumps['Number'] > 0:
        #         print('Pumps:')
        #         for k in range(self.pumps['Number']):
        #             for j in range(self.settings['NoTime']):
        #                 print("%f" %(self.solver.get_col_prim(\
        #                     str(self.pumpsvar[i, j][0]), k) * \
        #                         self.ENetwork.get_Base()), end = ' ')
        #             print('')
        #         print('')
        #     if len(self.Gen.Conv) > 0:
        #         print('Thermal Generation cost:')
        #         for k in range(len(self.Gen.Conv)):
        #             for j in range(self.settings['NoTime']):
        #                 print("%f" %(self.solver.get_col_prim(\
        #                     str(self.thermalCG[i, j][0]), k)), end = ' ')
        #             print('')
        #         print('')
        #     if len(self.Gen.RES) > 0:
        #         print('RES Generation cost:')
        #         for k in range(len(self.Gen.RES)):
        #             for j in range(self.settings['NoTime']):
        #                 print("%f" %(self.solver.get_col_prim(\
        #                     str(self.RESCG[i, j][0]), k)), end = ' ')
        #             print('')
        #         print('')
        #     if len(self.Gen.Hydro) > 0:
        #         print('Hydro Generation cost:')
        #         for k in range(len(self.Gen.Hydro)):
        #             for j in range(self.settings['NoTime']):
        #                 print("%f" %(self.solver.get_col_prim(\
        #                     str(self.HydroCG[i, j][0]), k)), end = ' ')
        #             print('')
        #         print('')

        #     if self.settings['Flag']:
        #     # Optimal Power Flow
        #         print('Voltage angle:')
        #         for k in range(len(self.settings['Security']) + 1):
        #             print('Contingency %d :' %(k))
        #             for ii in range(self.ENetwork.get_NoBus()):
        #                 for j in range(self.settings['NoTime']):
        #                     print("%f" %(self.solver.get_col_prim(\
        #                         str(self.VoltageAngle[i, j, k][0]), ii)),\
        #                             end = ' ')
        #                 print('')
        #             print('')
        #         print('Load Curtailment:')
        #         for k in range(len(self.settings['Security']) + 1):
        #             print('Contingency %d :' %(k))
        #             for ii in range(self.ENetwork.get_NoBus()):
        #                 for j in range(self.settings['NoTime']):
        #                     print("%f" %(self.solver.get_col_prim(\
        #                         str(self.LoadCurtailmentNode[i, j, k][0]), ii)\
        #                             * self.ENetwork.get_Base()), end = ' ')
        #                 print('')
        #             print('')
        #         print('Active Power Flow:')
        #         for k in range(len(self.settings['Security']) + 1):
        #             print('Contingency %d :' %(k))
        #             for ii in range(self.ENetwork.get_NoBra()):
        #                 for j in range(self.settings['NoTime']):
        #                     print("%f" %(self.solver.get_col_prim(\
        #                         str(self.ActivePowerFlow[i, j, k][0]), ii)\
        #                             * self.ENetwork.get_Base()), end = ' ')
        #                 print('')
        #             print('')
        #         if self.settings['Losses']:
        #             print('Active Power Losses:')
        #             for k in range(len(self.settings['Security']) + 1):
        #                 print('Contingency %d :' %(k))
        #                 for ii in range(self.ENetwork.get_NoBra()):
        #                     for j in range(self.settings['NoTime']):
        #                         print("%f" %(self.solver.get_col_prim(\
        #                             str(self.ActivePowerLosses[i, j, k][0]),\
        #                                 ii) * self.ENetwork.get_Base()),\
        #                                     end = ' ')
        #                     print('')
        #                 print('')
        #         print('\n\n')
        #     else:
        #     # Economic dispatch
        #         print('Load Curtailment:')
        #         for j in range(self.settings['NoTime']):
        #             print("%f" %(self.solver.get_col_prim(\
        #                         str(self.loadcurtailmentsystem[i, j][0]), 0) * \
        #                             self.ENetwork.get_Base()), end = ' ')
        #         print('\n\n')
        #         print('')

        # for i in range(self.NumberTrees):
        #     print("vector %d:" %(i))
        #     for j in range(self.TreeNodes):
        #          print("%f %f" %(self.solver.get_col_prim(str(\
        #              self.Partialstorage[i][0]), j), \
        #                 self.solver.get_col_prim(str(self.Totalstorage[i][0]),\
        #                 j)))

    def EnergyandEconomicDispatchModels(self):
        """ This class method builds the optimisation model
        for the energy and economic dispatch problem """
        # Function to determine de number of variables in the energy model
        self.dnvariablesEM()
        # Function to determine de number of constraints in the energy 
        # model
        self.dnconstraintsEM()
        # Creation of variables for the energy model in 
        # glpk (matrix A)
        self.variablesEM()


        if self.settings['Flag']:
            # Function to determine de number of variables in the optimal
            # power flow
            self.dnvariablesOPF()
            # Function to determine de number of constraints in the optimal
            # power flow
            self.dnconstraintsOPF()
            # Number of variables in the Energy and Optimal Power Flow models
            self.number_variablesENM = self.number_variablesOPF + \
                self.number_variablesEM
            # Number of constraints in the Energy and Optimal Power Flow models
            self.number_constraintsENM = self.number_constraintsOPF + \
                self.number_constraintsEM

            # Creation of variables for the Optimal Power Flow in 
            # glpk (matrix A)
            self.variablesOPF()

        else:
            # Function to determine de number of variables in the economic 
            # dispatch
            self.dnvariablesED()
            # Function to determine de number of constraints in the economic 
            # dispatch
            self.dnconstraintsED()
            # Number of variables in the Energy and Economic Dispatch models
            self.number_variablesENM = self.number_variablesED + \
                self.number_variablesEM
            # Number of constraints in the Energy and Economic Dispatch models
            self.number_constraintsENM = self.number_constraintsED + \
                self.number_constraintsEM

            # Creation of variables for the economic dispatch in 
            # glpk (matrix A)
            self.variablesED()

        self.coeffmatrixENM()

        self.Objective_functionENM()

    def coeffmatrixENM(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) 
        # to be later added to glpk
        self.ia = np.empty(math.ceil(self.number_constraintsENM * \
            self.number_variablesENM / 3), dtype=int) # Position in rows
        self.ja = np.empty(math.ceil(self.number_constraintsENM * \
            self.number_variablesENM / 3), dtype=int) # Position in columns
        self.ar = np.empty(math.ceil(self.number_constraintsENM * \
            self.number_variablesENM / 3), dtype=float) # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A

        self.constraintsEM()
        self.Energybalance()
        self.Aggregation()
        # if self.LL['NosUnc'] != 0:
        #     self.AggregationStochastic()

        if self.settings['Flag']:
            self.constraintsOPF()
            self.activepowerbalancepernode()
            self.activepowerflowconstraints()
            if self.settings['Losses']:
                self.activepowerlosses1constraints()
                self.activepowerlosses2constraints()
        else:
            self.constraintsED()
            self.activepowerbalancesystem()

        if len(self.Gen.Hydro) > 0:
            self.constraintsENM()
            self.releaselimitsvariables()
            self.EnergyandNetworkRelation()

        # Common constraints
        self.piecewiselinearisationcost()
        self.generationrampsconstraints()

        self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)

    # Variables ENM

    def releaselimitsvariables(self):
        """ This class method release the bounds of variables that were fixed
        for individual models but that need to be released for the calculations
        of the energy and economic dispatch in glpk 
        
        The released variables belong to:
        Energy model
        """
        for i in range(self.NumberTrees):
            for j in self.connections['set']:
                self.solver.set_col_bnds(\
                    str(self.OutputsTree[i][0]), self.p['pyeneE'][j], 'lower', \
                        0, sys.float_info.max)      

    # Constraints ENM

    def posconstraintsENM(self):
        """ This class method creates the vectors that store the positions
        of contraints that links the energy and ED problems """
        # Creating the matrices to store the position of constraints in
        # matrix A
        self.connectionNetworkandEnergy = np.empty((self.NumberTrees,\
            len(self.connections['set'])), dtype=[('napos', 'U80'),\
                ('nupos', 'i4')]) # Start position 
                    # of energy and economic dispatch constraints (rows)             
    
    def constraintsENM(self):
        """ This class method reserves the space in glpk for the constraints of
        that links the energy and economic dispatch problems """

        self.posconstraintsENM()

        for i in range(self.NumberTrees):
            for j in self.connections['set']:
                self.connectionNetworkandEnergy[i, j] = ('CEED'+str(i)+str(j),\
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
        for i in range(self.NumberTrees): # Vectors is equal to the number
            # of hydro generators (rivers) TODO: Explain this better and 
            # separate the data for this
            for j in self.connections['set']:
                # Storing the variables for the total storage of the tree
                self.ia[self.ne] = self.connectionNetworkandEnergy[i, j][1]
                self.ja[self.ne] = self.OutputsTree[i][1] + \
                    self.p['pyeneE'][j]
                self.ar[self.ne] = 1.0
                self.ne += 1
                for k in range(self.settings['NoTime']):
                    self.ia[self.ne] = self.connectionNetworkandEnergy[i, j][1]
                    self.ja[self.ne] = \
                        self.Hydrogenerators[j, k][1] + i
                    self.ar[self.ne] = -self.scenarios['Weights'][k] * \
                        self.ENetwork.get_Base()
                    self.ne += 1
                # Defining the resources (b) for the constraints
                self.solver.set_row_bnds(\
                    str(self.connectionNetworkandEnergy[i, j][0]), 0,\
                    'fixed', 0, 0)
        
    # Objective function ENM

    def Objective_functionENM(self):
        """ This class method defines the objective function of the economic
        dispatch in glpk """

        # Calculating the aggregated weights for the last nodes in the tree
        # TODO: explain the aggregated weights better
        
        WghtAgg = 0 + self.WeightNodes
        OFaux = np.ones(len(self.connections['set']), dtype=float)
        xp = 0
        for xn in range(self.TreeNodes):
            aux = self.LLNodesAfter[xn][0]
            if aux != 0:
                for xb in range(aux, self.LLNodesAfter[xn][1] + 1):
                    WghtAgg[xb] *= WghtAgg[xn]
            else:
                OFaux[xp] = WghtAgg[xn]
                xp += 1

        for i in self.connections['set']:
            for j in range(self.settings['NoTime']):
            # Cost for conventional generation    
                if len(self.Gen.Conv) > 0: 
                    for k in range(len(self.Gen.Conv)):
                        self.solver.set_obj_coef(\
                            str(self.thermalCG[i, j][0]),\
                            k, OFaux[i] * self.scenarios['Weights'][j])
            # Cost for RES generation    
                if len(self.Gen.RES) > 0: 
                    for k in range(len(self.Gen.RES)):
                        self.solver.set_obj_coef(\
                            str(self.RESCG[i, j][0]),\
                            k, OFaux[i] * self.scenarios['Weights'][j])
            # Cost for Hydroelectric generation    
                if len(self.Gen.Hydro) > 0: 
                    for k in range(len(self.Gen.Hydro)):
                        self.solver.set_obj_coef(\
                            str(self.HydroCG[i, j][0]),\
                            k, OFaux[i] * self.scenarios['Weights'][j])
            # Operation cost of pumps
                if self.pumps['Number'] > 0:
                    for k in range(self.pumps['Number']):
                        self.solver.set_obj_coef(\
                            str(self.pumpsvar[i, j][0]),\
                            k, -OFaux[i] * self.scenarios['Weights'][j] \
                                * self.ENetwork.get_Base() \
                                    * self.pumps['Value'][k])
            # Punitive cost for load curtailment
                if self.settings['Flag']:
                # Optimal Power Flow
                    for k in range(len(self.settings['Security']) + 1):
                        for ii in range(self.ENetwork.get_NoBus()):
                            self.solver.set_obj_coef(\
                                str(self.LoadCurtailmentNode[i, j, k][0]),\
                                ii, OFaux[i] * self.scenarios['Weights'][j] \
                                    * self.Penalty)
                else:
                # Economic Dispatch
                # TODO: Set a parameter penalty in pyeneN
                    self.solver.set_obj_coef(\
                                str(self.loadcurtailmentsystem[i, j][0]),\
                                0, OFaux[i] * self.scenarios['Weights'][j] \
                                    * self.Penalty)