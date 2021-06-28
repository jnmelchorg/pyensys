"""
Pyene Models provides a glpk implementation of different methods
for: 
1) Balancing multiple vectors at different time aggregation levels.
2) Optimal dispatch of diverse generation technologies without considering
the transmission system (Economic Dispatch)
3) Optimal dispatch of diverse generation technologies considering
the transmission system (Optimal Power Flow)

@author: Dr Jose Nicolas Melchor Gutierrez
"""

import copy
import numpy as np
import sys
import importlib
import collections
import networkx as nx
from dataclasses import dataclass, field
from typing import Any

try:
    cpp_energy_wrapper = importlib.import_module(\
        '.engines.cython.cpp_energy_wrapper', package="pyensys")
    models_cpp = cpp_energy_wrapper.models_cpp
except ImportError as err:
    print('Error:', err)

try:
    GLPKSolver_wrapper = importlib.import_module(\
        '.engines.cython._glpk', package="pyensys")
    GLPKSolver = GLPKSolver_wrapper.GLPKSolver
except ImportError as err:
    print('Error:', err)

@dataclass
class characteristic:
    name                :   str     = ""        # Name characteristic
    value               :   Any     = None      # Value characteristic
    data_type           :   str     = ""        # Type of data

@dataclass
class information:
    characteristics     :   list    = field(default_factory=list)        # list of characteristics that describe the information
    value               :   Any     = None      # Value of parameter
    data_type           :   str     = ""        # Type of data
    def get_characteristic(self, name=None):
        for cha in self.characteristics:
            if cha.name == name:
                return cha.value
        print("characteristic with name *{}* does not exist".format(name))
        return "N/A"
    def exist(self, name=None):
        for cha in self.characteristics:
            if cha.name == name:
                return True
        return False
    def update_characteristic(self, name=None, value=None):
        for cha in self.characteristics:
            if cha.name == name:
                cha.value = value
                return
        print("characteristic with name *{}* does not exist".format(name))

@dataclass
class parameters:
    connections         :   list    = field(default_factory=list)       # list of data with connections
    functions           :   list    = field(default_factory=list)       # list of data with functions
    outputs             :   list    = field(default_factory=list)       # list of data with output options
    data                :   dict    = field(default_factory=dict)       # dictionary with information

@dataclass
class model_options_parameter:
    name                :   str     = None      # Name parameter
    value               :   Any     = None      # Value of parameter
    engine              :   str     = None      # Indicates if the problem should be solved in the 
                                                # energy engine or in the FutureDAMS integrated 
                                                # framework

@dataclass
class tree_parameters:
    name                :   str     = None      # Name parameter
    level               :   int     = None      # Level in the tree
    name_node           :   str     = None      # Name of level, e.g. summer, weekday
    value               :   float   = None      # Value of parameter

@dataclass
class tree_variables:
    name                :   str     = None      # Name parameter
    level               :   int     = None      # Level in the tree
    name_node           :   str     = None      # Name of level, e.g. summer, weekday
    value               :   float   = None      # Value of the solution for this specific variable
    max                 :   float   = None      # Upper limit of the variable
    min                 :   float   = None      # Lower limit of the variable

@dataclass
class nodes_info_tree:
    level               :   int     = None      # Level in the tree
    name_node           :   str     = None      # Name of level, e.g. summer, weekday
    node                :   int     = None      # Number of node in graph
    parameters          :   list    = None      # Parameters associated to the node in the graph
    variables           :   list    = None      # Variables associated to the node in the graph

@dataclass
class network_variable:
    name                :   str     = None      # Name of the variable
    position_tree       :   dict    = None      # Position in the energy tree - representative days
    hour                :   int     = None      # Hour of the solution in case of multiple hours
    ID                  :   str     = None      # ID of element
    type                :   str     = None      # Type of element, e.g. bus, branch
    value               :   float   = None      # Value of the solution for this specific variable
    max                 :   float   = None      # Upper limit of the variable
    min                 :   float   = None      # Lower limit of the variable

@dataclass
class network_parameter:
    name                :   str     = None      # Name of the parameter
    position_tree       :   dict    = None      # Position in the energy tree - representative days
                                                # in case of parameters changing in time
    hour                :   int     = None      # Hour of the parameter in case of parameters 
                                                # changing in time
    ID                  :   str     = None      # ID of element
    type                :   str     = None      # Type of element, e.g. bus, branch                                                # related
    value               :   Any     = None      # Value of specific parameter

@dataclass
class nodes_info_network:
    type                :   str     = None      # Type of element, e.g. bus, branch, generator
    ID                  :   str     = None      # ID of element
    node                :   int     = None      # Number of node in graph
    parameters          :   list    = None      # Parameters associated to the node in the graph
    variables           :   list    = None      # Variables associated to the node in the graph

#      NAMES ELEMENTS NODES

#TODO: Define characteristics and parameters properly across all files

integer_characteristics_nodes = [
    "number",
    "typePF"
]

double_characteristics_nodes =[    
]

string_characteristics_nodes =[
    "ID",
    "name_bus",
    "zone",
    "group",
    "subtype"
]

bool_characteristics_nodes = []

integer_parameters_nodes =[
]

double_parameters_nodes =[
    "Pd",
    "Qd",
    "Gs",
    "Bs",
    "baseV",
    "Vmpr",
    "Vapr",
    "Vmax",
    "Vmin"
]

string_parameters_nodes =[
]

bool_parameters_nodes =[]

#      NAMES ELEMENTS BRANCHES

integer_characteristics_branches =[
    "from",
    "to"
]

double_characteristics_branches =[
]

string_characteristics_branches =[
    "ID",
    "subtype",
    "group"
]

bool_characteristics_branches =[
]

integer_parameters_branches =[
]

double_parameters_branches =[
    "resistance",
    "reactance",
    "LCsusceptance",
    "maxPflow",
    "CTEP",
    "Vmpr",
    "Vapr",
    "Vmax",
    "Vmin"
]

string_parameters_branches =[
]

bool_parameters_branches =[
    "status",
    "vTEP"
]

#      NAMES ELEMENTS GENERATORS

integer_characteristics_generators =[
    "number"
]

double_characteristics_generators =[
]

string_characteristics_generators =[
    "ID",
    "subtype",
    "group"
]

bool_characteristics_generators =[
]

integer_parameters_generators =[
]

double_parameters_generators =[
    "Pmax",
    "Pmin",
    "Pfix",
    "Qmax",
    "Qmin",
    "Qfix",
    "cUC",
    "cGEP",
    "fCPg",
    "vCPg",
    "emissions",
    "startup",
    "shutdown",
    "cost function"
]

string_parameters_generators =[
]

bool_parameters_generators =[
    "status",
    "vUC",
    "vGEP"
]

integer_characteristics = copy.copy(integer_characteristics_nodes)
integer_characteristics.extend(integer_characteristics_branches)
integer_characteristics.extend(integer_characteristics_generators)
integer_characteristics = list(dict.fromkeys(integer_characteristics))

double_characteristics = copy.copy(double_characteristics_nodes)
double_characteristics.extend(double_characteristics_branches)
double_characteristics.extend(double_characteristics_generators)
double_characteristics = list(dict.fromkeys(double_characteristics))

string_characteristics = copy.copy(string_characteristics_nodes)
string_characteristics.extend(string_characteristics_branches)
string_characteristics.extend(string_characteristics_generators)
string_characteristics = list(dict.fromkeys(string_characteristics))

bool_characteristics = copy.copy(bool_characteristics_nodes)
bool_characteristics.extend(bool_characteristics_branches)
bool_characteristics.extend(bool_characteristics_generators)
bool_characteristics = list(dict.fromkeys(bool_characteristics))

all_characteristics = copy.copy(integer_characteristics)
all_characteristics.extend(double_characteristics)
all_characteristics.extend(string_characteristics)
all_characteristics.extend(bool_characteristics)
all_characteristics = list(dict.fromkeys(all_characteristics))

integer_parameters = copy.copy(integer_parameters_nodes)
integer_parameters.extend(integer_parameters_branches)
integer_parameters.extend(integer_parameters_generators)

double_parameters = copy.copy(double_parameters_nodes)
double_parameters.extend(double_parameters_branches)
double_parameters.extend(double_parameters_generators)

string_parameters = copy.copy(string_parameters_nodes)
string_parameters.extend(string_parameters_branches)
string_parameters.extend(string_parameters_generators)

bool_parameters = copy.copy(bool_parameters_nodes)
bool_parameters.extend(bool_parameters_branches)
bool_parameters.extend(bool_parameters_generators)

# MODEL INFORMATION

problems_names_nodes = [
    "SEP"
]

problems_names_branches = [
    "TEP"
]

problems_names_generators = [
    "UC",
    "GEP"
]

problems_names_system = [
    "DC ED",
    "DC OPF",
    "AC PF",
    "NetR",
    "BT"
]

bool_model_characteristics = [
    "loss",
    "multiperiod",
    "MOEA",
    "representative periods"
]

double_model_characteristics = [
    "Sbase"
]

string_model_characteristics = [
    "solver",
    "engine",
    "output file name"
]

bool_model_parameters = copy.copy(problems_names_system)
bool_model_parameters.extend(problems_names_nodes)
bool_model_parameters.extend(problems_names_branches)
bool_model_parameters.extend(problems_names_generators)
bool_model_parameters.extend(bool_model_characteristics)

double_model_parameters = copy.copy(double_model_characteristics)

string_model_parameters = copy.copy(string_model_characteristics)

class models():
    def __init__(self):
        self.data = parameters()
        self.model_options = []
        self.tree_parameters = []
        self.network_parameters = []
    
    # def _create_nodes_graph_tree(self):

    #     # Getting levels of symetric tree
    #     levels = []
    #     for parameter in self.tree_parameters:
    #         if parameter.level not in levels:
    #             levels.append(parameter.level)
    #     levels.sort()

    #     # Getting elements per level
    #     elements_levels = []
    #     for level in levels:
    #         elements_level = []
    #         for parameter in self.tree_parameters:
    #             if parameter.level == level and parameter.name_node not in elements_level:
    #                 elements_level.append(parameter.name_node)
    #         elements_levels.append(elements_level)

    #     # Creating nodes
    #     nodes_graph = []
    #     counter = 0
    #     nodes_levels = 1
    #     # Creating list of nodes
    #     for level, names in zip(levels, elements_levels):
    #         # Creating nodes
    #         for _ in range(nodes_levels):
    #             for name in names:
    #                 node = nodes_info_tree(level=level, name_node=name, node=counter, parameters=[], variables=[])
    #                 counter = counter + 1
    #                 nodes_graph.append(node)
    #         nodes_levels = nodes_levels * len(names)

    #     # Adding parameters to nodes

    #     for parameter in self.tree_parameters:
    #         for node_g in nodes_graph:
    #             if parameter.level == node_g.level and node_g.name_node == parameter.name_node:
    #                 node_g.parameters.append(parameter)
    #                 break
        
    #     # Adding nodes to graph
    #     for node_g in nodes_graph:
    #         self.tree.add_node(node_g.node, obj=node_g)
        
    #     return levels, elements_levels

    # def _create_edges_graph_tree(self, levels, elements_levels):
    #     # Creating branches of graph
    #     branches_graph = []
    #     connected_nodes = [False for _ in self.tree]
    #     for aux in range(len(levels) - 1):
    #         for elements_pre in elements_levels[aux]:
    #             for elements_pos in elements_levels[aux + 1]:
    #                 origin = None
    #                 destiny = None
    #                 for pos, node_g in enumerate(self.tree.nodes(data=True)):
    #                     if node_g[1]['obj'].level == levels[aux] and node_g[1]['obj'].name_node == elements_pre:
    #                         connected_nodes[pos] = True
    #                         origin = node_g[1]['obj'].node
    #                     elif node_g[1]['obj'].level == levels[aux + 1] and node_g[1]['obj'].name_node == elements_pos and not connected_nodes[pos]:
    #                         connected_nodes[pos] = True
    #                         destiny = node_g[1]['obj'].node
    #                     if origin is not None and destiny is not None:
    #                         branches_graph.append([origin, destiny])
    #                         break

    #     for branches in  branches_graph:
    #         self.tree.add_edge(branches[0], branches[1])

    # def _create_graph_tree(self):
    #     self.tree = nx.MultiDiGraph()
    #     levels, elements_levels = self._create_nodes_graph_tree()
    #     self._create_edges_graph_tree(levels, elements_levels)

    # def _create_nodes_graph_network(self):
    #     nodes_graph = []
    #     exist = False
    #     counter = 0
    #     # Creating list of nodes  and adding parameters
    #     for parameter in self.network_parameters:
    #         if nodes_graph:
    #             for node_g in nodes_graph:
    #                 if node_g.ID == parameter.ID:
    #                     exist = True
    #                     node_g.parameters.append(parameter)
    #                     break
    #         if not exist:
    #             node = nodes_info_network(type=parameter.type, ID=parameter.ID, node=counter, parameters=[parameter], variables=[])
    #             counter += 1
    #             nodes_graph.append(node)
    #         exist = False
        
    #     # Adding nodes to graph
    #     for node_g in nodes_graph:
    #         self.network.add_node(node_g.node, obj=node_g)

    # def _create_edges_graph_network(self):
    #     # Creating branches of graph
    #     branches_graph = []
    #     for node_g in self.network.nodes(data=True):
    #         if node_g[1]['obj'].type == "generator":
    #             for gen in node_g[1]['obj'].parameters:
    #                 if gen.name == "number":
    #                     bus_gen = gen.value
    #                     break
    #             for aux in self.network.nodes(data=True):
    #                 if aux[1]['obj'].type == "bus":
    #                     for node in aux[1]['obj'].parameters:
    #                         if node.name == "number":
    #                             bus = node.value
    #                             break
    #                     if bus_gen == bus:
    #                         branches_graph.append([aux[1]['obj'].node, node_g[1]['obj'].node])
    #                         break
    #         elif node_g[1]['obj'].type == "branch":
    #             flag = [False, False]
    #             for branch in node_g[1]['obj'].parameters:
    #                 if branch.name == "from":
    #                     frm = branch.value
    #                     flag[0] = True
    #                 elif branch.name == "to":
    #                     to = branch.value
    #                     flag[1] = True
    #                 if flag[0] and flag[1]:
    #                     break
    #             flag = [False, False]
    #             for aux in self.network.nodes(data=True):
    #                 if aux[1]['obj'].type == "bus":
    #                     for node in aux[1]['obj'].parameters:
    #                         if node.name == "number":
    #                             bus = node.value
    #                             break
    #                     if bus == frm:                       
    #                         branches_graph.append([aux[1]['obj'].node, node_g[1]['obj'].node])
    #                         flag[0] = True
    #                     elif bus == to:
    #                         branches_graph.append([aux[1]['obj'].node, node_g[1]['obj'].node])
    #                         flag[1] = True
    #                 if flag[0] and flag[1]:
    #                     break
    #     for branches in  branches_graph:
    #         self.network.add_edge(branches[0], branches[1])

    # def _create_graph_network(self):
    #     self.network = nx.MultiGraph()
    #     self._create_nodes_graph_network()
    #     self._create_edges_graph_network()

    def _load_network_information_cpp(self):
        for parameter in self.network_parameters:
            if parameter.name not in all_characteristics and parameter.value is not None:
                self.model.create_parameter()
                self.model.load_value("string", b"ID", parameter.ID)
                self.model.load_value("string", b"type", parameter.type)
                self.model.load_value("string", b"name", parameter.name)
                if parameter.position_tree is not None:
                    parameter.position_tree = collections.OrderedDict(sorted(parameter. position_tree.items()))
                    tree = []
                    for val in parameter.position_tree.values():
                        tree.append(val)
                    self.model.load_value("v_string", b"pt", tree)
                subtype = []
                for par in self.network_parameters:
                    if par.name == "subtype" and parameter.ID == par.ID and parameter.type == par.type and par.value is not None:
                        subtype.append(par.value)
                    elif par.name != "subtype" and par.name in integer_characteristics and parameter.ID == par.ID and parameter.type == par.type and par.value is not None:
                        self.model.load_value("integer", par.name.encode('utf-8'), par.value)
                    elif par.name != "subtype" and par.name in double_characteristics and parameter.ID == par.ID and parameter.type == par.type and par.value is not None:
                        self.model.load_value("double", par.name.encode('utf-8'), par.value)
                    elif par.name != "subtype" and par.name in string_characteristics and parameter.ID == par.ID and parameter.type == par.type and par.value is not None:
                        self.model.load_value("string", par.name.encode('utf-8'), str(par.value))
                    elif par.name != "subtype" and par.name in bool_characteristics and parameter.ID == par.ID and parameter.type == par.type and par.value is not None:
                        self.model.load_value("bool", par.name.encode('utf-8'), par.value)
                if subtype:
                    self.model.load_value("v_string", b"subtype", subtype)
                if parameter.hour is not None:
                    self.model.load_value("double", b"hour", parameter.hour)
                if parameter.name in integer_parameters:
                    self.model.load_value("integer", b"value", parameter.value)
                elif parameter.name in double_parameters:
                    self.model.load_value("double", b"value", parameter.value)
                elif parameter.name in string_parameters:
                    self.model.load_value("string", b"value", str(parameter.value))
                elif parameter.name in bool_parameters:
                    self.model.load_value("bool", b"value", parameter.value)
                else:
                    print("parameter {} not identified with any type of data".format(parameter.name))
                self.model.set_parameter(b"network")            

    def _load_tree_information_cpp(self):
        for parameter in self.tree_parameters:
            self.model.create_parameter()
            self.model.load_value("string", b"name", parameter.name)
            self.model.load_value("string", b"name_node", parameter.name_node)
            self.model.load_value("integer", b"level", parameter.level)
            self.model.load_value("double", b"value", parameter.value)
            self.model.set_parameter(b"tree")
    
    def _load_model_information_cpp(self):
        for parameter in self.model_options:
            self.model.create_parameter()
            self.model.load_value("string", b"name", parameter.name)
            if parameter.engine is not None:
                self.model.load_value("string", b"engine", parameter.engine)
            if parameter.name in double_model_parameters:
                self.model.load_value("double", b"value", parameter.value)
            elif parameter.name in string_model_parameters:
                self.model.load_value("string", b"value", str(parameter.value))
            elif parameter.name in bool_model_parameters:
                self.model.load_value("bool", b"value", parameter.value)
            else:
                print("parameter {} not identified with any type of data".format(parameter.name))
            self.model.set_parameter(b"model")
    
    def _load2cpp(self, name=None):
        if name == "connections":
            elements = self.data.connections
        elif name == "functions":
            elements = self.data.functions
            name = "network"
        elif name == "outputs":
            elements = self.data.outputs
        else:
            elements = self.data.data.get(name, None)
            if (name == "bus" or name == "generator" or name == "branch") : name = "network"
        counter = 0
        for element in elements:
            self.model.create_parameter()
            for cha in element.characteristics:
                self.model.load_value(cha.data_type, cha.name.encode('utf-8'), cha.value)
            self.model.load_value(element.data_type, b"value", element.value)
            self.model.set_parameter(name.encode('utf-8'))
            counter = counter + 1

    def _load_information_cpp(self):
        # self._load_network_information_cpp()
        self._load_tree_information_cpp()
        self._load_model_information_cpp()
        self._load2cpp("connections")
        self._load2cpp("functions")
        self._load2cpp("outputs")
        # import cProfile, pstats, io
        # pr = cProfile.Profile()
        # pr.enable()
        self._load2cpp("bus")
        self._load2cpp("generator")
        self._load2cpp("branch")
        # pr.disable()
        # pr.dump_stats('profile_dump')

    def initialise(self):
        # self._create_graph_tree()
        # self._create_graph_network()
        self.model = models_cpp()
        self._load_information_cpp()
        self.model.initialise()
    
    def evaluate(self, subscripts):
        self.subscripts = subscripts
        self.model.create_parameter()
        for key, value in subscripts.items():
            self.model.load_value(value[1], key.encode('utf-8'), value[0])
        self.model.evaluate()

    def get_outputs(self):
        return self.model.return_outputs()

    def get_latest_subscripts(self):
        return self.subscripts
    
    def update_parameter(self, information):
        self.model.create_parameter()
        for key, val in information.items():
            self.model.load_value(val[0], key.encode('utf-8'), val[1])
        return self.model.update_parameter()
    
    def get_moea_variables(self):
        ids, names, min_bnd, max_bnd = self.model.get_MOEA_variables()
        ids_decoded = [ID.decode('utf-8') for ID in ids]
        names_decoded = [name.decode('utf-8') for name in names]
        return ids_decoded, names_decoded, min_bnd, max_bnd
    
    def get_moea_objectives(self):
        names = self.model.get_moea_objectives()
        names_decoded = [name.decode('utf-8') for name in names]
        return names_decoded

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
        assert obj is not None   # If the object is empty then raise an error

        self.tree_parameters = []
        self.tree_variables = []

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
        self.NumberNodesUnc = obj.LL['NosUnc'] # Number of nodes considered
                                # for uncertainty in the temporal tree
        self.LLNodesUnc = obj.p['LLTS3'] # Link List to connect the nodes
                                # with uncertainty in the temporal tree


    def _create_nodes_graph(self):

        # Getting levels of symetric tree
        levels = []
        for parameter in self.tree_parameters:
            if parameter.level not in levels:
                levels.append(parameter.level)
        levels.sort()

        # Getting elements per level
        elements_levels = []
        for level in levels:
            elements_level = []
            for parameter in self.tree_parameters:
                if parameter.level == level and parameter.name_node not in elements_level:
                    elements_level.append(parameter.name_node)
            elements_levels.append(elements_level)

        # Creating nodes
        nodes_graph = []
        counter = 0
        nodes_levels = 1
        # Creating list of nodes
        for level, names in zip(levels, elements_levels):
            # Creating nodes
            for _ in range(nodes_levels):
                for name in names:
                    node = nodes_info_tree
                    node.node = counter
                    counter = counter + 1
                    node.level = level
                    node.name_node = name
                    nodes_graph.append(node)
            nodes_levels = nodes_levels * len(names)

        # Adding parameters to nodes

        for node_g in nodes_graph:
            for parameter in self.tree_parameters:
                if parameter.level == node_g.level and node_g.name_node == parameter.name_node:
                    node_g.parameters.append(parameter)
                    break
        
        # Adding nodes to graph
        for node_g in nodes_graph:
            self.tree.add_node(node_g.node, obj=node_g)
        
        return levels, elements_levels

    def _create_edges_graph(self, levels, elements_levels):
        # Creating branches of graph
        branches_graph = []
        connected_nodes = [False for _ in len(self.tree)]
        for aux in range(len(levels) - 1):
            for elements_pre in elements_levels[aux]:
                for elements_pos in elements_levels[aux + 1]:
                    origin = None
                    destiny = None
                    for pos, node_g in enumerate(self.tree.nodes(data=True)):
                        if node_g['obj'].level == levels[aux] and node_g['obj'].name_node == elements_pre and not connected_nodes[pos]:
                            connected_nodes[pos] = True
                            origin = node_g['obj'].node
                        elif node_g['obj'].level == levels[aux] and node_g['obj'].name_node == elements_pos and not connected_nodes[pos]:
                            connected_nodes[pos] = True
                            destiny = node_g['obj'].node
                        if origin and destiny:
                            branches_graph = [origin, destiny]
                            break

        for branches in  branches_graph:
            self.tree.add_edge(branches[0], branches[1])

    def _create_graph(self):
        self.tree = nx.MultiDiGraph()
        levels, elements_levels = self._create_nodes_graph()
        self._create_edges_graph(levels, elements_levels)

    def initialise(self):
        pass

    def optimisationEM(self, solver_name=None):
        """ This class method solve the optimisation problem """
        # TODO to be expanded with a general optimisation problem       
        # Creation of model instance
        if solver_name == "GLPK":
            self.solver_problem = "GLPK"
            self.solver = GLPKSolver(message_level='all')
            # Definition of minimisation problem
            self.solver.set_dir('min')
            # Definition of the mathematical formulation
            self.modeldefinitionEM()
            ret = self.solver.simplex()
            assert ret == 0, "GLPK could not solve the problem"
        elif solver_name == "CLP":
            self.EnergyTreeCPP()
        else:
            print("incorrect solver has been selected")
        

    def modeldefinitionEM(self):
        """ This class method build and solve the optimisation problem,
         to be expanded with a general optimisation problem """
        # define matrix of coeficients (matrix A)
        self.variablesEM()
        self.coeffmatrixEM()
        self.Objective_functionEM()

    def coeffmatrixEM(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) to be
        # later added to glpk
        self.ia = [] # Position in rows
        self.ja = [] # Position in columns
        self.ar = [] # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A

        self.Energybalance()
        self.Aggregation()
        # if self.NumberNodesUnc != 0:
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
                
    def Energybalance(self):
        """ This class method writes the energy balance in glpk
        
        First, it is reserved space in memory to store the energy balance 
        constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """
        self.treebalance = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
                    # of the tree balance constraints (rows) 
                    # for vector
        
        for i in range(self.NumberTrees):
            self.treebalance[i] = ('TB'+str(i), \
                self.solver.add_rows('TB'+str(i), \
                    (self.TreeNodes - 1)))  # Number of 
                    # rows (constraints) in matrix A for the three balance
                    # for each vector

        # Generating the matrix A for the energy contraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                # Storing the Vin variables
                self.ia.append(self.treebalance[vectors][1] + nodes - 1)
                self.ja.append(self.Partialstorage[vectors][1] + nodes)
                self.ar.append(1)
                # Storing the Vout variables
                self.ne += 1
                self.ia.append(self.treebalance[vectors][1] + nodes - 1)
                self.ar.append(-1)
                if(self.LLEB[nodes, 1] == 0):
                    self.ja.append(self.Partialstorage[vectors][1] + \
                        self.LLEB[nodes, 0])
                elif(self.LLEB[nodes, 1] == 1):
                    self.ja.append(self.Totalstorage[vectors][1] + \
                        self.LLEB[nodes, 0])
                # Storing the Inputs            
                self.ne += 1
                self.ia.append(self.treebalance[vectors][1] + nodes - 1)
                self.ja.append(self.InputsTree[vectors][1] + nodes)
                self.ar.append(-1)
                # Storing the Outputs            
                self.ne += 1
                self.ia.append(self.treebalance[vectors][1] + nodes - 1)
                self.ja.append(self.OutputsTree[vectors][1] + nodes)
                self.ar.append(1)
                self.ne += 1

        # Defining the limits for the energy constraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                self.solver.set_row_bnds(str(self.treebalance[vectors][0]), \
                    nodes - 1, 'fixed', 0, 0)

    def Aggregation(self):
        """ This class method writes the aggregation constraints in glpk
        
        First, it is reserved space in memory to store the aggregation constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        self.treeaggregation = np.empty(self.NumberTrees, \
            dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start position 
                    # of the tree aggregation constraints (rows) 
                    # for vector
        
        for i in range(self.NumberTrees):
            self.treeaggregation[i] = ('TA'+str(i), \
                self.solver.add_rows('TA'+str(i), \
                    (self.TreeNodes - 1)))  # Number of 
                    # rows (constraints) in matrix A for the three aggregation
                    # for each vector

        # Generating the matrix A for the aggregation contraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                # Storing the Vout variables
                self.ia.append(self.treeaggregation[vectors][1] + nodes - 1)
                self.ja.append(self.Totalstorage[vectors][1] + nodes)
                self.ar.append(1)
                # Storing Vin or Vout variables
                self.ne += 1
                self.ia.append(self.treeaggregation[vectors][1] + nodes - 1)
                self.ar.append(-self.WeightNodes[self.LLEA\
                    [nodes, 0]])
                if(self.LLEA[nodes, 2] == 0):
                    self.ja.append(self.Partialstorage[vectors][1]\
                        + self.LLEA[nodes, 1])
                elif(self.LLEA[nodes, 2] == 1):
                    self.ja.append(self.Totalstorage[vectors][1]\
                        + self.LLEA[nodes, 1])
                # Storing Vin or Vout variables
                if(1 - self.WeightNodes[self.LLEA[nodes, 0]] != 0):
                    self.ne += 1
                    self.ia.append(self.treeaggregation[vectors][1]\
                        + nodes - 1)
                    self.ar.append(-(1 - self.WeightNodes\
                        [self.LLEA[nodes, 0]]))
                    if(self.LLEB[self.LLEA[nodes, 0], 1] == 0):
                        self.ja.append(self.Partialstorage[vectors][1] + \
                            self.LLEB[self.LLEA[nodes, 0], 0])
                    elif(self.LLEB[self.LLEA[nodes, 0], 1] == 1):
                        self.ja.append(self.Totalstorage[vectors][1] \
                            + self.LLEB[self.LLEA[nodes, 0], 0])
                self.ne += 1

        # Defining the limits for the aggregation constraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.TreeNodes):
                self.solver.set_row_bnds(str(self.treeaggregation[vectors][0]), \
                    nodes - 1, 'fixed', 0.0, 0.0)


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
        # Generating the matrix A for the aggregation contraints
        # TODO review this constraint
        for vectors in range(self.NumberTrees):
            for nodes in range(2, self.NumberNodesUnc+2): # TODO, does it start from position 2??
                # Storing the first variable of each constraint
                self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                    (self.TreeNodes - 1)) + nodes - 2
                self.ja[self.ne] = (vectors * \
                    (self.TreeNodes)) + \
                    (self.NumberTrees * (self.TreeNodes)) + \
                    self.LLNodesUnc[nodes - 1, 0] + 1
                self.ar[self.ne] = 1
                # Storing the second variable of each constraint
                if(1-self.WeightNodes[self.LLNodesUnc[nodes - 1, 0]] != 0):
                    self.ne += 1
                    self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                        (self.TreeNodes - 1)) + nodes - 2
                    self.ar[self.ne] = -(1-self.WeightNodes[self.LLNodesUnc[nodes - 1, 0]])
                    if(self.LLEB[self.LLNodesUnc[nodes - 1, 0], 1] == 0):
                        self.ja[self.ne] = (vectors * \
                            (self.TreeNodes)) + \
                            self.LLEB[self.LLNodesUnc[nodes - 1, 0], 0] + 1
                    elif(self.LLEB[self.LLNodesUnc[nodes - 1, 0], 1] == 1):
                        self.ja[self.ne] = (vectors * \
                            (self.TreeNodes)) + (self.NumberTrees * \
                            (self.TreeNodes)) + self.LLEB[self.LLNodesUnc[nodes - 1, 0], 0] + 1
                # Storing the third variable
                self.ne += 1
                self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                    (self.TreeNodes - 1)) + nodes - 2
                self.ar[self.ne] = -(\
                    self.WeightNodes[self.LLNodesUnc[nodes - 1, 0]] * \
                    -self.LLNodesUnc[nodes - 1, 2])
                self.ja[self.ne] = (vectors * \
                    (self.TreeNodes)) + self.LLNodesUnc[nodes - 1, 0] + 1
                # Storing variables in the summation
                for aux1 in range(self.LLNodesUnc[nodes - 1, 2] + 1):
                    self.ne += 1
                    self.ia[self.ne] = self.Agg_Sto_row_number + (vectors * \
                        (self.TreeNodes - 1)) + nodes - 2
                    self.ar[self.ne] = -(self.WeightNodes[self.LLNodesUnc[nodes - 1, 0]] * \
                        -self.LLNodesUnc[nodes - 1, 2])
                    self.ja[self.ne] = (vectors * \
                            (self.TreeNodes)) + (self.NumberTrees * \
                            (self.TreeNodes)) + self.LLNodesUnc[nodes, 1] + aux1 + 1
                self.ne += 1

                    

        # Defining the limits for the aggregation constraints
        for vectors in range(self.NumberTrees):
            for nodes in range(1, self.NumberNodesUnc+1):
                self.solver.set_row_bnds('AggStoch', (vectors *  \
                    (self.TreeNodes - 1)) + nodes - 1, 'fixed', \
                    0.0, 0.0)

    # Objective function EM

    def Objective_functionEM(self):
        """ This class method defines the cost coefficients for the
         objective function in glpk
         
        A dummy objective function is created for the problem """

        self.solver.set_obj_coef(str(self.Partialstorage[0][0]), 1, 1)

    #################################
    ###   ENERGY TREE CPP MODELS  ###
    #################################

    def EnergyTreeCPP(self):
        """ This class method builds the optimisation model
        for the optimal power flow problem using a fast implementation of 
        different mathematical models in c++ """

        self.solver_problem = "CLP"
        self.energy_model = models_cpp()
        self.set_parameters_cpp_energy_models()
        self.energy_model.run_energy_tree_cpp()

        # Retrieving solution
        aux_par, aux_tot, aux_in, aux_out = \
            self.energy_model.get_energy_tree_solution_cpp()

        self.PartialStorageSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        self.TotalStorageSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        self.InputsTreeSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        self.OutputsTreeSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        counter = 0
        for i in range(self.NumberTrees):
            for j in range(self.TreeNodes):
                self.PartialStorageSolution[i, j] = aux_par[counter]
                self.TotalStorageSolution[i, j] = aux_tot[counter]
                self.InputsTreeSolution[i, j] = aux_in[counter]
                self.OutputsTreeSolution[i, j] = aux_out[counter]
                counter += 1

    def set_parameters_cpp_energy_models(self):
        """ This class method set all parameters in the c++ implementation """
        # Information nodes
        aux_intake = []
        for vectors in range(self.NumberTrees):
            for nodes in range(self.TreeNodes):
                aux_intake.append(self.IntakeTree[nodes, vectors])
        aux_output = []
        for vectors in range(self.NumberTrees):
            for nodes in range(self.TreeNodes):
                aux_output.append(self.OutputTree[nodes, vectors])
        self.energy_model.load_energy_tree_information_cpp(self.TreeNodes, \
            self.NumberTrees, self.LLEB, self.LLEA, aux_intake, \
            aux_output, self.WeightNodes)

    # Data inputs of Energy model

    def SetTreeNodes(self, number_nodes=None):
        assert number_nodes is not None, \
            "No value for the number of nodes per tree" 
        self.TreeNodes = number_nodes + 1
    
    def SetNumberTrees(self, number_trees=None):
        assert number_trees is not None, \
            "No value for the number of trees to be optimised" 
        self.NumberTrees = number_trees

    def SetIntakeTree(self, intake_nodes=None):
        assert intake_nodes is not None, \
            "No value for the intake of water/energy for all nodes"
        self.IntakeTree = intake_nodes

    def SetOutputTree(self, output_nodes=None):
        assert output_nodes is not None, \
            "No value for the output of water/energy for all nodes"
        self.OutputTree = output_nodes

    def SetWeightNodes(self, weight_nodes=None):
        assert weight_nodes is not None, \
            "No value for the weights of all nodes"
        self.WeightNodes = weight_nodes

    def SetNumberNodesUnc(self, number_nodes_unc=None):
        assert number_nodes_unc is not None, \
            "No value for the number of nodes for uncertainty analysis"
        self.NumberNodesUnc = number_nodes_unc
    
    def SetLLNodesUnc(self, LL_nodes_unc=None):
        assert LL_nodes_unc is not None, \
            "No value for the number of nodes for uncertainty analysis"
        self.LLNodesUnc = LL_nodes_unc

    # TODO: Redefine link list

    def SetLLEB(self, LLEB_connections=None):
        assert LLEB_connections is None, \
            "No value for the link list of the energy balance constraint"
        self.LLEB = LLEB_connections

    def SetLLEA(self, LLEA_connections=None):
        assert LLEA_connections is None, \
            "No value for the link list of the energy aggregation constraint"
        self.LLEB = LLEA_connections



    # Data outputs of Energy model

    def _recursive_get_tree(self, number_node=None, name=None, position_tree=None, typ=None, level=None, current_position_tree=None):
        ''' This function search and retrieves the values of variables and parameters 
        
            Parameters
            ----------
            Mandatory:\\
            name            :   Name of variable or parameter to be retrieved\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree.\\
            name_node       :   Name of level, e.g. summer, weekday.\\                        
            typ             :   This refers to the type of element to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted
            number_node     :   Number of node in the tree
            level           :   level in the tree to be analysed
        '''
        if position_tree.get(str(level), None):
            current_position_tree[str(level)] = position_tree.get(str(level), None)
        if current_position_tree == position_tree and typ=="parameters":
            for parameter in self.tree[number_node]["obj"].parameters:
                if parameter.name == name:
                    return parameter.value
        elif current_position_tree == position_tree and typ=="variables":
            for variable in self.tree[number_node]["obj"].variables:
                if variable.name == name:
                    return variable.value
        else:
            level = level + 1
            for node in self.tree.successors(number_node):
                if position_tree[str(level)] == node.name_node:
                    return self._recursive_get_tree(node.node, name, position_tree, typ, level, current_position_tree)

    def get_value_tree(self, name=None, position_tree=None, typ=None):
        ''' This function retrieves the values of variables and parameters 
        
            Parameters
            ----------
            Mandatory:\\
            name            :   Name of variable or parameter to be retrieved\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree.\\
            typ             :   This refers to the type of element to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted
        '''
        for node in self.tree.nodes(data=True):
            if node.level == 0:
                return self._recursive_get_tree(node.node, name, position_tree, typ, 0, {})        
        return None
    
    def _recursive_set_tree(self, number_node=None, name=None, position_tree=None, typ=None, level=None, current_position_tree=None, value=None):
        ''' This function search and retrieves the values of variables and parameters 
        
            Parameters
            ----------
            Mandatory:\\
            name            :   Name of variable or parameter to be retrieved\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree.\\
            name_node       :   Name of level, e.g. summer, weekday.\\                        
            typ             :   This refers to the type of element to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted
            number_node     :   Number of node in the tree
            level           :   level in the tree to be analysed
        '''
        if position_tree.get(str(level), None):
            current_position_tree[str(level)] = position_tree.get(str(level), None)
        if current_position_tree == position_tree and typ=="parameters":
            for parameter in self.tree[number_node]["obj"].parameters:
                if parameter.name == name:
                    parameter.value = value
                    return
        elif current_position_tree == position_tree and typ=="variables":
            for variable in self.tree[number_node]["obj"].variables:
                if variable.name == name:
                    variable.value = value
                    return
        else:
            level = level + 1
            for node in self.tree.successors(number_node):
                if position_tree[str(level)] == node.name_node:
                    self._recursive_set_tree(node.node, name, position_tree, typ, level, current_position_tree)
                    return

    def set_value_tree(self, name=None, position_tree=None, typ=None, value=None):
        ''' This function retrieves the values of variables and parameters 
        
            Parameters
            ----------
            Mandatory:\\
            name            :   Name of variable or parameter to be set\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree.\\
            typ             :   This refers to the type of element to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted \\
            value           :   value to be set in the energy tree
        '''
        for node in self.tree.nodes(data=True):
            if node.level == 0:
                return self._recursive_set_tree(node.node, name, position_tree, typ, 0, {}, value)        
        return None    



    def GetPartialStorage(self):
        if self.solver_problem == "GLPK":
            PartialStorageSolution = \
                np.empty((self.NumberTrees, self.TreeNodes))
            for i in range(self.NumberTrees):
                for j in range(self.TreeNodes):
                    PartialStorageSolution[i, j] = \
                        self.solver.get_col_prim(str(\
                        self.Partialstorage[i][0]), j)
            return PartialStorageSolution
        elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
            return self.PartialStorageSolution
    
    def GetTotalStorage(self):
        if self.solver_problem == "GLPK":
            TotalStorageSolution = \
                np.empty((self.NumberTrees, self.TreeNodes))
            for i in range(self.NumberTrees):
                for j in range(self.TreeNodes):
                    TotalStorageSolution[i, j] = \
                        self.solver.get_col_prim(str(\
                        self.Totalstorage[i][0]), j)
            return TotalStorageSolution
        elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
            return self.TotalStorageSolution

    def GetInputsTree(self):
        if self.solver_problem == "GLPK":
            InputsTreeSolution = \
                np.empty((self.NumberTrees, self.TreeNodes))
            for i in range(self.NumberTrees):
                for j in range(self.TreeNodes):
                    InputsTreeSolution[i, j] = \
                        self.solver.get_col_prim(str(\
                        self.InputsTree[i][0]), j)
            return InputsTreeSolution
        elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
            return self.InputsTreeSolution

    def GetOutputsTree(self):
        if self.solver_problem == "GLPK":
            OutputsTreeSolution = \
                np.empty((self.NumberTrees, self.TreeNodes))
            for i in range(self.NumberTrees):
                for j in range(self.TreeNodes):
                    OutputsTreeSolution[i, j] = \
                        self.solver.get_col_prim(str(\
                        self.OutputsTree[i][0]), j)
            return OutputsTreeSolution
        elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
            return self.OutputsTreeSolution
    
    def GetEnergybalanceDual(self):
        EnergybalanceDualSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        for i in range(self.NumberTrees):
            EnergybalanceDualSolution[i, 0] = 0
            for j in range(1, self.TreeNodes):
                EnergybalanceDualSolution[i, j] = \
                    self.solver.get_row_dual(str(\
                    self.treebalance[i][0]), j - 1)
        return EnergybalanceDualSolution

class Networkmodel():
    """ This class builds and solve the network model(NM).

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

        self.network_parameters = []
        self.network_variables = []

        # Storing data input - Parameters
        assert obj is not None  # If the object is empty then raise an error
        self.LongTemporalConnections = obj.connections['set'] # Temporal
                            # interconnection of different instances of
                            # network problems (last nodes of the temporal
                            # tree)
        self.ShortTemporalConnections = obj.settings['NoTime'] # Number
                            # of sub-periods in a 24h period

        self.NumberConvGen = len(obj.Gen.Conv) # Number of conventional
                            # generators
        self.NumberRESGen = len(obj.Gen.RES) # Number of RES generators
        self.NumberHydroGen = len(obj.Gen.Hydro) # Number of Hydro generators
        self.NumberPumps = obj.pumps['Number'] # Number of Pumps
        self.NumberStorageDevices = obj.Storage['Number'] # Number of storage
                            # devices in the system
        self.NumberDemScenarios = obj.scenarios['NoDem'] # Number of
                            # demand scenarios
        self.NumberNodesPS = obj.ENetwork.get_NoBus() # Number of nodes
                            # in the power system
        self.NumberContingencies = len(obj.settings['Security']) # Number
                            # of devices analysed in the N-1 contingency
                            # analysis
        self.NumberLinesPS = obj.ENetwork.get_NoBra() # Number of transmission
                            # lines in the power system

        self.BaseUnitPower = obj.ENetwork.get_Base() # Base power for power
                            # system
        
        self.PercentageLosses = obj.settings['Loss'] # Percentage of losses
                            # that is considered in the formulation

        self.LossesFlag = obj.settings['Losses'] # Flag to indicate if the 
                            # losses are included in the mathematical
                            # formulation
        
        self.FlagProblem = obj.settings['Flag'] # Flag that indicates
                            # if the problem to solve is the economic
                            # dispatch or the optimal power flow
                            # False = Economic Dispatch
                            # True = Optimal Power Flow
        
        self.FlagFeasibility = obj.settings['Feasibility'] # Flag that 
                            # indicates if the problem should include
                            # load curtailment variables

        if self.LossesFlag:
            self.NumberPiecesTLLosses = obj.Number_LossCon # Number 
                            # of pieces
                            # in the piecewise linearisation of the
                            # transmission line losses
        
        # TODO: Generalise inputs as a list of values
        if self.NumberConvGen > 0:
            self.ActiveConv = np.ones(self.NumberConvGen, dtype=bool) # Array of boolean parameter 
                                                                        # indicating if power plan is active or not
            for i in range(self.NumberConvGen):
                self.ActiveConv[i] = obj.Gen.Conv[i].data['GEN']
            self.PWConvGen = np.empty((self.NumberConvGen), dtype=np.int_) # Number of pieces of
                                # the piecewise linearisation of the conventional 
                                # generation cost
            for i in range(self.NumberConvGen):
                self.PWConvGen[i] = obj.Gen.Conv[i].get_NoPieces()
            self.MinConvGen = np.empty(self.NumberConvGen) # Minimum generation
                                # limit for conventional generators
            for i in range(self.NumberConvGen):
                self.MinConvGen[i] = obj.Gen.Conv[i].get_Min()
            self.MaxConvGen = np.empty(self.NumberConvGen) # Maximum generation
                                # limit for conventional generators
            for i in range(self.NumberConvGen):
                self.MaxConvGen[i] = obj.Gen.Conv[i].get_Max()
            #TODO: Generalise for N number of pieces per generator
            self.ACoeffPWConvGen = np.empty((self.NumberConvGen,\
                self.PWConvGen[0])) # Coefficient A of the piece Ax + b for
                                    # conventional generation
            for i in range(self.NumberConvGen):
                for j in range(self.PWConvGen[i]):
                    self.ACoeffPWConvGen[i, j] = \
                        obj.Gen.Conv[i].cost['LCost'][j][0]
            self.BCoeffPWConvGen = np.empty((self.NumberConvGen,\
                self.PWConvGen[0])) # Coefficient b of the piece Ax + b for
                                    # conventional generation
            for i in range(self.NumberConvGen):
                for j in range(self.PWConvGen[i]):
                    self.BCoeffPWConvGen[i, j] = \
                        obj.Gen.Conv[i].cost['LCost'][j][1]
            self.RampConvGen = np.empty(self.NumberConvGen) # On/Off ramps
                                    # for conventional generators
            for i in range(self.NumberConvGen):
                self.RampConvGen[i] = obj.Gen.Conv[i].data['Ramp']
            self.OriginalNumberConvGen = \
                np.empty((self.NumberConvGen)) # Original numeration of 
                                # the conventional generators in the 
                                # power system
            for i in range(self.NumberConvGen):
                self.OriginalNumberConvGen[i] = \
                    obj.Gen.Conv[i].get_Bus()

        if self.NumberRESGen > 0:
            self.PWRESGen = np.empty((self.NumberRESGen), dtype=np.int_) # Number of pieces of
                                # the piecewise linearisation of the RES 
                                # generation cost
            for i in range(self.NumberRESGen):
                self.PWRESGen[i] = obj.Gen.RES[i].get_NoPieces()
            self.MinRESGen = np.empty(self.NumberRESGen) # Minimum generation
                                # limit for RES generators
            for i in range(self.NumberRESGen):
                self.MinRESGen[i] = obj.Gen.RES[i].get_Min()
            self.MaxRESGen = np.empty(self.NumberRESGen) # Minimum generation
                                # limit for RES generators
            for i in range(self.NumberRESGen):
                self.MaxRESGen[i] = obj.RES['Max'][i]
            self.RESScenarios = np.empty((len(self.LongTemporalConnections), \
            self.ShortTemporalConnections, self.NumberConvGen)) 
                                # Scenarios of generation for RES for different
                                # time periods
            for i in self.LongTemporalConnections:
                for j in range(self.ShortTemporalConnections):
                    for k in range(self.NumberRESGen):
                        self.RESScenarios[i, j, k] = obj.scenarios['RES']\
                                    [obj.resScenario[k][i]+j]
            #TODO: Generalise for N number of pieces per generator
            self.ACoeffPWRESGen = np.empty((self.NumberRESGen,\
                self.PWRESGen[0]))  # Coefficient A of the piece Ax + b for
                                    # RES generation
            for i in range(self.NumberRESGen):
                for j in range(self.PWRESGen[i]):
                    self.ACoeffPWRESGen[i, j] = \
                        obj.Gen.RES[i].cost['LCost'][j][0]
            self.BCoeffPWRESGen = np.empty((self.NumberRESGen,\
                self.PWRESGen[0]))  # Coefficient b of the piece Ax + b for
                                    # RES generation
            for i in range(self.NumberRESGen):
                for j in range(self.PWRESGen[i]):
                    self.BCoeffPWRESGen[i, j] = \
                        obj.Gen.RES[i].cost['LCost'][j][1]
            self.OriginalNumberRESGen = \
                np.empty((self.NumberRESGen)) # Original numeration of 
                                # the RES generators in the 
                                # power system
            for i in range(self.NumberRESGen):
                self.OriginalNumberRESGen[i] = \
                    obj.Gen.RES[i].get_Bus()
        
        if self.NumberHydroGen > 0:
            self.PWHydroGen = np.empty((self.NumberHydroGen), dtype=np.int_) # Number of pieces of
                                # the piecewise linearisation of the Hydro 
                                # generation cost
            for i in range(self.NumberHydroGen):
                self.PWHydroGen[i] = obj.Gen.Hydro[i].get_NoPieces()
            self.MinHydroGen = np.empty(self.NumberHydroGen) # Minimum generation
                                # limit for hydro generators
            for i in range(self.NumberHydroGen):
                self.MinHydroGen[i] = obj.Gen.Hydro[i].get_Min()
            self.MaxHydroGen = np.empty(self.NumberHydroGen) # Minimum generation
                                # limit for hydro generators
            for i in range(self.NumberHydroGen):
                self.MaxHydroGen[i] = obj.Gen.Hydro[i].get_Max()

            #TODO: Generalise for N number of pieces per generator
            self.ACoeffPWHydroGen = np.empty((self.NumberHydroGen,\
                self.PWHydroGen[0]))  # Coefficient A of the piece Ax + b for
                                    # RES generation
            for i in range(self.NumberHydroGen):
                for j in range(self.PWHydroGen[i]):
                    self.ACoeffPWHydroGen[i, j] = \
                        obj.Gen.Hydro[i].cost['LCost'][j][0]
            self.BCoeffPWHydroGen = np.empty((self.NumberHydroGen,\
                self.PWHydroGen[0]))  # Coefficient b of the piece Ax + b for
                                    # RES generation
            for i in range(self.NumberHydroGen):
                for j in range(self.PWHydroGen[i]):
                    self.BCoeffPWHydroGen[i, j] = \
                        obj.Gen.Hydro[i].cost['LCost'][j][1]
            self.RampHydroGen = np.empty(self.NumberHydroGen) # On/Off ramps
                                    # for hydro generators
            for i in range(self.NumberHydroGen):
                self.RampHydroGen[i] = obj.Gen.Hydro[i].data['Ramp']
            self.OriginalNumberHydroGen = \
                np.empty((self.NumberHydroGen)) # Original numeration of 
                                # the Hydro generators in the 
                                # power system
            for i in range(self.NumberHydroGen):
                self.OriginalNumberHydroGen[i] = \
                    obj.Gen.Hydro[i].get_Bus()

        if self.NumberPumps > 0:
            self.MaxPowerPumps = np.empty(self.NumberPumps) # Maximum 
                            # power capacity of pumps
            for i in range(self.NumberPumps):
                self.MaxPowerPumps[i] = obj.pumps['Max'][i]/self.BaseUnitPower
            self.CostOperPumps = obj.pumps['Value'] # Operational 
                            # cost of pumps

        if self.NumberStorageDevices > 0:
            self.EffStorage = obj.Storage['Efficiency'] # Efficiency 
                            # of storage elements
 

        self.TotalHoursPerPeriod = obj.scenarios['Weights'] # Number
                            # of hours per sub-period in a 24-hour period
        
        if self.FlagProblem:
            self.ActiveBranches = np.empty(\
                    ((self.NumberContingencies + 1),\
                    self.NumberLinesPS))    # Flag to indicate if the 
                                # transmission line or transformer is active 
                                # on each contingency
            for i in range(self.NumberContingencies + 1):
                for j in range(self.NumberLinesPS):
                    self.ActiveBranches[i, j] = \
                        obj.ENetwork.Branch[j].is_active(i) * \
                        obj.ENetwork.Branch[j].data['BR_STATUS']
            self.PowerRateLimitTL = np.empty((self.NumberLinesPS)) # Thermal
                                # limit of power transmission lines and 
                                # transformers
            for i in range(self.NumberLinesPS):
                self.PowerRateLimitTL[i] = \
                    obj.ENetwork.Branch[i].get_Rate()
            self.OriginalNumberBranchFrom = \
                np.empty((self.NumberLinesPS)) # Original numeration of 
                                    # the transmission lines and transformers
                                    # in the power system in the from end
            for i in range(self.NumberLinesPS):
                self.OriginalNumberBranchFrom[i] = \
                    obj.ENetwork.Branch[i].get_BusF()
            self.OriginalNumberBranchTo = \
                np.empty((self.NumberLinesPS)) # Original numeration of 
                                    # the transmission lines and transformers
                                    # in the power system in the to end
            for i in range(self.NumberLinesPS):
                self.OriginalNumberBranchTo[i] = \
                    obj.ENetwork.Branch[i].get_BusT()

            self.PosNumberBranchFrom = \
                np.empty((self.NumberLinesPS)) # Position of the from end of
                                    # the transmission lines and transformers
                                    # in the vector that stores the node data.
                                    # The position start from zero in the node
                                    # data
            for i in range(self.NumberLinesPS):
                self.PosNumberBranchFrom[i] = \
                    obj.ENetwork.Branch[i].get_PosF()
            self.PosNumberBranchTo = \
                np.empty((self.NumberLinesPS)) # Position of the to end of
                                    # the transmission lines and transformers
                                    # in the vector that stores the node data.
                                    # The position start from zero in the node
                                    # data
            for i in range(self.NumberLinesPS):
                self.PosNumberBranchTo[i] = \
                    obj.ENetwork.Branch[i].get_PosT()
            self.ReactanceBranch = \
                np.empty((self.NumberLinesPS)) # Reactance of the transmission 
                                    # lines and transformers
            for i in range(self.NumberLinesPS):
                # if obj.ENetwork.Branch[i].get_X() != 0:
                self.ReactanceBranch[i] = \
                    obj.ENetwork.Branch[i].get_X()
                # else:
                #     self.ReactanceBranch[i] = 0.0001
            self.ResistanceBranch = \
                np.empty((self.NumberLinesPS)) # Resistance of the transmission 
                                    # lines and transformers
            for i in range(self.NumberLinesPS):
                self.ResistanceBranch[i] = \
                    obj.ENetwork.Branch[i].get_R()
            
            self.NontechnicalLosses = \
                np.empty((self.NumberLinesPS)) # Non-technical losses of the 
                                    # transmission lines and transformers
            for i in range(self.NumberLinesPS):
                self.NontechnicalLosses[i] = \
                    obj.ENetwork.Branch[i].getLoss()/self.BaseUnitPower

            if self.LossesFlag:
                self.ACoeffPWBranchLosses = \
                    np.empty((self.NumberPiecesTLLosses)) # Coefficient A of the 
                                        # piece Ax + b for the piecewise 
                                        # linearisation of the nonlinear branch
                                        # Losses
                for i in range(self.NumberPiecesTLLosses):
                    self.ACoeffPWBranchLosses[i] = \
                        obj.ENetwork.loss['A'][i]
                self.BCoeffPWBranchLosses = \
                    np.empty((self.NumberPiecesTLLosses)) # Coefficient A of the 
                                        # piece Ax + b for the piecewise 
                                        # linearisation of the nonlinear branch
                                        # Losses
                for i in range(self.NumberPiecesTLLosses):
                    self.BCoeffPWBranchLosses[i] = \
                        obj.ENetwork.loss['B'][i]
        
        if self.NumberDemScenarios == 0:
            self.MultScenariosDemand = np.empty(\
                (len(self.LongTemporalConnections),\
                self.NumberNodesPS))  # Multiplier to adjust the demand
                            # on each node for each temporal representative
                            # day
            for i in self.LongTemporalConnections:
                for j in range(self.NumberNodesPS):
                    self.MultScenariosDemand[i, j] = \
                        obj.scenarios['Demand'][obj.busScenario[j][i]]
        else:
            self.MultScenariosDemand = np.empty(\
                (len(self.LongTemporalConnections),\
                self.ShortTemporalConnections,\
                self.NumberNodesPS))  # Multiplier to adjust the demand
                            # on each node for each temporal representative
                            # day and for each sub-period in the 24h period
            for i in self.LongTemporalConnections:
                for j in range(self.ShortTemporalConnections):
                    for k in range(self.NumberNodesPS):
                        self.MultScenariosDemand[i, j, k] = \
                            obj.scenarios['Demand']\
                                [j+obj.busScenario[k][i]]
        
        self.PowerDemandNode = np.empty((self.NumberNodesPS)) # Active
                            # Power demand at each node
        for i in range(self.NumberNodesPS):
            self.PowerDemandNode[i] = \
                obj.busData[i]
        self.TypeNode = np.empty((self.NumberNodesPS)) # Type
                            # of node
        for i in range(self.NumberNodesPS):
            self.TypeNode[i] = \
                obj.ENetwork.Bus[i].get_Type()
        self.OriginalNumberNodes = \
                np.empty((self.NumberNodesPS)) # Original numeration of 
                                # the nodes in the power system
        for i in range(self.NumberNodesPS):
            self.OriginalNumberNodes[i] = \
                obj.ENetwork.Bus[i].get_Number()
        
        for i in range(self.NumberConvGen):
            for j in range(self.NumberNodesPS):
                if self.OriginalNumberConvGen[i] == self.OriginalNumberNodes[j]:
                    if self.TypeNode[j] == 4:
                        self.ActiveConv[i] = False
                    break
        
        self.OFaux = np.ones(len(self.LongTemporalConnections), dtype=float)

    def _create_nodes_graph(self):
        nodes_graph = []
        exist = False
        counter = 0
        # Creating list of nodes  and adding parameters
        for parameter in self.network_parameters:
            if nodes_graph:
                for node_g in nodes_graph:
                    if node_g.ID == parameter.ID:
                        exist = True
                        node_g.parameters.append(parameter)
                        break
            if not exist:
                node = nodes_info_network
                node.node = counter
                node.type = parameter.type
                node.subtype = parameter.subtype
                node.ID = parameter.ID
                node.parameters = [parameter]
                node.bus = parameter.bus
                node.ends = parameter.ends
                counter += 1
                nodes_graph.append(node)
            exist = False
        
        # Adding variables to nodes
        # for variable in self.network_variables:
        #     for node_g in nodes_graph:
        #         if node_g.ID == variable.ID:
        #             if node_g.variables:
        #                 node_g.variables.append(variable)
        #             else:
        #                 node_g.variables = [variable]
        #             break
        
        # Adding nodes to graph
        for node_g in nodes_graph:
            self.network.add_node(node_g.node, obj=node_g)

    def _create_edges_graph(self):
        # Creating branches of graph
        branches_graph = []
        for node_g in self.network.nodes(data=True):
            if node_g[1]['obj'].type == "generator":
                for aux in self.network.nodes(data=True):
                    if aux[1]['obj'].type == "bus" and node_g[1].bus == aux[1]['obj'].bus:
                        branches_graph.append([aux[1]['obj'].node, node_g[1].node])
                        break
            elif node_g[1]['obj'].type == "branch":
                flag = [False, False]
                for aux in self.network.nodes(data=True):
                    if aux[1]['obj'].type == "bus" and node_g[1]['obj'].ends[0] == aux[1]['obj'].bus:
                        branches_graph.append([aux[1]['obj'].node, node_g[1]['obj'].node])
                        flag[0] = True
                    elif aux[1]['obj'].type == "bus" and node_g[1]['obj'].ends[1] == aux[1]['obj'].bus:
                        branches_graph.append([aux[1]['obj'].node, node_g[1]['obj'].node])
                        flag[1] = True
                    if flag[0] and flag[1]:
                        break
        for branches in  branches_graph:
            self.network.add_edge(branches[0], branches[1])

    def _create_graph(self):
        self.network = nx.MultiGraph()
        self._create_nodes_graph()
        self._create_edges_graph()

    def optimisationNM(self, solver_name=None):
        """ This class method solve the optimisation problem """
        if solver_name == "GLPK":
            # Creation of model instance
            self.solver = GLPKSolver(message_level='all', \
                simplex_method='dualprimal')      
            # Definition of minimisation problem
            self.solver.set_dir('min')
            self.solver_problem = "GLPK"
            # Definition of the mathematical formulation
            if self.FlagProblem:
                self.OptimalPowerFlowModel()
            else:
                self.EconomicDispatchModel()
            ret = self.solver.simplex()
            assert ret == 0, "GLPK could not solve the problem"
        elif solver_name == "CLP":
            self.solver_problem = "CLP"
            self.OptimalPowerFlowModelCPP()
        elif solver_name == "CLP-I":
            self.solver_problem = "CLP-I"
            self.OptimalPowerFlowModelCPP()
        elif solver_name == "CLP-IR":
            self.solver_problem = "CLP-IR"
            self.OptimalPowerFlowModelCPP()
        else:
            print("incorrect solver has been selected")


    ############################################
    ###   COMMON VARIABLES AND CONSTRAINTS   ###
    ###   FOR DIFFERENT MODELS               ###
    ############################################

    # Number of variables and constraints
 
    # Variables
    def PosvariablesCommon(self):
        """ This class method creates the vector that stores the positions of 
        variables that are common for various problems """

        if self.NumberConvGen > 0:
            self.thermalgenerators = np.empty(\
                (len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of thermal generators' variables in matrix A (rows)
                # for each period and each tree node
        if self.NumberRESGen > 0:
            self.RESgenerators = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of RES generators' variables in matrix A (rows)
                # for each period and each tree node
        if self.NumberHydroGen > 0:        
            self.Hydrogenerators = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of Hydroelectric generators' variables in matrix A (rows)
                # for each period and each tree node
        # TODO: Change this with a flag for batteries
        if self.NumberStorageDevices > 0:
            self.ESS = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of Energy Storage Systems' variables in matrix A (rows)
                # for each period and each tree node
        # TODO: Change this with a flag for pumps
        if self.NumberPumps > 0:
            self.pumpsvar = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of pumps' variables in matrix A (rows)
                # for each period and each tree node
        if self.NumberConvGen > 0:
            self.thermalCG = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of thermal generation cost variables in matrix A (rows)
                # for each period and each tree node
        if self.NumberRESGen > 0:
            self.RESCG = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of RES generation cost variables in matrix A (rows)
                # for each period and each tree node
        if self.NumberHydroGen > 0:
            self.HydroCG = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
                    ('nupos', 'i4')]) # Start position 
                # of Hydroelectric generation cost variables in matrix A (rows)
                # for each period and each tree node

    def variablesCommon(self):
        """ This class method defines the variables and their limits that 
        are common for various problems """
        self.PosvariablesED()
        
        # Reserving space in glpk for ED variables
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                # Generation variables
                if self.NumberConvGen > 0:
                    self.thermalgenerators[i, j] = (\
                        'ThermalGen'+str(i)+','+str(j),\
                        self.solver.add_cols('ThermalGen'+str(i)+','+str(j),\
                        self.NumberConvGen))
                if self.NumberRESGen > 0:
                    self.RESgenerators[i, j] = (\
                        'RESGen'+str(i)+','+str(j),\
                        self.solver.add_cols('RESGen'+str(i)+','+str(j),\
                        self.NumberRESGen))
                if self.NumberHydroGen > 0:
                    self.Hydrogenerators[i, j] = (\
                        'HydroGen'+str(i)+','+str(j),\
                        self.solver.add_cols('HydroGen'+str(i)+','+str(j),\
                        self.NumberHydroGen))
                # Generation cost variables
                if self.NumberConvGen > 0:
                    self.thermalCG[i, j] = ('ThermalCG'+str(i)+','+str(j),\
                        self.solver.add_cols('ThermalCG'+str(i)+','+str(j),\
                        self.NumberConvGen))
                if self.NumberRESGen > 0:
                    self.RESCG[i, j] = ('RESCG'+str(i)+','+str(j),\
                        self.solver.add_cols('RESCG'+str(i)+','+str(j),\
                        self.NumberRESGen))
                if self.NumberHydroGen > 0:
                    self.HydroCG[i, j] = ('HydroCG'+str(i)+','+str(j),\
                        self.solver.add_cols('HydroCG'+str(i)+','+str(j),\
                        self.NumberHydroGen))
                # TODO: Change this with a flag for batteries
                if self.NumberStorageDevices > 0:
                    self.ESS[i, j] = ('ESS'+str(i)+','+str(j),\
                        self.solver.add_cols('ESS'+str(i)+','+str(j),\
                        self.NumberStorageDevices))
                # TODO: Change this with a flag for pumps
                if self.NumberPumps > 0:
                    self.pumpsvar[i, j] = ('Pumps'+str(i)+','+str(j),\
                        self.solver.add_cols('Pumps'+str(i)+','+str(j),\
                        self.NumberPumps))


        # Defining the limits of the variables
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                # Limits for the thermal generators
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        if self.ActiveConv[k] and self.MinConvGen[k] != \
                            self.MaxConvGen[k]:
                            self.solver.set_col_bnds(\
                                str(self.thermalgenerators[i, j][0]), k,\
                                'bounded', self.MinConvGen[k],\
                                self.MaxConvGen[k])
                        elif self.ActiveConv[k] and self.MinConvGen[k] == \
                            self.MaxConvGen[k]:
                            self.solver.set_col_bnds(\
                                str(self.thermalgenerators[i, j][0]), k,\
                                'fixed', self.MinConvGen[k],\
                                self.MaxConvGen[k])
                        else:
                            self.solver.set_col_bnds(\
                                str(self.thermalgenerators[i, j][0]), k,\
                                'fixed', 0, 0)
                # Limits for the RES generators
                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        if self.MinRESGen[k] != (\
                            self.RESScenarios[i, j, k] * self.MaxRESGen[k]):
                            self.solver.set_col_bnds(\
                                str(self.RESgenerators[i, j][0]), k,\
                                'bounded', self.MinRESGen[k],\
                                self.RESScenarios[i, j, k] * \
                                    self.MaxRESGen[k])
                        else:
                            self.solver.set_col_bnds(\
                                str(self.RESgenerators[i, j][0]), k,\
                                'fixed', self.MinRESGen[k],\
                                self.MinRESGen[k])

                # Limits for the Hydroelectric generators
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        if self.MinHydroGen[k] != self.MaxHydroGen[k]:
                            self.solver.set_col_bnds(\
                                str(self.Hydrogenerators[i, j][0]), k,\
                                'bounded', self.MinHydroGen[k],\
                                self.MaxHydroGen[k])
                        else:
                            self.solver.set_col_bnds(\
                                str(self.Hydrogenerators[i, j][0]), k,\
                                'fixed', self.MinHydroGen[k],\
                                self.MaxHydroGen[k])
                # TODO: Modify information of storage, e.g. m.sNSto
                # if self.NumberStorageDevices > 0:
                if self.NumberPumps > 0:
                    for k in range(self.NumberPumps):
                        if self.MaxPowerPumps[k] != 0:
                            self.solver.set_col_bnds(\
                                str(self.pumpsvar[i, j][0]), k,\
                                'bounded', 0, self.MaxPowerPumps[k])
                        else:
                            self.solver.set_col_bnds(\
                                str(self.pumpsvar[i, j][0]), k,\
                                'fixed', 0, 0)

    # Constraints

    def piecewiselinearisationcost(self):
        """ This class method writes the piecewise linearisarion of
        the generation cost in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        if self.NumberConvGen > 0:
            self.thermalpiecewisecost = \
                np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, self.NumberConvGen),\
                    dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                    # position of piecewise linearisation constraints 
                    # (rows) for each tree node, for each period and 
                    # for each thermal generator
        if self.NumberRESGen > 0:
            self.RESpiecewisecost = \
                np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, self.NumberRESGen),\
                    dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                    # position of piecewise linearisation constraints 
                    # (rows) for each tree node, for each period and 
                    # for each RES generator
        if self.NumberHydroGen > 0:
            self.Hydropiecewisecost = \
                np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, self.NumberHydroGen),\
                    dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                    # position of piecewise linearisation constraints 
                    # (rows) for each tree node, for each period and 
                    # for each Hydro generator

        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        self.thermalpiecewisecost[i, j, k] =\
                            ('ThermalPWC'+str(i)+','+str(j)+','+str(k),\
                            self.solver.add_rows(\
                                'ThermalPWC'+str(i)+','+str(j)+','+str(k), \
                                self.PWConvGen[k]))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each thermal generator
                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        self.RESpiecewisecost[i, j, k] =\
                            ('RESPWC'+str(i)+','+str(j)+','+str(k),\
                            self.solver.add_rows(\
                                'RESPWC'+str(i)+','+str(j)+','+str(k), \
                                self.PWRESGen[k]))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each RES generator
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        self.Hydropiecewisecost[i, j, k] =\
                            ('HydroPWC'+str(i)+','+str(j)+','+str(k),\
                            self.solver.add_rows(\
                                'HydroPWC'+str(i)+','+str(j)+','+str(k), \
                                self.PWHydroGen[k]))
                            # Number of columns (constraints) in matrix A 
                            # for the piecewise linearisation constraints 
                            # of the generation cost for each period, 
                            # each tree node and each Hydro generator

        # Generating the matrix A for the piecewise linearisation constraints of
        # the generation cost
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        for l in range(self.PWConvGen[k]):
                        # Storing the generation cost variables
                            self.ia.append(self.thermalpiecewisecost[i, j, k][1] + l)
                            self.ja.append(self.thermalCG[i, j][1] + k)
                            self.ar.append(1.0)
                            self.ne += 1
                        # Storing the generation variables
                            self.ia.append(self.thermalpiecewisecost[i, j, k][1] + l)
                            self.ja.append(self.thermalgenerators[i, j][1] + k)
                            self.ar.append(-self.TotalHoursPerPeriod[j] * \
                                self.ACoeffPWConvGen[k, l])
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.thermalpiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.TotalHoursPerPeriod[j] * \
                                self.BCoeffPWConvGen[k, l], 0)

                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        for l in range(self.PWRESGen[k]):
                        # Storing the generation cost variables
                            self.ia.append(self.RESpiecewisecost[i, j, k][1] + l)
                            self.ja.append(self.RESCG[i, j][1] + k)
                            self.ar.append(1.0)
                            self.ne += 1
                        # Storing the generation variables
                            self.ia.append(self.RESpiecewisecost[i, j, k][1] + l)
                            self.ja.append(self.RESgenerators[i, j][1] + k)
                            self.ar.append(-self.TotalHoursPerPeriod[j] * \
                                self.ACoeffPWRESGen[k, l])
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.RESpiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.TotalHoursPerPeriod[j] * \
                                self.BCoeffPWRESGen[k, l], 0)

                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        for l in range(self.PWHydroGen[k]):
                        # Storing the generation cost variables
                            self.ia.append(self.Hydropiecewisecost[i, j, k][1] + l)
                            self.ja.append(self.HydroCG[i, j][1] + k)
                            self.ar.append(1.0)
                            self.ne += 1
                        # Storing the generation variables
                            self.ia.append(self.Hydropiecewisecost[i, j, k][1] + l)
                            self.ja.append(self.Hydrogenerators[i, j][1] + k)
                            self.ar.append(-self.TotalHoursPerPeriod[j] * \
                                self.ACoeffPWHydroGen[k, l])
                            self.ne += 1
                        # Defining the resources (b) for the constraints
                            self.solver.set_row_bnds(\
                                str(self.Hydropiecewisecost[i, j, k][0]),\
                                l, 'lower',\
                                self.TotalHoursPerPeriod[j] * \
                                self.BCoeffPWHydroGen[k, l], 0)

    def generationrampsconstraints(self):
        """ This class method writes the constraints for the generation ramps
        in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        if self.NumberConvGen > 0:
            self.thermalgenerationramps = \
                np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections - 1),\
                    dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                    # position of thermal generation ramps constraints 
                    # (rows) for each tree node, for each period and for 
                    # each thermal generator
        if self.NumberHydroGen > 0:
            self.Hydrogenerationramps = \
                np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections - 1),\
                    dtype=[('napos', 'U80'), ('nupos', 'i4')]) # Start 
                    # position of Hydroelectrical generation ramps constraints
                    # (rows) for each tree node, for each period and for 
                    # each hydroelectrical generator

        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                if j > 0:
                    if self.NumberConvGen > 0:
                        self.thermalgenerationramps[i, j - 1] = \
                            ('ThermalGR'+str(i)+','+str(j),\
                            self.solver.add_rows('ThermalGR'+str(i)+','+str(j),\
                                self.NumberConvGen))  # Number of 
                                # columns (constraints) in matrix A for the 
                                # generation ramps constraints for each 
                                # period, for each tree node and for each 
                                # thermal generator
                    if self.NumberHydroGen > 0:
                        self.Hydrogenerationramps[i, j - 1] = \
                            ('HydroGR'+str(i)+','+str(j),\
                            self.solver.add_rows('HydroGR'+str(i)+','+str(j),\
                                self.NumberHydroGen))  # Number of 
                                # columns (constraints) in matrix A for the 
                                # generation ramps constraints for each 
                                # period, for each tree node and for each 
                                # thermal generator

        # Generating the matrix A for the generation ramps constraints
        for i in self.LongTemporalConnections:
            for j in range(1, self.ShortTemporalConnections):
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                    # Storing the generation variables for current period
                        self.ia.append(self.thermalgenerationramps[i, j - 1][1] + k)
                        self.ja.append(self.thermalgenerators[i, j][1] + k)
                        self.ar.append(1.0)
                        self.ne += 1
                    # Storing the generation variables for previous period
                        self.ia.append(self.thermalgenerationramps[i, j - 1][1] + k)
                        self.ja.append(self.thermalgenerators[i, j - 1][1] + k)
                        self.ar.append(-1.0)
                        self.ne += 1
                    # Defining the resources (b) for the constraints
                        self.solver.set_row_bnds(\
                            str(self.thermalgenerationramps[i, j - 1][0]),\
                            k, 'bounded', -self.RampConvGen[k],\
                            self.RampConvGen[k])
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                    # Storing the generation variables for current period
                        self.ia.append(self.Hydrogenerationramps[i, j - 1][1] + k)
                        self.ja.append(self.Hydrogenerators[i, j][1] + k)
                        self.ar.append(1.0)
                        self.ne += 1
                    # Storing the generation variables for previous period
                        self.ia.append(self.Hydrogenerationramps[i, j - 1][1] + k)
                        self.ja.append(self.Hydrogenerators[i, j - 1][1] + k)
                        self.ar.append(-1.0)
                        self.ne += 1
                    # Defining the resources (b) for the constraints
                        self.solver.set_row_bnds(\
                            str(self.Hydrogenerationramps[i, j - 1][0]),\
                            k, 'bounded', -self.RampHydroGen[k],\
                            self.RampHydroGen[k])

    # Objective function

    def Objective_functionCommon(self):
        """ This class method defines the objective function of the economic
        dispatch in glpk """

        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
            # Cost for conventional generation    
                if self.NumberConvGen > 0: 
                    for k in range(self.NumberConvGen):
                        self.solver.set_obj_coef(\
                            str(self.thermalCG[i, j][0]),\
                            k, self.TotalHoursPerPeriod[j])
            # Cost for RES generation    
                if self.NumberRESGen > 0: 
                    for k in range(self.NumberRESGen):
                        self.solver.set_obj_coef(\
                            str(self.RESCG[i, j][0]),\
                            k, self.TotalHoursPerPeriod[j])
            # Cost for Hydroelectric generation    
                if self.NumberHydroGen > 0: 
                    for k in range(self.NumberHydroGen):
                        self.solver.set_obj_coef(\
                            str(self.HydroCG[i, j][0]),\
                            k, self.TotalHoursPerPeriod[j])
            # Operation cost of pumps
                if self.NumberPumps > 0:
                    for k in range(self.NumberPumps):
                        self.solver.set_obj_coef(\
                            str(self.pumpsvar[i, j][0]),\
                            k, -self.TotalHoursPerPeriod[j] \
                                * self.BaseUnitPower \
                                    * self.CostOperPumps[k])
            # Punitive cost for load curtailment
                if self.FlagProblem and self.FlagFeasibility:
                # Optimal Power Flow
                    for k in range(self.NumberContingencies + 1):
                        for ii in range(self.NumberNodesPS):
                            self.solver.set_obj_coef(\
                                str(self.LoadCurtailmentNode[i, j, k][0]),\
                                ii, self.TotalHoursPerPeriod[j] \
                                    * 100000000)
                        for ii in range(self.NumberNodesPS):
                            self.solver.set_obj_coef(\
                                str(self.GenerationCurtailmentNode\
                                [i, j, k][0]), ii, \
                                self.TotalHoursPerPeriod[j] * 100000000)
                elif not self.FlagProblem and self.FlagFeasibility:
                # Economic Dispatch
                # TODO: Set a parameter penalty in pyeneN
                    self.solver.set_obj_coef(\
                                str(self.loadcurtailmentsystem[i, j][0]),\
                                0, self.TotalHoursPerPeriod[j] \
                                    * 100000000)

    #############################
    ###   ECONOMIC DISPATCH   ###
    #############################

    def EconomicDispatchModel(self):
        """ This class method builds the optimisation model
        for the economic dispatch problem """

        # define matrix of coeficients (matrix A)
        self.variablesED()
        self.coeffmatrixED()
        self.Objective_functionCommon()

    def coeffmatrixED(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) to be
        # later added to glpk
        self.ia = []
        self.ja = []
        self.ar = []
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
        if self.FlagFeasibility:
            self.loadcurtailmentsystem = np.empty((\
                len(self.LongTemporalConnections),\
                self.ShortTemporalConnections),\
                dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
                # in matrix A (rows) of variables
                # for load curtailment in the system for each tree node
            self.generationcurtailmentsystem = np.empty((\
                len(self.LongTemporalConnections),\
                self.ShortTemporalConnections),\
                dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
                # in matrix A (rows) of variables
                # for generation curtailment in the system for each tree node

    def variablesED(self):
        """ This class method defines the variables and their limits for the
        economic dispatch problem """
        self.PosvariablesED()

        self.variablesCommon()
        
        # Reserving space in glpk for ED variables
        if self.FlagFeasibility:
            for i in self.LongTemporalConnections:
                for j in range(self.ShortTemporalConnections):
                    self.loadcurtailmentsystem[i, j] = (\
                        'LCS'+str(i)+','+str(j),\
                        self.solver.add_cols('LCS'+str(i)+','+str(j), 1))
                    self.generationcurtailmentsystem[i, j] = (\
                        'GCS'+str(i)+','+str(j),\
                        self.solver.add_cols('GCS'+str(i)+','+str(j), 1))    

    # Constraints ED        

    def activepowerbalancesystem(self):
        """ This class method writes the power balance constraint in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        self.powerbalance = np.empty((len(self.LongTemporalConnections),\
            self.ShortTemporalConnections), dtype=[('napos', 'U80'),\
            ('nupos', 'i4')]) # Start position 
                    # of active power balance constraints (rows) 
                    # for each tree node
        
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                self.powerbalance[i, j] = ('PB'+str(i)+','+str(j),\
                    self.solver.add_rows('PB'+str(i)+','+str(j), 1))  # Number of 
                        # columns (constraints) in matrix A for the active 
                        # power balance constraints fo each period and each 
                        # tree node


        # Generating the matrix A for the active power balance constraints
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
            # Storing the thermal generation variables
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        self.ia.append(self.powerbalance[i, j][1])
                        self.ja.append(self.thermalgenerators[i, j][1] + k)
                        self.ar.append(1.0)
                        self.ne += 1
            # Storing the RES generation variables
                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        self.ia.append(self.powerbalance[i, j][1])
                        self.ja.append(self.RESgenerators[i, j][1] + k)
                        self.ar[self.ne] = 1.0
                        self.ne += 1
            # Storing the Hydroelectric generation variables
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        self.ia.append(self.powerbalance[i, j][1])
                        self.ja.append(self.Hydrogenerators[i, j][1] + k)
                        self.ar.append(1.0)
                        self.ne += 1
            # Storing variables for ESS
            # TODO: Modify the constraint for the first period
                if self.NumberStorageDevices > 0:
                    if j > 0: # Start only after the first period
                        for k in range(self.NumberStorageDevices):
                            self.ia.append(self.powerbalance[i, j][1])
                            self.ja.append(self.ESS[i, j][1] + k)
                            self.ar.append(self.EffStorage[k] \
                                / self.TotalHoursPerPeriod[j - 1])
                            self.ne += 1
                        for k in range(self.NumberStorageDevices):
                            self.ia.append(self.powerbalance[i, j][1])
                            self.ja.append(self.ESS[i, j - 1][1] + k)
                            self.ar.append(-self.EffStorage[k] \
                                / self.TotalHoursPerPeriod[j - 1])
                            self.ne += 1
            # Storing the variables for load and generation curtailment
                if self.FlagFeasibility:
                    self.ia.append(self.powerbalance[i, j][1])
                    self.ja.append(self.loadcurtailmentsystem[i, j][1])
                    self.ar.append(1.0)
                    self.ne += 1
                    self.ia.append(self.powerbalance[i, j][1])
                    self.ja.append(self.generationcurtailmentsystem[i, j][1])
                    self.ar.append(-1.0)
                    self.ne += 1
            # Storing the variables for pumps
                if self.NumberPumps > 0:
                    for k in range(self.NumberPumps):
                        self.ia.append(self.powerbalance[i, j][1])
                        self.ja.append(self.pumpsvar[i, j][1] + k)
                        self.ar.append(-1.0)
                        self.ne += 1
            # Defining the resources (b) for the constraints
                totaldemand = 0                
                # TODO: Change the inputs of losses and demand scenarios
                # for parameters
                if self.NumberDemScenarios == 0:
                    if self.PercentageLosses is None:
                        for k in range(self.NumberNodesPS):
                            totaldemand = totaldemand + \
                                self.PowerDemandNode[k] * \
                                self.MultScenariosDemand[i, k]
                    else:
                        for k in range(self.NumberNodesPS):
                            totaldemand = totaldemand + \
                                self.PowerDemandNode[k] * \
                                self.MultScenariosDemand[i, k] * \
                                (1 + self.PercentageLosses)
                else:
                    if self.PercentageLosses is None:
                        for k in range(self.NumberNodesPS):
                            totaldemand = totaldemand + \
                                self.PowerDemandNode[k] * \
                                self.MultScenariosDemand[i, j, k]
                    else:
                        for k in range(self.NumberNodesPS):
                            totaldemand = totaldemand + \
                                self.PowerDemandNode[k] * \
                                self.MultScenariosDemand[i, j, k] * \
                                (1 + self.PercentageLosses)
                self.solver.set_row_bnds(str(self.powerbalance[i, j][0]), 0,\
                    'fixed', totaldemand, totaldemand)

    ########################################
    ###   OPTIMAL POWER FLOW FULL MODEL  ###
    ########################################

    def OptimalPowerFlowModel(self):
        """ This class method builds the optimisation model
        for the optimal power flow problem """

        # define matrix of coeficients (matrix A)
        self.variablesOPF()
        self.coeffmatrixOPF()
        self.Objective_functionCommon()

    def coeffmatrixOPF(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) to be
        # later added to glpk
        self.ia = [] # Position in rows
        self.ja = [] # Position in columns
        self.ar = [] # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A
        
        self.piecewiselinearisationcost()
        # self.generationrampsconstraints()

        self.activepowerbalancepernode()
        self.activepowerflowconstraints()
        if self.LossesFlag:
            self.activepowerlosses1constraints()
            self.activepowerlosses2constraints()

        self.solver.load_matrix(self.ne, self.ia, self.ja, self.ar)

    # Variables ED

    def PosvariablesOPF(self):
        """ This class method creates the vector that stores the positions of 
        variables for the ED problem """

        self.PosvariablesCommon()

        self.ActivePowerFlow = np.empty((len(self.LongTemporalConnections),\
            self.ShortTemporalConnections, (self.NumberContingencies + 1)),\
            dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
            # in matrix A (rows) of variables for the active power flow
        if self.LossesFlag:
            self.ActivePowerLosses = np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, (self.NumberContingencies + 1)),\
                dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
                # in matrix A (rows) of variables for the active power losses
        if self.FlagFeasibility:
            self.LoadCurtailmentNode = np.empty((\
                len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1)),\
                dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
                # in matrix A (rows) of variables for load curtailment per
                # node
            self.GenerationCurtailmentNode = np.empty((\
                len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1)),\
                dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
                # in matrix A (rows) of variables for generation 
                # curtailment per node
        self.VoltageAngle = np.empty((len(self.LongTemporalConnections),\
            self.ShortTemporalConnections, (self.NumberContingencies + 1)),\
            dtype=[('napos', 'U80'),('nupos', 'i4')]) # Start position
            # in matrix A (rows) of variables for the voltage angle per
            # node

    def variablesOPF(self):
        """ This class method defines the variables and their limits for the
        economic dispatch problem """
        self.PosvariablesOPF()

        self.variablesCommon()
        
        # Reserving space in glpk for OPF variables
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    self.ActivePowerFlow[i, j, k] = \
                        ('ActivePowerFlow'+str(i)+','+str(j)+','+str(k),\
                        self.solver.add_cols(\
                        'ActivePowerFlow'+str(i)+','+str(j)+','+str(k),\
                        self.NumberLinesPS))
                    if self.LossesFlag:
                         self.ActivePowerLosses[i, j, k] = \
                            ('ActivePowerLosses'+str(i)+','+str(j)+','+str(k),\
                            self.solver.add_cols(\
                            'ActivePowerLosses'+str(i)+','+str(j)+','+str(k),\
                            self.NumberLinesPS))
                    if self.FlagFeasibility:
                        self.LoadCurtailmentNode[i, j, k] = \
                            ('LoadCurtailmentNode'+str(i)+',\
                                '+str(j)+','+str(k),\
                            self.solver.add_cols(\
                            'LoadCurtailmentNode'+str(i)+',\
                                '+str(j)+','+str(k),\
                            self.NumberNodesPS))
                        self.GenerationCurtailmentNode[i, j, k] = \
                            ('GenerationCurtailmentNode'+str(i)+',\
                            '+str(j)+','+str(k), self.solver.add_cols(\
                            'GenerationCurtailmentNode'+str(i)+',\
                            '+str(j)+','+str(k), self.NumberNodesPS))
                    self.VoltageAngle[i, j, k] = \
                        ('VoltageAngle'+str(i)+','+str(j)+','+str(k),\
                        self.solver.add_cols(\
                        'VoltageAngle'+str(i)+','+str(j)+','+str(k),\
                        self.NumberNodesPS))

        
        # Defining the limits of the variables
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberLinesPS):
                        # If the line is active in the current contingency then
                        # define the limits
                        if self.ActiveBranches[k, ii] and \
                            self.PowerRateLimitTL[ii] > 0:
                            self.solver.set_col_bnds(\
                                str(self.ActivePowerFlow[i, j, k][0]), ii,\
                                'bounded', \
                                -self.PowerRateLimitTL[ii],\
                                self.PowerRateLimitTL[ii])
                        # If the line is not active in the current contingency 
                        # then fix the active power flow to zero
                        elif self.ActiveBranches[k, ii] and \
                            self.PowerRateLimitTL[ii] == 100:
                            self.solver.set_col_bnds(\
                                str(self.ActivePowerFlow[i, j, k][0]), ii,\
                                'free', \
                                -self.PowerRateLimitTL[ii],\
                                self.PowerRateLimitTL[ii])
                        else:
                            self.solver.set_col_bnds(\
                                str(self.ActivePowerFlow[i, j, k][0]), ii,\
                                'fixed', 0, 0)
                    if self.LossesFlag:
                        for ii in range(self.NumberLinesPS):
                        # If the line is not active in the current contingency 
                        # then fix the active power losses to zero
                            if not self.ActiveBranches[k, ii]:
                                self.solver.set_col_bnds(\
                                    str(self.ActivePowerLosses[i, j, k][0]),\
                                        ii, 'fixed', 0, 0)
                    if self.FlagFeasibility:
                        for ii in range(self.NumberNodesPS):
                            # If the demand in the node is greater than zero
                            # then define the limits
                            if self.PowerDemandNode[ii] > 0:
                                if self.NumberDemScenarios == 0:
                                    self.solver.set_col_bnds(\
                                        str(\
                                        self.LoadCurtailmentNode[i, j, k][0]),\
                                        ii,'bounded', 0, \
                                        self.PowerDemandNode[ii] * \
                                        self.MultScenariosDemand[i, ii])
                                else:
                                    self.solver.set_col_bnds(\
                                        str(\
                                        self.LoadCurtailmentNode[i, j, k][0]),\
                                        ii,'bounded', 0, \
                                        self.PowerDemandNode[ii] * \
                                        self.MultScenariosDemand[i, j, ii])                        
                            # If the demand in the node is zero then
                            # fix the load curtailment to zero
                            else:
                                self.solver.set_col_bnds(\
                                    str(self.LoadCurtailmentNode[i, j, k][0]),\
                                    ii, 'fixed', 0, 0)
                        for ii in range(self.NumberNodesPS):
                            flag_gen = False
                            if self.NumberConvGen > 0:
                                for jj in range(self.NumberConvGen):
                                    if self.OriginalNumberConvGen[jj] == \
                                        self.OriginalNumberNodes[ii]:
                                        flag_gen=True
                                        break
                            # Storing the RES generation curtailment 
                            # variables
                            if self.NumberRESGen > 0 and not flag_gen:
                                for jj in range(self.NumberRESGen):
                                    if self.OriginalNumberRESGen[jj] == \
                                        self.OriginalNumberNodes[ii]:
                                        flag_gen=True
                                        break
                            # Storing the Hydro generation curtailment 
                            # variables
                            if self.NumberHydroGen > 0 and not flag_gen:
                                for jj in range(self.NumberHydroGen):
                                    if self.OriginalNumberHydroGen[jj] == \
                                        self.OriginalNumberNodes[ii]:
                                        flag_gen=True
                                        break
                            if flag_gen:
                                self.solver.set_col_bnds(\
                                str(self.GenerationCurtailmentNode\
                                    [i, j, k][0]), ii, 'lower', 0.0, 0.0)
                            else:
                                self.solver.set_col_bnds(\
                                str(self.GenerationCurtailmentNode\
                                    [i, j, k][0]), ii, 'fixed', 0.0, 0.0)
                    for ii in range(self.NumberNodesPS):
                        if self.TypeNode[ii] != 3:
                            self.solver.set_col_bnds(\
                                str(self.VoltageAngle[i, j, k][0]),\
                                    ii,'free', 0, 0)
                        else:
                            self.solver.set_col_bnds(\
                                str(self.VoltageAngle[i, j, k][0]),\
                                    ii,'fixed', 0, 0)

    # Constraints OPF

    def activepowerbalancepernode(self):
        """ This class method writes the power balance constraint in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        # Creating the matrices to store the position of constraints in
        # matrix A
        self.activepowerbalancenode = np.empty(\
            (len(self.LongTemporalConnections),\
            self.ShortTemporalConnections, \
            (self.NumberContingencies + 1)), dtype=[('napos', 'U80'),\
            ('nupos', 'i4')]) # Start position of active power balance 
                              # constraints (rows) per node
        
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    self.activepowerbalancenode[i, j, k] = \
                        ('activepowerbalancenode'+str(i)+','+str(j)+','+str(k),\
                        self.solver.add_rows(\
                        'activepowerbalancenode'+str(i)+','+str(j)+','+str(k),\
                        self.NumberNodesPS))  # Number of 
                            # rows (constraints) in matrix A for the active 
                            # power balance constraints per node

        # Generating the matrix A for the active power balance constraints
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberNodesPS):
                        if self.TypeNode[ii] != 4:
                        # Storing the thermal generation variables
                            if self.NumberConvGen > 0:
                                for jj in range(self.NumberConvGen):
                                    if self.OriginalNumberConvGen[jj] == \
                                        self.OriginalNumberNodes[ii]:
                                        self.ia.append(self.activepowerbalancenode\
                                            [i, j, k][1] + ii)
                                        self.ja.append(self.thermalgenerators[i, j][1] + jj)
                                        self.ar.append(1.0)
                                        self.ne += 1
                        # Storing the RES generation variables
                            if self.NumberRESGen > 0:
                                for jj in range(self.NumberRESGen):
                                    if self.OriginalNumberRESGen[jj] == \
                                        self.OriginalNumberNodes[ii]:
                                        self.ia.append(self.activepowerbalancenode\
                                            [i, j, k][1] + ii)
                                        self.ja.append(self.RESgenerators[i, j][1] + jj)
                                        self.ar.append(1.0)
                                        self.ne += 1
                        # Storing the Hydroelectric generation variables
                            if self.NumberHydroGen > 0:
                                for jj in range(self.NumberHydroGen):
                                    if self.OriginalNumberHydroGen[jj] == \
                                        self.OriginalNumberNodes[ii]:
                                        self.ia.append(self.activepowerbalancenode\
                                            [i, j, k][1] + ii)
                                        self.ja.append(self.Hydrogenerators[i, j][1] + jj)
                                        self.ar.append(1.0)
                                        self.ne += 1
                        # Storing variables for ESS
                        # TODO: Modify the constraint for the first period
                        # TODO: create an input for storage without the 
                        # Link List
                            if self.NumberStorageDevices > 0:
                                if j > 0: # Start only after the first period
                                    for jj in range(self.NumberStorageDevices):
                                        self.ia.append(self.activepowerbalancenode\
                                            [i, j, k][1] + ii)
                                        self.ja.append(self.ESS[i, j][1] + jj)
                                        self.ar.append(self.EffStorage[jj] \
                                            / self.TotalHoursPerPeriod[j - 1])
                                        self.ne += 1
                                    for k in range(self.NumberStorageDevices):
                                        self.ia.append(self.activepowerbalancenode\
                                            [i, j, k][1] + ii)
                                        self.ja.append(self.ESS[i, j - 1][1] + jj)
                                        self.ar.append(-self.EffStorage[jj] \
                                            / self.TotalHoursPerPeriod[j - 1])
                                        self.ne += 1
                        # Storing the variables for pumps
                        # TODO: create an input for storage without the 
                        # Link List
                            if self.NumberPumps > 0:
                                for jj in range(self.NumberPumps):
                                    if self.MaxPowerPumps[jj] > 0:
                                        self.ia.append(self.activepowerbalancenode\
                                                [i, j, k][1] + ii)
                                        self.ja.append(self.pumpsvar[i, j][1] + jj)
                                        self.ar.append(-1.0)
                                        self.ne += 1
                        # Storing the variables for active power flows
                            for jj in range(self.NumberLinesPS):
                                if self.OriginalNumberBranchFrom[jj] ==\
                                    self.OriginalNumberNodes[ii]:
                                    self.ia.append(self.activepowerbalancenode\
                                            [i, j, k][1] + ii)
                                    self.ja.append(self.ActivePowerFlow[i, j, k][1] + jj)
                                    self.ar.append(-1.0)
                                    self.ne += 1
                                if self.OriginalNumberBranchTo[jj] ==\
                                    self.OriginalNumberNodes[ii]:
                                    self.ia.append(self.activepowerbalancenode\
                                            [i, j, k][1] + ii)
                                    self.ja.append(self.ActivePowerFlow[i, j, k][1] + jj)
                                    self.ar.append(1.0)
                                    self.ne += 1
                        # Storing the variables for active power losses
                            if self.LossesFlag:
                                for jj in range(self.NumberLinesPS):
                                    if self.OriginalNumberBranchFrom[jj] ==\
                                        self.OriginalNumberNodes[ii] or \
                                        self.OriginalNumberBranchTo[jj] ==\
                                        self.OriginalNumberNodes[ii]:
                                        self.ia.append(self.activepowerbalancenode\
                                                [i, j, k][1] + ii)
                                        self.ja.append(self.ActivePowerLosses[i, j, k][1]\
                                                + jj)
                                        self.ar.append(-0.5)
                                        self.ne += 1

                        # Storing the variables for load curtailment
                            if self.FlagFeasibility:
                                self.ia.append(self.activepowerbalancenode[i, j, k][1] + ii)
                                self.ja.append(self.LoadCurtailmentNode[i, j, k][1] + ii)
                                self.ar.append(1.0)
                                self.ne += 1
                                # Storing the thermal generation curtailment 
                                # variables
                                self.ia.append(self.activepowerbalancenode\
                                    [i, j, k][1] + ii)
                                self.ja.append(self.GenerationCurtailmentNode\
                                    [i, j, k][1] + ii)
                                self.ar.append(-1.0)
                                self.ne += 1

                        # Defining the resources (b) for the constraints
                            totaldemand = 0                
                            # TODO: Change the inputs of losses and demand scenarios
                            # for parameters
                            if self.NumberDemScenarios == 0:
                                if self.PercentageLosses is None:
                                    totaldemand = totaldemand + \
                                        self.PowerDemandNode[ii] * \
                                        self.MultScenariosDemand[i, ii]
                                else:
                                    totaldemand = totaldemand + \
                                        self.PowerDemandNode[ii] * \
                                        self.MultScenariosDemand[i, ii] * \
                                        (1 + self.PercentageLosses)
                            else:
                                if self.PercentageLosses is None:
                                    totaldemand = totaldemand + \
                                        self.PowerDemandNode[ii] * \
                                        self.MultScenariosDemand[i, j, ii]
                                else:
                                    totaldemand = totaldemand + \
                                        self.PowerDemandNode[ii] * \
                                        self.MultScenariosDemand[i, j, ii] * \
                                        (1 + self.PercentageLosses)

                            totalresource = totaldemand

                            if not self.LossesFlag:
                                totalnontechnicallosses = 0
                                for jj in range(self.NumberLinesPS):
                                    if self.OriginalNumberBranchFrom[jj] ==\
                                        self.OriginalNumberNodes[ii]:
                                        totalnontechnicallosses += \
                                            0.5*self.NontechnicalLosses[jj]
                                    if self.OriginalNumberBranchTo[jj] ==\
                                        self.OriginalNumberNodes[ii]:
                                        totalnontechnicallosses += \
                                            0.5*self.NontechnicalLosses[jj]                        
                                totalresource += totalnontechnicallosses

                            self.solver.set_row_bnds(\
                                str(self.activepowerbalancenode[i, j, k][0]), ii,\
                                'fixed', totalresource, totalresource)

    def activepowerflowconstraints(self):
        """ This class method writes the active power flow constraints in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        self.activepowerflowconstraint = np.empty(\
            (len(self.LongTemporalConnections),\
            self.ShortTemporalConnections, \
            (self.NumberContingencies + 1)), dtype=[('napos', 'U80'),\
            ('nupos', 'i4')]) # Start position of active power flow 
                              # constraints (rows) per line
        
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    # Pre-contingency
                    if k == 0:
                        self.activepowerflowconstraint[i, j, k] = \
                            ('activepowerflowconstraint'+str(i)+','+str(j)\
                            +','+str(k), self.solver.add_rows(\
                            'activepowerflowconstraint'+str(i)+','+str(j)\
                            +','+str(k),self.NumberLinesPS))  # Number of
                                # rows (constraints) in matrix A for the active 
                                # power flow constraints per line
                    # Post-contingency
                    else:
                        self.activepowerflowconstraint[i, j, k] = \
                            ('activepowerflowconstraint'+str(i)+','+str(j)\
                            +','+str(k),self.solver.add_rows(\
                            'activepowerflowconstraint'+str(i)+','+str(j)\
                            +','+str(k),self.NumberLinesPS - 1))  
                                # Number of 
                                # rows (constraints) in matrix A for the active 
                                # power flow constraints per line

        # Generating the matrix A for the active power flow constraints
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    counter = 0
                    for ii in range(self.NumberLinesPS):
                        if self.ActiveBranches[k, ii]:
                        # Storing the active power flow variables
                            self.ia.append(\
                                self.activepowerflowconstraint[i, j, k][1] \
                                + counter)
                            self.ja.append(\
                                self.ActivePowerFlow[i, j, k][1] + ii)
                            self.ar.append(1.0)
                            self.ne += 1
                        # Storing the voltage angle variables at end "from"
                            self.ia.append(\
                                self.activepowerflowconstraint[i, j, k][1] \
                                + counter)
                            self.ja.append(self.VoltageAngle[i, j, k][1] + \
                                self.PosNumberBranchFrom[ii])
                            self.ar.append(-1.0/self.ReactanceBranch[ii])
                            self.ne += 1
                        # Storing the voltage angle variables at end "to"
                            self.ia.append(\
                                self.activepowerflowconstraint[i, j, k][1] \
                                + counter)
                            self.ja.append(self.VoltageAngle[i, j, k][1] + \
                                    self.PosNumberBranchTo[ii])
                            self.ar.append(1.0/self.ReactanceBranch[ii])
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

        self.activepowerlosses1 = np.empty(\
            (len(self.LongTemporalConnections),\
            self.ShortTemporalConnections, \
            (self.NumberContingencies + 1), \
            self.NumberLinesPS), dtype=[('napos', 'U80'),\
            ('nupos', 'i4')]) # Start position of active power losses 
                              # constraints (rows) per line and per piece
        
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberLinesPS):
                        # If the line is active in the current contingency
                        # then reserve the space
                        if self.ActiveBranches[k, ii]:
                            self.activepowerlosses1[i, j, k, ii] = \
                                ('activepowerlosses1'+str(i)+','\
                                +str(j)+','+str(k)+','+str(ii),\
                                self.solver.add_rows(\
                                'activepowerlosses1'+str(i)+','\
                                +str(j)+','+str(k)+','+str(ii), \
                                self.NumberPiecesTLLosses)) # Number
                                    # of rows (constraints) in matrix A 
                                    # for the active power losses 
                                    # constraints per line 
                                    # and per piece

        # Generating the matrix A for the active power losses constraints
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberLinesPS):
                        if self.ActiveBranches[k, ii]:
                            for jj in range(self.NumberPiecesTLLosses):
                            # Storing the active power losses variables
                                self.ia.append(\
                                    self.activepowerlosses1\
                                        [i, j, k, ii][1] + jj)
                                self.ja.append(\
                                    self.ActivePowerLosses[i, j, k][1] \
                                        + ii)
                                self.ar.append(1.0)
                                self.ne += 1
                            # Storing the active power flow variables
                                self.ia.append(\
                                    self.activepowerlosses1\
                                        [i, j, k, ii][1] + jj)
                                self.ja.append(\
                                    self.ActivePowerFlow[i, j, k][1] \
                                        + ii)
                                self.ar.append(\
                                    -self.BCoeffPWBranchLosses[jj]\
                                    * self.ResistanceBranch[ii])
                                self.ne += 1
                            # Defining the resources (b) for the constraints
                                self.solver.set_row_bnds(\
                                    str(self.activepowerlosses1\
                                        [i, j, k, ii][0]), jj, 'lower', \
                                        self.ACoeffPWBranchLosses[jj]\
                                        * self.ResistanceBranch[ii], 0)

    def activepowerlosses2constraints(self):
        """ This class method writes the active power losses constraints in glpk
        
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """

        self.activepowerlosses2 = np.empty(\
            (len(self.LongTemporalConnections),\
            self.ShortTemporalConnections, \
            (self.NumberContingencies + 1), \
            self.NumberLinesPS), dtype=[('napos', 'U80'),\
            ('nupos', 'i4')]) # Start position of active power losses 
                              # constraints (rows) per line and per piece

        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    if self.LossesFlag:
                        for ii in range(self.NumberLinesPS):
                            # If the line is active in the current contingency
                            # then reserve the space
                            if self.ActiveBranches[k, ii]:
                                self.activepowerlosses2[i, j, k, ii] = \
                                    ('activepowerlosses2'+str(i)+','\
                                    +str(j)+','+str(k)+','+str(ii), \
                                    self.solver.add_rows(\
                                    'activepowerlosses2'+str(i)+','\
                                    +str(j)+','+str(k)+','+str(ii), \
                                    self.NumberPiecesTLLosses)) # Number
                                        # of rows (constraints) in matrix A 
                                        # for the active power losses 
                                        # constraints per line 
                                        # and per piece

        # Generating the matrix A for the active power losses constraints
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberLinesPS):
                        if self.ActiveBranches[k, ii]:
                            for jj in range(self.NumberPiecesTLLosses):
                            # Storing the active power losses variables
                                self.ia.append(\
                                    self.activepowerlosses2\
                                        [i, j, k, ii][1] + jj)
                                self.ja.append(\
                                    self.ActivePowerLosses[i, j, k][1] \
                                        + ii)
                                self.ar.append(1.0)
                                self.ne += 1
                            # Storing the active power flow variables
                                self.ia.append(\
                                    self.activepowerlosses2\
                                        [i, j, k, ii][1] + jj)
                                self.ja.append(\
                                    self.ActivePowerFlow[i, j, k][1] \
                                        + ii)
                                self.ar.append(\
                                    self.BCoeffPWBranchLosses[jj]\
                                    * self.ResistanceBranch[ii])
                                self.ne += 1
                            # Defining the resources (b) for the constraints
                                self.solver.set_row_bnds(\
                                    str(self.activepowerlosses2\
                                        [i, j, k, ii][0]), jj, 'lower', \
                                        self.ACoeffPWBranchLosses[jj]\
                                        * self.ResistanceBranch[ii], 0)

    ########################################
    ###   OPTIMAL POWER FLOW CPP MODELS  ###
    ########################################

    def OptimalPowerFlowModelCPP(self):
        """ This class method builds the optimisation model
        for the optimal power flow problem using a fast implementation of 
        different mathematical models in c++ """

        self.network_model = models_cpp()
        self.set_parameters_cpp_models()
        if self.solver_problem == "CLP":
            self.network_model.run_reduced_dc_opf_cpp()
        elif self.solver_problem == "CLP-I":
            self.network_model.run_iterative_reduced_dc_opf_cpp()
        elif self.solver_problem == "CLP-IR":
            self.network_model.run_iterative_reduced_dc_opf_v2_cpp()
            
        # retrieving solution
        aux_gen, aux_gen_cost = \
            self.network_model.get_generation_solution_cpp()
        
        aux_flows = self.network_model.get_branch_solution_cpp()

        aux_angles, aux_load_cur, aux_gen_cur = self.network_model.get_node_solution_cpp()
        
        self.ThermalGenerationSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberConvGen))
        self.RESGenerationSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberRESGen))
        self.HydroGenerationSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberHydroGen))
        
        self.GenerationCurtailmentNodesSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberNodesPS))
        
        self.ThermalGenerationCostSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberConvGen))
        self.RESGenerationCostSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberRESGen))
        self.HydroGenerationCostSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberHydroGen))
        
        self.VoltageAngleSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberNodesPS))
        
        self.LoadCurtailmentNodesSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberNodesPS))
        
        self.ActivePowerFlowSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberLinesPS))

        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        if self.ActiveConv[k]:
                            self.ThermalGenerationSolution[i, j, k] = \
                                aux_gen[counter]
                            counter += 1
                        else:
                            self.ThermalGenerationSolution[i, j, k] = 0.0
                    
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        self.HydroGenerationSolution[i, j, k] = \
                            aux_gen[counter]
                        counter += 1
                
                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        self.RESGenerationSolution[i, j, k] = \
                            aux_gen[counter]
                        counter += 1
        
        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        if self.ActiveConv[k]:
                            self.ThermalGenerationCostSolution[i, j, k] = \
                                aux_gen_cost[counter]
                            counter += 1
                        else:
                            self.ThermalGenerationCostSolution[i, j, k] = 0.0
                    
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        self.HydroGenerationCostSolution[i, j, k] = \
                            aux_gen_cost[counter]
                        counter += 1
                
                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        self.RESGenerationCostSolution[i, j, k] = \
                            aux_gen_cost[counter]
                        counter += 1
        
        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberLinesPS):
                        if self.ActiveBranches[k, ii]:
                            self.ActivePowerFlowSolution[i, j, k ,ii] = \
                                aux_flows[counter]
                            counter += 1
                        else:
                            self.ActivePowerFlowSolution[i, j, k ,ii] =  0.0
        
        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberNodesPS):
                        if self.TypeNode[ii] != 4:
                            self.VoltageAngleSolution[i, j, k ,ii] = \
                                aux_angles[counter]
                            self.LoadCurtailmentNodesSolution[i, j, k ,ii] = \
                                aux_load_cur[counter]
                            self.GenerationCurtailmentNodesSolution[i, j, k, ii] = \
                                aux_gen_cur[counter]
                            counter += 1
                        else:
                            self.VoltageAngleSolution[i, j, k ,ii] =  0.0
                            self.LoadCurtailmentNodesSolution[i, j, k ,ii] =  0.0
                            self.GenerationCurtailmentNodesSolution[i, j, k, ii] = 0.0

    def set_parameters_cpp_models(self):
        """ This class method set all parameters in the c++ implementation """
        # Information nodes
        slack = self.OriginalNumberNodes[0]
        for ii in range(self.NumberNodesPS):
            if self.TypeNode[ii] != 4:
                if self.TypeNode[ii] == 3:
                    slack = self.OriginalNumberNodes[ii]
                demand_vals = []
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        if self.NumberDemScenarios == 0:
                            demand_vals.append(self.PowerDemandNode[ii] * \
                                self.MultScenariosDemand[i, ii])
                        else:
                            demand_vals.append(self.PowerDemandNode[ii] * \
                                self.MultScenariosDemand[i, j, ii])
                self.network_model.add_bus_cpp(demand_vals, \
                    self.OriginalNumberNodes[ii], 'ac')
        
        # Information transmission lines
        # TODO: implement the same for transformers
        for xtl in range(self.NumberLinesPS):
            if self.ActiveBranches[0, xtl]:
                self.network_model.add_branch_cpp([self.ReactanceBranch[xtl]], \
                    [self.ResistanceBranch[xtl]], [self.PowerRateLimitTL[xtl]], \
                    self.OriginalNumberBranchFrom[xtl], \
                    self.OriginalNumberBranchTo[xtl], xtl, 'ac_transmission_line')
        
        # Information for generators        
        counter_gen = 0
        if self.NumberConvGen > 0:
            for xgen in range(self.NumberConvGen):
                if self.ActiveConv[xgen]:
                    P_max = []
                    P_min = []
                    for i in self.LongTemporalConnections:
                        for j in range(self.ShortTemporalConnections):
                            P_max.append(self.MaxConvGen[xgen])
                            P_min.append(self.MinConvGen[xgen])
                    self.network_model.add_generator_cpp(P_max, P_min, \
                        self.OriginalNumberConvGen[xgen], counter_gen, 'conv', 0.0, \
                        0.0, self.ACoeffPWConvGen[xgen,:], \
                        self.BCoeffPWConvGen[xgen,:], self.ActiveConv[xgen])
                    counter_gen += 1
        if self.NumberHydroGen > 0:
            for xgen in range(self.NumberHydroGen):
                P_max = []
                P_min = []
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        P_max.append(self.MaxHydroGen[xgen])
                        P_min.append(self.MinHydroGen[xgen])
                self.network_model.add_generator_cpp(P_max, P_min, \
                    self.OriginalNumberHydroGen[xgen], counter_gen, 'hydro', 0.0,
                    0.0, self.ACoeffPWHydroGen[xgen,:], \
                    self.BCoeffPWHydroGen[xgen,:], True)
                counter_gen += 1
        if self.NumberRESGen > 0:
            for xgen in range(self.NumberRESGen):
                P_max = []
                P_min = []
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        P_max.append(self.MaxRESGen[xgen]* \
                            self.RESScenarios[i, j, xgen])
                        P_min.append(self.MinRESGen[xgen])
                self.network_model.add_generator_cpp(P_max, P_min, \
                    self.OriginalNumberRESGen[xgen], counter_gen, 'RES', 0.0,
                    0.0, self.ACoeffPWRESGen[xgen,:], \
                    self.BCoeffPWRESGen[xgen,:], True)
                counter_gen += 1
        
        self.network_model.set_integer_data_power_system_cpp("number periods",\
            self.ShortTemporalConnections)
        self.network_model.set_integer_data_power_system_cpp(\
            "number representative days", len(self.LongTemporalConnections))
        self.network_model.set_integer_data_power_system_cpp(\
            "slack bus", slack)
        
        self.network_model.set_continuous_data_power_system_cpp(\
            "total hours period", self.TotalHoursPerPeriod[0])
        self.network_model.set_continuous_data_power_system_cpp(\
            "base power", self.BaseUnitPower)
        

    # Data inputs of Network model

    def set_value_network(self, ID=None, name=None, position_tree=None, hour=None, typ=None, value=None):
        ''' This function set the value of a variable or parameter.

            ----------
            Description:\\
            * If the parameter or variable exist then the value is overwritten\\
            * If the parameter or variable does not exist then the value is created
        
            Parameters
            ----------
            Mandatory:\\
            ID              :   Unique ID of the network element\\
            name            :   Name of variable or parameter to be retrieved\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree. If the value does not vary with
                                the energy tree then this value should be left in None\\
            hour            :   integer that indicates the specific hour of the requested data. If
                                the data does not change in time then this input must be left in
                                None\\
            typ             :   This refers to the type of element to be retrieved. This value
                                can be either "variable" or "parameter". Other types will not
                                be accepted
            value           :   Value to be set on the variable or parameter
        '''
        for node in self.network.nodes(data=True):
            if node['obj'].ID == ID and typ == "parameter":
                exist = False
                for parameter in node['obj'].parameters:
                    if parameter.name == name and (not hour or hour == parameter.hour) and position_tree == parameter.position_tree:
                        exist = True
                        parameter.value = value
                        break
                if not exist:
                    parameter = network_parameter
                    parameter.name = name
                    parameter.position_tree = position_tree
                    parameter.hour = hour
                    parameter.bus = node.bus
                    parameter.ends = node.ends
                    parameter.type = node.type
                    parameter.subtype = node.subtype
                    parameter.ID = node.ID
                    parameter.value = value
                    node.parameters.append(parameter)
            if node['obj'].ID == ID and typ == "variable":
                exist = False
                for variable in node['obj'].variables:
                    if variable.name == name and (not hour or hour == variable.hour) and position_tree == variable.position_tree:
                        exist = True
                        variable.value = value
                        break
                if not exist:
                    variable = network_variable
                    variable.name = name
                    variable.position_tree = position_tree
                    variable.hour = hour
                    variable.bus = node.bus
                    variable.ends = node.ends
                    variable.type = node.type
                    variable.subtype = node.subtype
                    variable.ID = node.ID
                    variable.value = value
                    node.variables.append(variable)
    


    def SetLongTemporalConnections(self, \
        long_temporal_connections=None):
        assert long_temporal_connections is not None, \
            "No value for the nodes of the temporal tree to be \
            analised" 
        self.LongTemporalConnections = \
            long_temporal_connections

    def SetShortTemporalConnections(self, \
        short_temporal_connections=None):
        assert short_temporal_connections is not None, \
            "No value for the number of subperiods in a 24h period" 
        self.ShortTemporalConnections = \
            short_temporal_connections

    def SetNumberConvGen(self, \
        number_conv_gen=None):
        assert number_conv_gen is not None, \
            "No value for the number of conventional generators" 
        self.NumberConvGen = \
            number_conv_gen

    def SetNumberRESGen(self, \
        number_RES_gen=None):
        assert number_RES_gen is not None, \
            "No value for the number of RES generators" 
        self.NumberRESGen = \
            number_RES_gen

    def SetNumberHydroGen(self, \
        number_Hydro_gen=None):
        assert number_Hydro_gen is not None, \
            "No value for the number of Hydro generators" 
        self.NumberHydroGen = \
            number_Hydro_gen

    def SetNumberPumps(self, \
        number_pumps=None):
        assert number_pumps is not None, \
            "No value for the number of pumps" 
        self.NumberPumps = \
            number_pumps

    def SetNumberStorageDevices(self, \
        number_storage_devices=None):
        assert number_storage_devices is not None, \
            "No value for the number of storage elements" 
        self.NumberStorageDevices = \
            number_storage_devices
    
    def SetNumberDemScenarios(self, \
        number_dem_scenarios=None):
        assert number_dem_scenarios is not None, \
            "No value for the number of demand scenarios" 
        self.NumberDemScenarios = \
            number_dem_scenarios
    
    def SetNumberNodesPS(self, \
        number_nodes_PS=None):
        assert number_nodes_PS is not None, \
            "No value for the number of nodes in the power system" 
        self.NumberNodesPS = \
            number_nodes_PS

    def SetNumberContingencies(self, \
        number_contingencies=None):
        assert number_contingencies is not None, \
            "No value for the number of contingencies" 
        self.NumberContingencies = \
            number_contingencies
    
    def SetNumberLinesPS(self, \
        number_lines_PS=None):
        assert number_lines_PS is not None, \
            "No value for the number of transmission lines and/or \
                transformers in the power system" 
        self.NumberLinesPS = \
            number_lines_PS
    
    def SetNumberPiecesTLLosses(self, \
        number_pieces_TL_losses=None):
        assert number_pieces_TL_losses is not None, \
            "No value for the number of pieces in the piecewise \
                linearisation of transmission lines and \
                transformers losses" 
        self.NumberPiecesTLLosses = \
            number_pieces_TL_losses
    
    def SetBaseUnitPower(self, \
        base_unit_power=None):
        assert base_unit_power is not None, \
            "No value for the base power" 
        self.BaseUnitPower = \
            base_unit_power
    
    def SetPercentageLosses(self, \
        percentage_losses=None):
        assert percentage_losses is not None, \
            "No value for the percentage of losses" 
        self.PercentageLosses = \
            percentage_losses
    
    def SetLossesFlag(self, \
        losses_flag=None):
        assert losses_flag is not None, \
            "No value for the flag to consider transmission losses" 
        self.LossesFlag = \
            losses_flag

    def SetFlagProblem(self, \
        flag_problem=None):
        assert flag_problem is not None, \
            "No value for the flag that indicates the problem to be \
                solved"
        self.FlagProblem = \
            flag_problem


    def SetPWConvGen(self, \
        PW_conv_gen=None):
        assert PW_conv_gen is not None, \
            "No value for the number of pieces of the piecewise linearisation\
                of the generation cost for conventional generators"
        self.PWConvGen = \
            PW_conv_gen
    
    def SetMinConvGen(self, \
        min_conv_gen=None):
        assert min_conv_gen is not None, \
            "No value for the minimum limit of power generation for \
                conventional generators"
        self.MinConvGen = \
            min_conv_gen

    def SetMaxConvGen(self, \
        max_conv_gen=None):
        assert max_conv_gen is not None, \
            "No value for the maximum limit of power generation for \
                conventional generators"
        self.MaxConvGen = \
            max_conv_gen

    def SetACoeffPWConvGen(self, \
        A_coeff_PW_conv_gen=None):
        assert A_coeff_PW_conv_gen is not None, \
            "No value for the coefficient A of the piece Ax + b for \
                conventional generators"
        self.ACoeffPWConvGen = \
            A_coeff_PW_conv_gen
    
    def SetBCoeffPWConvGen(self, \
        B_coeff_PW_conv_gen=None):
        assert B_coeff_PW_conv_gen is not None, \
            "No value for the coefficient b of the piece Ax + b for \
                conventional generators"
        self.BCoeffPWConvGen = \
            B_coeff_PW_conv_gen

    def SetRampConvGen(self, \
        ramp_conv_gen=None):
        assert ramp_conv_gen is not None, \
            "No value for the On/Off ramps for conventional generators"
        self.RampConvGen = \
            ramp_conv_gen
    
    def SetOriginalNumberConvGen(self, \
        original_number_conv_gen=None):
        assert original_number_conv_gen is not None, \
            "No value for the original numeration of conventional generators"
        self.OriginalNumberConvGen = \
            original_number_conv_gen


    def SetPWRESGen(self, \
        PW_RES_gen=None):
        assert PW_RES_gen is not None, \
            "No value for the number of pieces of the piecewise linearisation\
                of the generation cost for RES generators"
        self.PWRESGen = \
            PW_RES_gen
    
    def SetMinRESGen(self, \
        min_RES_gen=None):
        assert min_RES_gen is not None, \
            "No value for the minimum limit of power generation for \
                RES generators"
        self.MinRESGen = \
            min_RES_gen

    def SetMaxRESGen(self, \
        max_RES_gen=None):
        assert max_RES_gen is not None, \
            "No value for the maximum limit of power generation for \
                RES generators"
        self.MaxRESGen = \
            max_RES_gen

    def SetACoeffPWRESGen(self, \
        A_coeff_PW_RES_gen=None):
        assert A_coeff_PW_RES_gen is not None, \
            "No value for the coefficient A of the piece Ax + b for \
                RES generators"
        self.ACoeffPWRESGen = \
            A_coeff_PW_RES_gen
    
    def SetBCoeffPWRESGen(self, \
        B_coeff_PW_RES_gen=None):
        assert B_coeff_PW_RES_gen is not None, \
            "No value for the coefficient b of the piece Ax + b for \
                RES generators"
        self.BCoeffPWRESGen = \
            B_coeff_PW_RES_gen
    
    def SetOriginalNumberRESGen(self, \
        original_number_RES_gen=None):
        assert original_number_RES_gen is not None, \
            "No value for the original numeration of RES generators"
        self.OriginalNumberRESGen = \
            original_number_RES_gen
    
    def SetRESScenarios(self, \
        RES_scenarios=None):
        assert RES_scenarios is not None, \
            "No value for scenarios of generation for RES"
        self.RESScenarios = \
            RES_scenarios
    

    def SetPWHydroGen(self, \
        PW_hydro_gen=None):
        assert PW_hydro_gen is not None, \
            "No value for the number of pieces of the piecewise linearisation\
                of the generation cost for conventional generators"
        self.PWHydroGen = \
            PW_hydro_gen

    def SetMinHydroGen(self, \
        min_hydro_gen=None):
        assert min_hydro_gen is not None, \
            "No value for the minimum limit of power generation for \
                hydro generators"
        self.MinHydroGen = \
            min_hydro_gen

    def SetMaxHydroGen(self, \
        max_hydro_gen=None):
        assert max_hydro_gen is not None, \
            "No value for the maximum limit of power generation for \
                hydro generators"
        self.MaxHydroGen = \
            max_hydro_gen

    def SetACoeffPWHydroGen(self, \
        A_coeff_PW_hydro_gen=None):
        assert A_coeff_PW_hydro_gen is not None, \
            "No value for the coefficient A of the piece Ax + b for \
                hydro generators"
        self.ACoeffPWHydroGen = \
            A_coeff_PW_hydro_gen
    
    def SetBCoeffPWHydroGen(self, \
        B_coeff_PW_hydro_gen=None):
        assert B_coeff_PW_hydro_gen is not None, \
            "No value for the coefficient b of the piece Ax + b for \
                hydro generators"
        self.BCoeffPWHydroGen = \
            B_coeff_PW_hydro_gen

    def SetRampHydroGen(self, \
        ramp_hydro_gen=None):
        assert ramp_hydro_gen is not None, \
            "No value for the On/Off ramps for hydro generators"
        self.RampHydroGen = \
            ramp_hydro_gen
    
    def SetOriginalNumberHydroGen(self, \
        original_number_hydro_gen=None):
        assert original_number_hydro_gen is not None, \
            "No value for the original numeration of hydro generators"
        self.OriginalNumberHydroGen = \
            original_number_hydro_gen


    def SetMaxPowerPumps(self, \
        max_power_pumps=None):
        assert max_power_pumps is not None, \
            "No value for the maximum power capacity of pumps"
        self.MaxPowerPumps = \
            max_power_pumps
    
    def SetCostOperPumps(self, \
        cost_oper_pumps=None):
        assert cost_oper_pumps is not None, \
            "No value for the operational cost of pumps"
        self.CostOperPumps = \
            cost_oper_pumps


    def SetEffStorage(self, \
        eff_storage=None):
        assert eff_storage is not None, \
            "No value for the efficiency of storage"
        self.EffStorage = \
            eff_storage
    
    def SetTotalHoursPerPeriod(self, \
        total_hours_per_period=None):
        assert total_hours_per_period is not None, \
            "No value for the number of hours per sub-period in \
                a 24-hour period"
        self.TotalHoursPerPeriod = \
            total_hours_per_period

    def SetMultScenariosDemand(self, \
        mult_scenarios_demand=None):
        assert mult_scenarios_demand is not None, \
            "No value for the Multiplier to adjust the demand \
                on each node for each temporal representative \
                day and for each sub-period in the 24h period"
        self.MultScenariosDemand = \
            mult_scenarios_demand


    def SetActiveBranches(self, \
        active_branches=None):
        assert active_branches is not None, \
            "No value for the Flag that indicates if the \
                transmission line or transformer is active \
                on each contingency"
        self.ActiveBranches = \
            active_branches
    
    def SetPowerRateLimitTL(self, \
        power_rate_limit_TL=None):
        assert power_rate_limit_TL is not None, \
            "No value for the Thermal \
                limit of power transmission lines and \
                transformers"
        self.PowerRateLimitTL = \
            power_rate_limit_TL
    
    def SetOriginalNumberBranchFrom(self, \
        original_number_branch_from=None):
        assert original_number_branch_from is not None, \
            "No value for the Original numeration of \
                the transmission lines and transformers \
                in the power system in the from end"
        self.OriginalNumberBranchFrom = \
            original_number_branch_from

    def SetOriginalNumberBranchTo(self, \
        original_number_branch_to=None):
        assert original_number_branch_to is not None, \
            "No value for the Original numeration of \
                the transmission lines and transformers \
                in the power system in the to end"
        self.OriginalNumberBranchTo = \
            original_number_branch_to

    def SetPosNumberBranchFrom(self, \
        pos_number_branch_from=None):
        assert pos_number_branch_from is not None, \
            "No value for the Position of the from end of \
                the transmission lines and transformers \
                in the vector that stores the node data. \
                The position start from zero in the node \
                data"
        self.PosNumberBranchFrom = \
            pos_number_branch_from
    
    def SetPosNumberBranchTo(self, \
        pos_number_branch_to=None):
        assert pos_number_branch_to is not None, \
            "No value for the Position of the to end of \
                the transmission lines and transformers \
                in the vector that stores the node data. \
                The position start from zero in the node \
                data"
        self.PosNumberBranchTo = \
            pos_number_branch_to

    def SetReactanceBranch(self, \
        reactance_branch=None):
        assert reactance_branch is not None, \
            "No value for the Reactance of the transmission \
                lines and transformers"
        self.ReactanceBranch = \
            reactance_branch

    def SetResistanceBranch(self, \
        resistance_branch=None):
        assert resistance_branch is not None, \
            "No value for the resistance of the transmission \
                lines and transformers"
        self.ResistanceBranch = \
            resistance_branch
    
    def SetACoeffPWBranchLosses(self, \
        A_coeff_PW_branch_losses=None):
        assert A_coeff_PW_branch_losses is not None, \
            "No value for the Coefficient A of the \
                piece Ax + b for the piecewise \
                linearisation of the nonlinear branch \
                Losses"
        self.ACoeffPWBranchLosses = \
            A_coeff_PW_branch_losses
    
    def SetBCoeffPWBranchLosses(self, \
        B_coeff_PW_branch_losses=None):
        assert B_coeff_PW_branch_losses is not None, \
            "No value for the Coefficient b of the \
                piece Ax + b for the piecewise \
                linearisation of the nonlinear branch \
                Losses"
        self.BCoeffPWBranchLosses = \
            B_coeff_PW_branch_losses


    def SetPowerDemandNode(self, \
        power_demand_node=None):
        assert power_demand_node is not None, \
            "No value for the Active Power demand at each node"
        self.PowerDemandNode = \
            power_demand_node
    
    def SetTypeNode(self, \
        type_node=None):
        assert type_node is not None, \
            "No value for the type of node"
        self.TypeNode = \
            type_node
    
    def SetOriginalNumberNodes(self, \
        original_number_nodes=None):
        assert original_number_nodes is not None, \
            "No value for the type of node"
        self.OriginalNumberNodes = \
            original_number_nodes

    # Data outputs of Energy model

    def _get_initial_tree_node(self, position_tree=None):
        ''' This function retrieves the position of the node in the energy tree that the user 
            requires
        
            Parameters
            ----------
            Mandatory:\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree. If the value does not vary with
                                the energy tree then this value should be left in None
        '''
        number_node = None
        initial = True
        for value in position_tree.values():
            if initial:
                for node in self.tree.nodes(data=True):
                    if node['obj'].name_node == value:
                        number_node = node['obj'].node
                        break
                initial = False
            else:
                for node in self.tree.successors(number_node):
                    if node['obj'].name_node == value:
                        number_node = node['obj'].node
                        break
        return number_node
    
    def _calculate_value_tree(self, number_node=None, ID=None, name=None, position_tree=None, hour=None, typ=None, flag=False):
        ''' This function retrieves the value of variables or parameters based on a given position 
            of the energy tree
        
            Parameters
            ----------
            Mandatory:\\
            number_node     :   Number of the node in the energy tree that corresponds to the 
                                specific position that the user wants to retreive the information\\
            ID              :   Unique ID of the network element\\
            name            :   Name of variable or parameter to be retrieved\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree. If the value does not vary with
                                the energy tree then this value should be left in None \\
            hour            :   integer that indicates the specific hour of the requested data. If
                                the data does not change in time then this input must be left in
                                None\\
            typ             :   This refers to the type of element to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted\\
            flag            :   Boolean variable indicating if the recursive function is pointing to 
                                the original node or one of its succesors
        '''
        value_node = 0
        original_number_node = -1
        inner_flag = False
        if not flag:
            original_number_node = number_node
        for node in self.tree.successors(number_node):
            flag = True
            inner_flag = True
            position_tree[str(node['obj'].level)] = node['obj'].name_node
            value_node += self._calculate_value_tree(node['obj'].node, ID, position_tree, hour, typ, flag)
            del position_tree[str(node['obj'].level)]
        
        if not inner_flag:
            for node in self.network.nodes(data=True):
                if node['obj'].ID == ID and typ == "parameter":
                    for parameter in node['obj'].parameters:
                        if parameter.name == name and (not hour or hour == parameter.hour) and position_tree == parameter.position_tree:
                            value_node = parameter.value
                if node['obj'].ID == ID and typ == "variable":
                    for variable in node['obj'].variables:
                        if variable.name == name and (not hour or hour == variable.hour) and position_tree == variable.position_tree:
                            value_node = variable.value
        
        if flag and original_number_node != number_node:
            for parameter in self.tree[number_node]['obj'].parameters:
                if parameter.name == "weight":
                    value_node = value_node * parameter.value
                    break

        return value_node

    def get_value_network(self, ID=None, name=None, position_tree=None, hour=None, typ=None):
        ''' This function retrieves the values of variables and parameters

            Parameters
            ----------
            Mandatory:\\
            ID              :   Unique ID of the network element\\
            name            :   Name of variable or parameter to be retrieved\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree. If the value does not vary with
                                the energy tree then this value should be left in None\\
            hour            :   integer that indicates the specific hour of the requested data. If
                                the data does not change in time then this input must be left in
                                None\\
            typ             :   This refers to the type of element to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted
        '''
        if not position_tree:
            for node in self.network.nodes(data=True):
                if node[1]['obj'].ID == ID and typ == "parameter":
                    for parameter in node[1]['obj'].parameters:
                        if parameter.name == name and (not hour or hour == parameter.hour):
                            return parameter.value
                if node[1]['obj'].ID == ID and typ == "variable":
                    for variable in node[1]['obj'].variables:
                        if variable.name == name and (not hour or hour == variable.hour):
                            return variable.value
        else:
            number_node = self._get_initial_tree_node(position_tree)
            return self._calculate_value_tree(number_node, ID, position_tree, hour, typ, False)

        return None

    def _get_all_hours(self, ID=None, name=None, type_data=None):
        ''' This function return a list with all the hours in which a parameter or variable
            changes

            Parameters
            ----------
            Mandatory:\\
            IDs             :   List of unique IDs of network elements\\
            name            :   Name of variable or parameter to be retrieved\\
            type_data       :   This refers to the type of data to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted
        '''
        hours = []
        for node in self.network.nodes(data=True):
            if node.ID == ID and type_data == "parameter":
                for parameter in node.parameters:
                    if parameter.name == name and parameter.hour is not None and parameter.hour not in hours:
                        hours.append(parameter.hour)
            elif node.ID == ID and type_data == "variable":
                for variable in node.variables:
                    if variable.name == name and variable.hour is not None and variable.hour not in hours:
                        hours.append(variable.hour)
        hours.sort()
        return hours

    def _get_all_IDs(self, typ=None, subtype=None):
        ''' This function return a list with all IDs that correspond to the indicated type and subtype

            Parameters
            ----------
            Mandatory:\\
            typ            :   Type of element to be retrieved, e.g. branch, nodes\\
            subtype         :   Subtype of element to be retrieved, e.g. thermal, hydro
        '''
        IDs = []
        for node in self.network.nodes(data=True):
            if (subtype == "all" and node.typ == typ) or (subtype == node.subtype and node.typ == typ):
                IDs.append(node.ID)
            else:
                raise ValueError('Incorrect values for this function, check description')
        return IDs

    def get_values_network(self, **kwargs):
        ''' This function retrieves the values of variables and parameters

            Description
            ----------
            * If list of IDs is provided then type and subtype are ignored and a list of type_data
            with "name" is returned\\
            * If IDs is 'all' and type and subtype are provided then a list of type_data
            with "name" for all elements of the same type and subtype is returned\\
            * If IDs and suttype are 'all' and only type is provided then a list of type_data
            with "name" for all elements of the same type and all subtypes is returned\\
            * If hour is 'all' then a list of all hours is returned\\
            * If hour is a list then a list of those hours is returned\\
            * If hour is an integer then a list of an specific hour is returned\\
            * If position_tree is provided then a list for that specific position in the tree 
            is provided\\

            Parameters
            ----------
            Mandatory:\\
            IDs             :   List of unique IDs of network elements\\
            typ            :   Type of element to be retrieved, e.g. branch, nodes\\
            subtype         :   Subtype of element to be retrieved, e.g. thermal, hydro\\
            name            :   Name of variable or parameter to be retrieved\\
            position_tree   :   Dictionary containing information of the location of the information
                                in relation with the energy tree. If the value does not vary with
                                the energy tree then this value should be left in None\\
            hours           :   integer that indicates the specific hour of the requested data. If
                                the data does not change in time then this input must be left in
                                None\\
            type_data       :   This refers to the type of data to be retrieved. This value
                                can be either "variable" or "parameter". Other values will not
                                be accepted
            
        '''
        IDs             = kwargs.pop('IDs', None)
        typ             = kwargs.pop('typ', None)
        subtype         = kwargs.pop('subtype', None)
        name            = kwargs.pop('name', None)
        position_tree   = kwargs.pop('position_tree', None)
        hours            = kwargs.pop('hours', None)
        type_data       = kwargs.pop('type_data', None)
        
        elements = []
        if isinstance(IDs, list) and isinstance(hours, list):
            for id in IDs:
                for hour in hours:
                    elements.append(self.get_value_network(id, name, position_tree, hour, type_data))
        elif isinstance(IDs, list) and hours == "all":
            for id in IDs:
                hours = self._get_all_hours(self, name, type_data)
                for hour in hours:
                    elements.append(self.get_value_network(id, name, position_tree, hour, type_data))
        elif isinstance(IDs, list) and not isinstance(hours, list) and hours is not None:
            for id in IDs:
                elements.append(self.get_value_network(id, name, position_tree, hours, type_data))
        elif IDs == "all" and isinstance(hours, list):
            IDs = self._get_all_IDs(typ=typ, subtype=subtype)
            for id in IDs:
                for hour in hours:
                    elements.append(self.get_value_network(id, name, position_tree, hour, type_data))
        elif IDs == "all" and hours == "all":
            IDs = self._get_all_IDs(typ=typ, subtype=subtype)
            for id in IDs:
                hours = self._get_all_hours(self, name, type_data)
                for hour in hours:
                    elements.append(self.get_value_network(id, name, position_tree, hour, type_data))
        elif IDs == "all" and not isinstance(hours, list) and hours is not None:
            IDs = self._get_all_IDs(typ=typ, subtype=subtype)
            for id in IDs:
                elements.append(self.get_value_network(id, name, position_tree, hours, type_data))
        else:
            raise ValueError('Incorrect values for this function, check description')

        return np.array(elements)



    def GetThermalGeneration(self):
        if self.NumberConvGen > 0:
            if self.solver_problem == "GLPK":
                ThermalGenerationSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                            self.NumberConvGen))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberConvGen):
                            ThermalGenerationSolution[i, j, k] = \
                                self.solver.get_col_prim(\
                                    str(self.thermalgenerators[i, j][0]), k) * \
                                        self.BaseUnitPower
                return ThermalGenerationSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.ThermalGenerationSolution
        else:
            return None

    def GetRESGeneration(self):
        if self.NumberRESGen > 0:
            if self.solver_problem == "GLPK":
                RESGenerationSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                            self.NumberRESGen))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberRESGen):
                            RESGenerationSolution[i, j, k] = \
                                self.solver.get_col_prim(\
                                    str(self.RESgenerators[i, j][0]), k) * \
                                        self.BaseUnitPower
                return RESGenerationSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.RESGenerationSolution
        else:
            return None

    def GetHydroGeneration(self):
        if self.NumberHydroGen > 0:
            if self.solver_problem == "GLPK":
                HydroGenerationSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                            self.NumberHydroGen))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberHydroGen):
                            HydroGenerationSolution[i, j, k] = \
                                self.solver.get_col_prim(\
                                    str(self.Hydrogenerators[i, j][0]), k) * \
                                        self.BaseUnitPower
                return HydroGenerationSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.HydroGenerationSolution
        else:
            return None
    
    def GetPumpOperation(self):
        if self.NumberPumps > 0:
            pumpsvarSolution = \
                np.empty((len(self.LongTemporalConnections),\
                    self.ShortTemporalConnections, \
                        self.NumberPumps))
            for i in self.LongTemporalConnections:
                for j in range(self.ShortTemporalConnections):
                    for k in range(self.NumberPumps):
                        pumpsvarSolution[i, j, k] = \
                            self.solver.get_col_prim(\
                                str(self.pumpsvar[i, j][0]), k) * \
                                    self.BaseUnitPower
            return pumpsvarSolution
        else:
            return None

    def GetThermalGenerationCost(self):
        if self.NumberConvGen > 0:
            if self.solver_problem == "GLPK":
                ThermalGenerationCostSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                            self.NumberConvGen))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberConvGen):
                            ThermalGenerationCostSolution[i, j, k] = \
                                self.solver.get_col_prim(\
                                    str(self.thermalCG[i, j][0]), k)
                return ThermalGenerationCostSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.ThermalGenerationCostSolution
        else:
            return None
    
    def GetRESGenerationCost(self):
        if self.NumberRESGen > 0:
            if self.solver_problem == "GLPK":
                RESGenerationCostSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                            self.NumberRESGen))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberRESGen):
                            RESGenerationCostSolution[i, j, k] = \
                                self.solver.get_col_prim(\
                                    str(self.RESCG[i, j][0]), k)
                return RESGenerationCostSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.RESGenerationCostSolution
        else:
            return None
    
    def GetHydroGenerationCost(self):
        if self.NumberHydroGen > 0:
            if self.solver_problem == "GLPK":
                HydroGenerationCostSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                            self.NumberHydroGen))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberHydroGen):
                            HydroGenerationCostSolution[i, j, k] = \
                                self.solver.get_col_prim(\
                                    str(self.HydroCG[i, j][0]), k)
                return HydroGenerationCostSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.HydroGenerationCostSolution
        else:
            return None
    
    def GetVoltageAngle(self):
        if self.FlagProblem:
            if self.solver_problem == "GLPK":
                VoltageAngleSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                        (self.NumberContingencies + 1), \
                        self.NumberNodesPS))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberContingencies + 1):
                            for ii in range(self.NumberNodesPS):
                                VoltageAngleSolution[i, j, k, ii] = \
                                    self.solver.get_col_prim(\
                                    str(self.VoltageAngle[i, j, k][0]), ii)
                return VoltageAngleSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.VoltageAngleSolution
        else:
            return None
    
    def GetLoadCurtailmentNodes(self):
        if self.FlagProblem and self.FlagFeasibility:
            if self.solver_problem == "GLPK":
                LoadCurtailmentNodesSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                        (self.NumberContingencies + 1), \
                        self.NumberNodesPS))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberContingencies + 1):
                            for ii in range(self.NumberNodesPS):
                                LoadCurtailmentNodesSolution[i, j, k, ii] = \
                                    self.solver.get_col_prim(\
                                    str(self.LoadCurtailmentNode[i, j, k][0]), ii)\
                                        * self.BaseUnitPower
                return LoadCurtailmentNodesSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.LoadCurtailmentNodesSolution
        else:
            return None

    def GetGenerationCurtailmentNodes(self):
        if self.FlagProblem and self.FlagFeasibility:
            if self.solver_problem == "GLPK":
                GenerationCurtailmentNodesSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                        (self.NumberContingencies + 1), \
                        self.NumberNodesPS))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberContingencies + 1):
                            for ii in range(self.NumberNodesPS):
                                GenerationCurtailmentNodesSolution\
                                    [i, j, k, ii] = \
                                    self.solver.get_col_prim(\
                                    str(self.GenerationCurtailmentNode\
                                        [i, j, k][0]), ii)\
                                        * self.BaseUnitPower
                return GenerationCurtailmentNodesSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.GenerationCurtailmentNodesSolution
        else:
            return None

    def GetActivePowerFlow(self):
        if self.FlagProblem:
            if self.solver_problem == "GLPK":
                ActivePowerFlowSolution = \
                    np.empty((len(self.LongTemporalConnections),\
                        self.ShortTemporalConnections, \
                        (self.NumberContingencies + 1), \
                        self.NumberLinesPS))
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        for k in range(self.NumberContingencies + 1):
                            for ii in range(self.NumberLinesPS):
                                ActivePowerFlowSolution[i, j, k, ii] = \
                                    self.solver.get_col_prim(\
                                    str(self.ActivePowerFlow[i, j, k][0]), ii)\
                                        * self.BaseUnitPower
                return ActivePowerFlowSolution
            elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
                return self.ActivePowerFlowSolution
        else:
            return None
    
    def GetActivePowerLosses(self):
        if self.FlagProblem and self.LossesFlag:
            ActivePowerLossesSolution = \
                np.empty((len(self.LongTemporalConnections),\
                    self.ShortTemporalConnections, \
                    (self.NumberContingencies + 1), \
                    self.NumberLinesPS))
            for i in self.LongTemporalConnections:
                for j in range(self.ShortTemporalConnections):
                    for k in range(self.NumberContingencies + 1):
                        for ii in range(self.NumberLinesPS):
                            ActivePowerLossesSolution[i, j, k, ii] = \
                                self.solver.get_col_prim(\
                                str(self.ActivePowerLosses[i, j, k][0]), ii)\
                                    * self.BaseUnitPower
            return ActivePowerLossesSolution
        else:
            return None

    def GetLoadCurtailmentSystemED(self):
        if not self.FlagProblem and self.FlagFeasibility:
            LoadCurtailmentSystemEDSolution = \
                np.empty((len(self.LongTemporalConnections),\
                    self.ShortTemporalConnections))
            for i in self.LongTemporalConnections:
                for j in range(self.ShortTemporalConnections):
                    LoadCurtailmentSystemEDSolution[i, j] = \
                        self.solver.get_col_prim(\
                        str(self.loadcurtailmentsystem[i, j][0]), 0) * \
                        self.BaseUnitPower
            return LoadCurtailmentSystemEDSolution
        else:
            return None
    
    def GetGenerationCurtailmentSystemED(self):
        if not self.FlagProblem and self.FlagFeasibility:
            GenerationCurtailmentSystemEDSolution = \
                np.empty((len(self.LongTemporalConnections),\
                    self.ShortTemporalConnections))
            for i in self.LongTemporalConnections:
                for j in range(self.ShortTemporalConnections):
                    GenerationCurtailmentSystemEDSolution[i, j] = \
                        self.solver.get_col_prim(\
                        str(self.generationcurtailmentsystem[i, j][0]), 0) * \
                        self.BaseUnitPower
            return GenerationCurtailmentSystemEDSolution
        else:
            return None
    
    def GetObjectiveFunctionNM(self):
        if self.solver_problem == "GLPK":
            return self.solver.get_obj_val()
        elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
            return self.network_model.get_objective_function_cpp()


class EnergyandNetwork(Energymodel, Networkmodel):
    """ This class builds and solve the energy and network models(NM) 
    using the gplk wrapper.

    The information of the pyeneClass is passed to this class,
    which provides the parameters for the model. Furthermore,
    the GLPKSolver class that contains the GLPK wrapper is imported """

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
        self.ConnectionTreeGen = obj3.p['pyeneE']   # Connections
                        # between the energy model and the network
                        # model. This parameters connects the inputs 
                        # of each tree with the outputs of its 
                        # related hydro generator
        self.PenaltyCurtailment = obj3.Penalty # Penalty for
                        # load curtailment in the power system

        # Storing the data of other objects
        Energymodel.__init__(self, obj1)
        Networkmodel.__init__(self, obj2)


    def optimisationENM(self, solver_name=None):
        """ This class method solve the optimisation problem """
        if solver_name == "GLPK":
            # Creation of model instance
            self.solver = GLPKSolver(message_level='all', \
                simplex_method='dualprimal')
            self.solver_problem = "GLPK"
            # Definition of minimisation problem
            self.solver.set_dir('min')
            # Definition of the mathematical formulation
            self.EnergyandNetworkModels()
            ret = self.solver.simplex()
            assert ret == 0, "GLPK could not solve the problem"
        elif solver_name == "CLP":
            self.solver_problem = "CLP"
            self.Energy_OPF_R1CPP()
        elif solver_name == "CLP-I":
            self.solver_problem = "CLP-I"
            self.Energy_OPF_R1CPP()
        elif solver_name == "CLP-IR":
            self.solver_problem = "CLP-IR"
            self.Energy_OPF_R1CPP()
        else:
            print("incorrect solver has been selected")

    def EnergyandNetworkModels(self):
        """ This class method builds the optimisation model
        for the energy and network related problems """
        self.variablesEM()

        if self.FlagProblem:
            self.variablesOPF()
        else:
            self.variablesED()

        self.coeffmatrixENM()

        self.Objective_functionENM()

    def coeffmatrixENM(self):
        """ This class method contains the functions that allow building 
        the coefficient matrix (matrix A) for the simplex method """
        # The coefficient matrix is stored in CSR format (sparse matrix) 
        # to be later added to glpk
        self.ia = [] # Position in rows
        self.ja = [] # Position in columns
        self.ar = [] # Value
        self.ne = 0 # Number of non-zero coefficients in matrix A

        self.Energybalance()
        self.Aggregation()
        # if self.NumberNodesUnc != 0:
        #     self.AggregationStochastic()

        if self.FlagProblem:
            self.activepowerbalancepernode()
            self.activepowerflowconstraints()
            if self.LossesFlag:
                self.activepowerlosses1constraints()
                self.activepowerlosses2constraints()
        else:
            self.activepowerbalancesystem()

        if self.NumberHydroGen > 0:
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
            for j in self.LongTemporalConnections:
                self.solver.set_col_bnds(\
                    str(self.OutputsTree[i][0]), self.ConnectionTreeGen[j],\
                        'lower', 0, sys.float_info.max)      

    # Constraints ENM

    def EnergyandNetworkRelation(self):
        """ This class method writes the constraint that links the energy
        model with the network model in glpk.
    
        First, it is reserved space in memory to store the constraints.
        Second, the coefficients of the constraints are introduced
        in the matrix of coefficients (matrix A).
        Third, the bounds of the constraints are defined """
        # Creating the matrices to store the position of constraints in
        # matrix A
        self.connectionNetworkandEnergy = np.empty((self.NumberTrees,\
            len(self.LongTemporalConnections)), dtype=[('napos', 'U80'),\
                ('nupos', 'i4')]) # Start position 
                    # of energy and economic dispatch constraints (rows)

        for i in range(self.NumberTrees):
            for j in self.LongTemporalConnections:
                self.connectionNetworkandEnergy[i, j] = ('CEED'+str(i)+str(j),\
                    self.solver.add_rows('CEED'+str(i)+str(j), 1))  # Number of 
                        # columns (constraints) in matrix A for the 
                        # constraints that links the energy and economic 
                        # dispatch model

        # Generating the matrix A for energy and network constraints
        for i in range(self.NumberTrees): # Vectors is equal to the number
            # of hydro generators (rivers) TODO: Explain this better and 
            # separate the data for this
            for j in self.LongTemporalConnections:
                # Storing the variables for the total storage of the tree
                self.ia.append(self.connectionNetworkandEnergy[i, j][1])
                self.ja.append(self.OutputsTree[i][1] + \
                    self.ConnectionTreeGen[j])
                self.ar.append(1.0)
                self.ne += 1
                for k in range(self.ShortTemporalConnections):
                    self.ia.append(self.connectionNetworkandEnergy[i, j][1])
                    self.ja.append(self.Hydrogenerators[j, k][1] + i)
                    self.ar.append(-self.TotalHoursPerPeriod[k] * \
                        self.BaseUnitPower)
                    self.ne += 1
                # Defining the resources (b) for the constraints
                self.solver.set_row_bnds(\
                    str(self.connectionNetworkandEnergy[i, j][0]), 0,\
                    'fixed', 0, 0)
        
    # Objective function ENM

    def Objective_functionENM(self):
        """ This class method defines the objective function of the network and
        energy model in glpk """

        # Calculating the aggregated weights for the last nodes in the tree
        # TODO: explain the aggregated weights better
        
        WghtAgg = 0 + self.WeightNodes
        xp = 0
        for xn in range(self.TreeNodes):
            aux = self.LLNodesAfter[xn][0]
            if aux != 0:
                for xb in range(aux, self.LLNodesAfter[xn][1] + 1):
                    WghtAgg[xb] *= WghtAgg[xn]
            else:
                self.OFaux[xp] = WghtAgg[xn]
                xp += 1

        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
            # Cost for conventional generation    
                if self.NumberConvGen > 0: 
                    for k in range(self.NumberConvGen):
                        self.solver.set_obj_coef(\
                            str(self.thermalCG[i, j][0]),\
                            k, self.OFaux[i] * self.TotalHoursPerPeriod[j])
            # Cost for RES generation    
                if self.NumberRESGen > 0: 
                    for k in range(self.NumberRESGen):
                        self.solver.set_obj_coef(\
                            str(self.RESCG[i, j][0]),\
                            k, self.OFaux[i] * self.TotalHoursPerPeriod[j])
            # Cost for Hydroelectric generation    
                if self.NumberHydroGen > 0: 
                    for k in range(self.NumberHydroGen):
                        self.solver.set_obj_coef(\
                            str(self.HydroCG[i, j][0]),\
                            k, self.OFaux[i] * self.TotalHoursPerPeriod[j])
            # Operation cost of pumps
                if self.NumberPumps > 0:
                    for k in range(self.NumberPumps):
                        self.solver.set_obj_coef(\
                            str(self.pumpsvar[i, j][0]),\
                            k, -self.OFaux[i] * self.TotalHoursPerPeriod[j] \
                                * self.BaseUnitPower \
                                    * self.CostOperPumps[k])
            # Punitive cost for load curtailment
                if self.FlagProblem and self.FlagFeasibility:
                # Optimal Power Flow
                    for k in range(self.NumberContingencies + 1):
                        for ii in range(self.NumberNodesPS):
                            self.solver.set_obj_coef(\
                                str(self.LoadCurtailmentNode[i, j, k][0]),\
                                ii, self.OFaux[i] * self.TotalHoursPerPeriod[j] \
                                    * self.PenaltyCurtailment)
                            self.solver.set_obj_coef(\
                                str(self.GenerationCurtailmentNode\
                                [i, j, k][0]), ii, \
                                self.OFaux[i] * self.TotalHoursPerPeriod[j] * \
                                self.PenaltyCurtailment)
                elif not self.FlagProblem and self.FlagFeasibility:
                # Economic Dispatch
                # TODO: Set a parameter penalty in pyeneN
                    self.solver.set_obj_coef(\
                        str(self.loadcurtailmentsystem[i, j][0]),\
                        0, self.OFaux[i] * self.TotalHoursPerPeriod[j] \
                            * self.PenaltyCurtailment)
    
    ####################################
    ###   ENERGY AND OPF CPP MODELS  ###
    ####################################

    def Energy_OPF_R1CPP(self):
        """ This class method builds the optimisation model
        for the optimal power flow problem using a fast implementation of 
        different mathematical models in c++ """

        self.combined_energy_dc_opf_r1 = models_cpp()
        self.set_parameters_cpp_combined_energy_dc_opf_r1_models()
        if self.solver_problem == "CLP":
            self.combined_energy_dc_opf_r1.run_combined_energy_dc_opf_r1_cpp()
        elif self.solver_problem == "CLP-I":
            self.combined_energy_dc_opf_r1.run_iterative_combined_energy_dc_opf_cpp()
        elif self.solver_problem == "CLP-IR":
            self.combined_energy_dc_opf_r1.run_iterative_combined_energy_dc_opf_v2_cpp()
        

        # Retrieving solution
        aux_par, aux_tot, aux_in, aux_out = \
            self.combined_energy_dc_opf_r1.get_energy_tree_solution_cpp()

        self.PartialStorageSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        self.TotalStorageSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        self.InputsTreeSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        self.OutputsTreeSolution = \
            np.empty((self.NumberTrees, self.TreeNodes))
        counter = 0
        for i in range(self.NumberTrees):
            for j in range(self.TreeNodes):
                self.PartialStorageSolution[i, j] = aux_par[counter]
                self.TotalStorageSolution[i, j] = aux_tot[counter]
                self.InputsTreeSolution[i, j] = aux_in[counter]
                self.OutputsTreeSolution[i, j] = aux_out[counter]
                counter += 1
        
        # retrieving solution
        aux_gen, aux_gen_cost = \
            self.combined_energy_dc_opf_r1.get_generation_solution_cpp()
        
        aux_flows = self.combined_energy_dc_opf_r1.get_branch_solution_cpp()

        aux_angles, aux_load_cur, aux_gen_cur = self.combined_energy_dc_opf_r1.get_node_solution_cpp()
        
        self.ThermalGenerationSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberConvGen))
        self.RESGenerationSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberRESGen))
        self.HydroGenerationSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberHydroGen))
        
        self.GenerationCurtailmentNodesSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberNodesPS))
        
        self.ThermalGenerationCostSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberConvGen))
        self.RESGenerationCostSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberRESGen))
        self.HydroGenerationCostSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                    self.NumberHydroGen))
        
        self.VoltageAngleSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberNodesPS))
        
        self.LoadCurtailmentNodesSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberNodesPS))
        
        self.ActivePowerFlowSolution = \
            np.empty((len(self.LongTemporalConnections),\
                self.ShortTemporalConnections, \
                (self.NumberContingencies + 1), \
                self.NumberLinesPS))

        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        if self.ActiveConv[k]:
                            self.ThermalGenerationSolution[i, j, k] = \
                                aux_gen[counter]
                            counter += 1
                        else:
                            self.ThermalGenerationSolution[i, j, k] = 0.0
                    
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        self.HydroGenerationSolution[i, j, k] = \
                            aux_gen[counter]
                        counter += 1
                
                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        self.RESGenerationSolution[i, j, k] = \
                            aux_gen[counter]
                        counter += 1
        
        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                if self.NumberConvGen > 0:
                    for k in range(self.NumberConvGen):
                        if self.ActiveConv[k]:
                            self.ThermalGenerationCostSolution[i, j, k] = \
                                aux_gen_cost[counter]
                            counter += 1
                        else:
                            self.ThermalGenerationCostSolution[i, j, k] = 0.0
                    
                if self.NumberHydroGen > 0:
                    for k in range(self.NumberHydroGen):
                        self.HydroGenerationCostSolution[i, j, k] = \
                            aux_gen_cost[counter]
                        counter += 1
                
                if self.NumberRESGen > 0:
                    for k in range(self.NumberRESGen):
                        self.RESGenerationCostSolution[i, j, k] = \
                            aux_gen_cost[counter]
                        counter += 1
        
        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberLinesPS):
                        if self.ActiveBranches[k, ii]:
                            self.ActivePowerFlowSolution[i, j, k ,ii] = \
                                aux_flows[counter]
                            counter += 1
                        else:
                            self.ActivePowerFlowSolution[i, j, k ,ii] =  0.0
        
        counter = 0
        for i in self.LongTemporalConnections:
            for j in range(self.ShortTemporalConnections):
                for k in range(self.NumberContingencies + 1):
                    for ii in range(self.NumberNodesPS):
                        if self.TypeNode[ii] != 4:
                            self.VoltageAngleSolution[i, j, k ,ii] = \
                                aux_angles[counter]
                            self.LoadCurtailmentNodesSolution[i, j, k ,ii] = \
                                aux_load_cur[counter]
                            self.GenerationCurtailmentNodesSolution[i, j, k, ii] = \
                                aux_gen_cur[counter]
                            counter += 1
                        else:
                            self.VoltageAngleSolution[i, j, k ,ii] =  0.0
                            self.LoadCurtailmentNodesSolution[i, j, k ,ii] =  0.0
                            self.GenerationCurtailmentNodesSolution[i, j, k, ii] = 0.0
    
    def set_parameters_cpp_combined_energy_dc_opf_r1_models(self):
        """ This class method set all parameters in the c++ implementation """
        # Information nodes
        aux_intake = []
        for vectors in range(self.NumberTrees):
            for nodes in range(self.TreeNodes):
                aux_intake.append(self.IntakeTree[nodes, vectors])
        aux_output = []
        for vectors in range(self.NumberTrees):
            for nodes in range(self.TreeNodes):
                aux_output.append(self.OutputTree[nodes, vectors])

        self.combined_energy_dc_opf_r1.load_energy_tree_information_cpp(\
            self.TreeNodes, self.NumberTrees, self.LLEB, self.LLEA, aux_intake, \
            aux_output, self.WeightNodes)

        # Information nodes
        slack = self.OriginalNumberNodes[0]
        for ii in range(self.NumberNodesPS):
            if self.TypeNode[ii] != 4:
                if self.TypeNode[ii] == 3:
                    slack = self.OriginalNumberNodes[ii]
                demand_vals = []
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        if self.NumberDemScenarios == 0:
                            demand_vals.append(self.PowerDemandNode[ii] * \
                                self.MultScenariosDemand[i, ii])
                        else:
                            demand_vals.append(self.PowerDemandNode[ii] * \
                                self.MultScenariosDemand[i, j, ii])
                self.combined_energy_dc_opf_r1.add_bus_cpp(demand_vals, \
                    self.OriginalNumberNodes[ii], 'ac')

        # Information transmission lines
        # TODO: implement the same for transformers
        for xtl in range(self.NumberLinesPS):
            if self.ActiveBranches[0, xtl]:
                self.combined_energy_dc_opf_r1.add_branch_cpp([self.ReactanceBranch[xtl]], \
                    [self.ResistanceBranch[xtl]], [self.PowerRateLimitTL[xtl]], \
                    self.OriginalNumberBranchFrom[xtl], \
                    self.OriginalNumberBranchTo[xtl], xtl, 'ac_transmission_line')
        
        # Information for generators        
        counter_gen = 0
        if self.NumberConvGen > 0:
            for xgen in range(self.NumberConvGen):
                if self.ActiveConv[xgen]:
                    P_max = []
                    P_min = []
                    for i in self.LongTemporalConnections:
                        for j in range(self.ShortTemporalConnections):
                            P_max.append(self.MaxConvGen[xgen])
                            P_min.append(self.MinConvGen[xgen])
                    self.combined_energy_dc_opf_r1.add_generator_cpp(P_max, P_min, \
                        self.OriginalNumberConvGen[xgen], counter_gen, 'conv', 0.0, \
                        0.0, self.ACoeffPWConvGen[xgen,:], \
                        self.BCoeffPWConvGen[xgen,:], self.ActiveConv[xgen])
                    counter_gen += 1
        if self.NumberHydroGen > 0:
            for xgen in range(self.NumberHydroGen):
                P_max = []
                P_min = []
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        P_max.append(self.MaxHydroGen[xgen])
                        P_min.append(self.MinHydroGen[xgen])
                self.combined_energy_dc_opf_r1.add_generator_cpp(P_max, P_min, \
                    self.OriginalNumberHydroGen[xgen], counter_gen, 'hydro', 0.0,
                    0.0, self.ACoeffPWHydroGen[xgen,:], \
                    self.BCoeffPWHydroGen[xgen,:], True)
                counter_gen += 1
        if self.NumberRESGen > 0:
            for xgen in range(self.NumberRESGen):
                P_max = []
                P_min = []
                for i in self.LongTemporalConnections:
                    for j in range(self.ShortTemporalConnections):
                        P_max.append(self.MaxRESGen[xgen]* \
                            self.RESScenarios[i, j, xgen])
                        P_min.append(self.MinRESGen[xgen])
                self.combined_energy_dc_opf_r1.add_generator_cpp(P_max, P_min, \
                    self.OriginalNumberRESGen[xgen], counter_gen, 'RES', 0.0,
                    0.0, self.ACoeffPWRESGen[xgen,:], \
                    self.BCoeffPWRESGen[xgen,:], True)
                counter_gen += 1
        
        self.combined_energy_dc_opf_r1.set_integer_data_power_system_cpp("number periods",\
            self.ShortTemporalConnections)
        self.combined_energy_dc_opf_r1.set_integer_data_power_system_cpp(\
            "number representative days", len(self.LongTemporalConnections))
        self.combined_energy_dc_opf_r1.set_integer_data_power_system_cpp(\
            "slack bus", slack)
        
        self.combined_energy_dc_opf_r1.set_continuous_data_power_system_cpp(\
            "total hours period", self.TotalHoursPerPeriod[0])
        self.combined_energy_dc_opf_r1.set_continuous_data_power_system_cpp(\
            "base power", self.BaseUnitPower)

        self.combined_energy_dc_opf_r1.load_combined_energy_dc_opf_information_cpp(\
            self.LLNodesAfter, self.ConnectionTreeGen)

    def GetObjectiveFunctionENM(self):
        if self.solver_problem == "GLPK":
            return self.solver.get_obj_val()
        elif self.solver_problem == "CLP" or self.solver_problem == "CLP-I" \
                or self.solver_problem == "CLP-IR":
            return self.combined_energy_dc_opf_r1.get_objective_function_combined_energy_dc_opf_r1_cpp()