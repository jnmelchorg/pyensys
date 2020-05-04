"""
Created on Mon April 06 2020

This python file containts the classes and methods for the analysis and
modifications of the topology and electrical characteristics of power system.
Furthermore, tools to build the temporal tree are provided

@author: Dr. Jose Nicolas Melchor Gutierrez
"""

import networkx as nx
import logging
from .pyene_Parameters import ElectricityNetwork, Bus

class PowerSystemIslandsIsolations(ElectricityNetwork):
    ''' This class contains all necessary methods to find all islands and 
    isolated nodes in a network '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__data = {
            'NoIsolatedNodes': 0, # Number of isolated nodes in the 
            # power system
            'NoIslands': 0, # Number of islands in the whole power
            # system
            'IsolatedNodes': [], # Isolated nodes in the power system
            'Islands': [] # Islands in the whole power system
        }
        logging.basicConfig(format='%(asctime)s %(message)s', \
            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
    def IslandsIsolations(self):
        ''' This class method calls and controls all methods in this class'''
        auxp = 'Running network analyser - Determining islands and \
            isolated nodes in the power system'
        logging.info(" ".join(auxp.split()))
        self.find_isolates()
        self.find_islands()

    def edges_graph(self, graph=None):
        ''' This class method load all edges in the graph'''
        auxp = "No edge data has been loaded in the \
                class PowerSystemIslandsIsolations"
        assert self.get_objects(obj='transmissionline') != [] or \
            self.get_objects(obj='twowindingtrafo') != [] or \
            self.get_objects(obj='threewindingtrafo') != [], \
            " ".join(auxp.split())
        assert graph.is_multigraph(), "graph is not a graph"
        # Edges for transmission lines
        aux = self.get_objects(obj='transmissionline')
        for edges in range(self.get_no_transmission_lines()):
            graph.add_edge(aux[edges].get_pos_from(), \
                aux[edges].get_pos_to())
        # Edges for two winding transformers
        aux = self.get_objects(obj='twowindingtrafo')
        for edges in range(self.get_no_two_winding_trafos()):
            graph.add_edge(aux[edges].get_pos_from(), \
                aux[edges].get_pos_to())
        # Edges for three winding transformers
        aux = self.get_objects(obj='threewindingtrafo')
        for edges in range(self.get_no_two_winding_trafos()):
            graph.add_edge(aux[edges].get_pos_bus1(), \
                aux[edges].get_pos_bus2())
            graph.add_edge(aux[edges].get_pos_bus2(), \
                aux[edges].get_pos_bus3())
            graph.add_edge(aux[edges].get_pos_bus3(), \
                aux[edges].get_pos_bus1())
        return graph
    
    def nodes_graph(self, graph=None):
        ''' This class method load all nodes in the graph'''
        auxp = "No bus data has been loaded in the class \
            PowerSystemIslandsIsolations"
        assert self.get_objects(obj='bus') != [], " ".join(auxp.split())
        assert graph.is_multigraph(), "graph is not a graph"

        aux = self.get_objects(obj='bus')
        for nodes in range(self.get_no_buses()):
            graph.add_node(aux[nodes].get_pos(), obj=aux[nodes])
        return graph
    
    def find_islands(self):
        ''' This class method finds all islands in the whole power system '''
        G = nx.MultiGraph()
        G = self.nodes_graph(G)
        G = self.edges_graph(G)
        if nx.number_connected_components(G) > 0:
            self.__set_no_islands(nx.number_connected_components(G))
            self.__data['Islands'] = [ElectricityNetwork() for _ in \
                range(self.get_no_islands())]
            S = [G.subgraph(c).copy() for c in \
                nx.connected_components(G)]
            self.__add_nodes_to_island(S, list(range(self.get_no_islands())))
            self.__add_conv_to_island()
            self.__add_hydro_to_island()
            self.__add_renewables_to_island()
            self.__add_transmission_lines_to_island()
            self.__add_two_winding_trafos_to_island()
            self.__add_three_winding_trafos_to_island()
            auxp = 'Network analyser message - the power system under \
                analysis has {0} islands'.format(\
                str(self.get_no_islands()))
            logging.info(" ".join(auxp.split()))
        else:
            auxp = 'Network analyser message - the power system under \
                analysis does not have islands'
            logging.info(" ".join(auxp.split()))

    def find_isolates(self):
        ''' This class method finds all isolated nodes in the power system '''
        G = nx.MultiGraph()
        G = self.nodes_graph(G)
        G = self.edges_graph(G)
        isolatednodesgraph = list(nx.isolates(G))
        if isolatednodesgraph != []:
            G.remove_nodes_from(isolatednodesgraph)
            isolatednodesgraph.sort()
            self.__data['IsolatedNodes'] = \
                self.__delete_nodes_graph(isolatednodesgraph)
            self.__data['NoIsolatedNodes'] = len(self.__data['IsolatedNodes'])
            auxp = 'Network analyser message - the power system under \
                analysis has {0} isolated nodes'.format(\
                str(self.__data['NoIsolatedNodes']))
            logging.info(" ".join(auxp.split()))
        else:
            auxp = 'Network analyser message - the power system under \
                analysis does not have isolated nodes'
            logging.info(" ".join(auxp.split()))
    
    def get_islands(self):
        ''' Get islands - list of ElectricityNetwork objects'''
        return self.__data['Islands']
    
    def get_isolated_nodes(self):
        ''' Get isolated nodes - list of Bus objects'''
        return self.__data['IsolatedNodes']

    def get_no_islands(self):
        ''' Get number of islands '''
        return self.__data['NoIslands']
    
    def get_no_isolated_nodes(self):
        ''' Get number of isolated nodes '''
        return self.__data['NoIsolatedNodes']

    def __add_conv_to_island(self):
        ''' This class method add the nodes to the islands '''
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
                "List of islands does not have the necessary format - conv"
        for aux1 in range(self.get_no_islands()):
            aux_conv = []
            for aux2 in self.__data['Islands'][aux1].get_objects(obj='bus'):
                for aux3, aux4 in zip(aux2.get_gen_type(), aux2.get_gen_pos()):
                    if aux3 == 'conv':
                        aux_conv.append(self.get_objects(obj=aux3, pos=aux4))
            if aux_conv != []:
                self.__data['Islands'][aux1].set_conv_data(aux_conv)
    
    def __add_hydro_to_island(self):
        ''' This class method add the nodes to the islands '''
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
                "List of islands does not have the necessary format - hydro"
        for aux1 in range(self.get_no_islands()):
            aux_hydro = []
            for aux2 in self.__data['Islands'][aux1].get_objects(obj='bus'):
                for aux3, aux4 in zip(aux2.get_gen_type(), aux2.get_gen_pos()):
                    if aux3 == 'hydro':
                        aux_hydro.append(getattr(self,aux3)[aux4])
            if aux_hydro != []:
                self.__data['Islands'][aux1].set_hydro_data(aux_hydro)

    def __add_nodes_to_island(self, graphs=None, lt=None):
        ''' This class method add the nodes to the islands '''
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
                "List of islands does not have the necessary format"
        if isinstance(graphs, list):
            assert graphs[0].is_multigraph(), "graphs is not a graph"
            auxp = "No list of islands or island number has been \
                passed"
            assert lt != None, " ".join(auxp.split())
            if isinstance(lt, list):
                for aux in lt:
                    aux1 = [aux2[1]['obj'] for aux2 in \
                        graphs[aux].nodes(data=True)]
                    self.__data['Islands'][aux].set_bus_data(aux1)
                    del aux1
            else:
                aux1 = [aux2[1]['obj'] for aux2 in \
                        graphs[lt].nodes(data=True)]
                self.__data['Islands'][lt].set_bus_data(aux1)

    def __add_renewables_to_island(self):
        ''' This class method add the nodes to the islands '''
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
                "List of islands does not have the necessary format - RES"
        for aux1 in range(self.get_no_islands()):
            aux_renewables = []
            for aux2 in self.__data['Islands'][aux1].get_objects(obj='bus'):
                for aux3, aux4 in zip(aux2.get_gen_type(), aux2.get_gen_pos()):
                    if aux3 == 'RES':
                        aux_renewables.append(getattr(self,aux3)[aux4])
            if aux_renewables != []:
                self.__data['Islands'][aux1].set_renewable_data(aux_renewables)#

    def __add_transmission_lines_to_island(self):
        ''' This class method add the transmission lines to the islands '''
        auxp = "List of islands does not have the necessary format - \
            Transmission Lines"
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
            " ".join(auxp.split())
        for aux1 in range(self.get_no_islands()):
            aux_transmission_lines = []
            aux_tranmission_lines_list = []
            for aux2 in self.__data['Islands'][aux1].get_objects(obj='bus'):
                for aux3 in aux2.get_transmission_line_from():
                    aux_transmission_lines.append(aux3)
                for aux3 in aux2.get_transmission_line_to():
                    aux_transmission_lines.append(aux3)
            aux_transmission_lines = list(dict.fromkeys(aux_transmission_lines))
            for aux2 in aux_transmission_lines:
                aux_tranmission_lines_list.append(\
                    self.get_objects(obj='transmissionline', pos=aux2))
            if aux_tranmission_lines_list != []:
                self.__data['Islands'][aux1].set_transmission_line_data(\
                    aux_tranmission_lines_list)
    
    def __add_two_winding_trafos_to_island(self):
        ''' This class method add the two winding transformers to the islands '''
        auxp = "List of islands does not have the necessary format - \
            Two winding transformers"
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
            " ".join(auxp.split())
        for aux1 in range(self.get_no_islands()):
            aux_two_winding_trafos = []
            aux_two_winding_trafos_list = []
            for aux2 in self.__data['Islands'][aux1].get_objects(obj='bus'):
                for aux3 in aux2.get_two_trafo_from():
                    aux_two_winding_trafos.append(aux3)
                for aux3 in aux2.get_two_trafo_to():
                    aux_two_winding_trafos.append(aux3)
            aux_two_winding_trafos = list(dict.fromkeys(aux_two_winding_trafos))
            for aux2 in aux_two_winding_trafos:
                aux_two_winding_trafos_list.append(\
                    self.get_objects(obj='twowindingtrafo', pos=aux2))
            if aux_two_winding_trafos_list != []:
                self.__data['Islands'][aux1].set_two_winding_trafos_data(\
                    aux_two_winding_trafos_list)

    def __add_three_winding_trafos_to_island(self):
        ''' This class method add the three winding transformers to the islands '''
        auxp = "List of islands does not have the necessary format - \
            Three winding transformers"
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
            " ".join(auxp.split())
        for aux1 in range(self.get_no_islands()):
            aux_three_winding_trafos_list = []
            aux_three_winding_trafos = \
                self.__get_three_winding_trafos_from_nodes(\
                self.__data['Islands'][aux1])
            for aux2 in aux_three_winding_trafos:
                aux_three_winding_trafos_list.append(\
                    self.get_objects(obj='threewindingtrafo', pos=aux2))
            if aux_three_winding_trafos_list != []:
                self.__data['Islands'][aux1].set_three_winding_trafos_data(\
                    aux_three_winding_trafos_list)
    
    def __get_three_winding_trafos_from_nodes(self, obj=None):
        ''' Get the three winding transformers end nodes '''
        assert isinstance(obj, ElectricityNetwork), \
            "Object is not an ElectricityNetwork"
        aux_three_winding_trafos = []
        for aux2 in obj.get_objects(obj='bus'):
            for aux3 in aux2.get_three_trafo_end1():
                aux_three_winding_trafos.append(aux3)
            for aux3 in aux2.get_three_trafo_end2():
                aux_three_winding_trafos.append(aux3)
            for aux3 in aux2.get_three_trafo_end3():
                aux_three_winding_trafos.append(aux3)
        aux_three_winding_trafos = list(\
            dict.fromkeys(aux_three_winding_trafos))
        return aux_three_winding_trafos

    def __delete_nodes_graph(self, nodes=None):
        ''' This class method remove the isolated nodes from the graph. The 
        method return a list with all information of the deleted nodes'''
        assert isinstance(nodes, list), "Isolated nodes are not an empty list"
        aux = range(len(nodes))
        isolated_nodes = [Bus() for _ in aux]
        for aux1 in aux:
            isolated_nodes[aux1] = self.get_objects(obj='bus', pos=nodes[aux1])
        self.del_nodes(nodes)
        return isolated_nodes

    def __set_no_islands(self, val=None):
        ''' Set number of islands '''
        assert val is not None, "No value passed for the number of islands"
        self.__data['NoIslands'] = val



class PowerSystemReduction(ElectricityNetwork):
    ''' This class contains all necessary methods to reduce a network depending 
    on the requirements of the user
    
    The options considered are:
    
    1. Simplify generators connected to the same node to an equivalent generator
    2. Simplify loads connected to the same node to an equivalent load
    2. Simplify power system network until a desired voltage level '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def Networkreduction(self):
        ''' This class method controls the network reduction '''
        self.G = nx.MultiGraph()
        

    def Reduction(self):
        ''' This is the main class method'''
        self.Networkreduction()

class TemporalTree():
    
    G = nx.DiGraph()