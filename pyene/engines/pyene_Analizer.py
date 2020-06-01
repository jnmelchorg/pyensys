"""
Created on Mon April 06 2020

This python file containts the classes and methods for the analysis and
modifications of the topology and electrical characteristics of power system.
Furthermore, tools to build the temporal tree are provided

@author: Dr. Jose Nicolas Melchor Gutierrez
"""

import networkx as nx
import logging
import copy
from .pyene_Parameters import ElectricityNetwork, Bus, TransmissionLine

class PowerSystemIslandsIsolations(ElectricityNetwork):
    ''' This class contains all necessary methods to find all islands and 
    isolated nodes in a network '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        __data2 = {
            'IsolatedNodes': [], # Isolated nodes in the power system
            'Islands': [] # Islands in the whole power system
        }
        self._data.update(__data2)
        del __data2
        logging.basicConfig(format='%(asctime)s %(message)s', \
            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
    def islands_isolations(self):
        ''' This class method calls and controls all main methods in this class'''
        auxp = 'Running network analyser - Determining islands and \
            isolated nodes in the power system'
        logging.info(" ".join(auxp.split()))
        copy_electricity_network = copy.deepcopy(\
            self.copy_electricity_network_data())
        self.find_isolates()
        self.find_islands()
        self.set_electricity_network_data(ob=copy_electricity_network)

    def edges_graph(self, graph=None):
        ''' This class method load all edges in the graph'''
        auxp = "No edge data has been loaded in the \
                class PowerSystemIslandsIsolations"
        aux = self.get_object_elements(name_object='transmissionline', \
            name_element='position',  pos_object=0)
        aux1 = self.get_object_elements(name_object='twowindingtrafo', \
            name_element='position',  pos_object=0)
        aux2 = self.get_object_elements(name_object='threewindingtrafo', \
            name_element='position',  pos_object=0)
        assert aux != None or aux1 != None or aux2 != None, \
            " ".join(auxp.split())
        assert graph.is_multigraph(), "graph is not a multigraph"
        # Edges for transmission lines
        aux = self.get_objects(name='transmissionline')
        for edges in aux:
            aux1 = edges.get_element(name='bus_position')
            graph.add_edge(aux1[0], aux1[1])
        # Edges for two winding transformers
        aux = self.get_objects(name='twowindingtrafo')
        for edges in aux:
            aux1 = edges.get_element(name='bus_position')
            graph.add_edge(aux1[0], aux1[1])
        # Edges for three winding transformers
        aux = self.get_objects(name='threewindingtrafo')
        for edges in aux:
            aux1 = edges.get_element(name='bus_position')
            graph.add_edge(aux1[0], aux1[1])
            graph.add_edge(aux1[1], aux1[2])
            graph.add_edge(aux1[2], aux1[0])
        return graph
    
    def nodes_graph(self, graph=None):
        ''' This class method load all nodes in the graph'''
        auxp = "No bus data has been loaded in the class \
            PowerSystemIslandsIsolations"
        aux = self.get_object_elements(name_object='bus', \
            name_element='position',  pos_object=0)
        assert aux != None, " ".join(auxp.split())
        assert graph.is_multigraph(), "graph is not a multigraph"

        aux = self.get_objects(name='bus')
        for nodes in aux:
            aux1 = nodes.get_element(name='position')
            graph.add_node(aux1, obj=nodes)
        return graph
    
    def find_islands(self):
        ''' This class method finds all islands in the whole power system '''
        G = nx.MultiGraph()
        G = self.nodes_graph(G)
        G = self.edges_graph(G)
        if nx.number_connected_components(G) > 0:
            self.__data['Islands'] = [ElectricityNetwork() for _ in \
                range(self.get_no_islands())]
            S = [G.subgraph(c).copy() for c in \
                nx.connected_components(G)]
            self.__add_nodes_to_island(S)
            self.__add_object_to_islands(name_object='conv')
            self.__add_object_to_islands(name_object='hydro')
            self.__add_object_to_islands(name_object='RES')
            self.__add_object_to_islands(name_object='transmissionline')
            self.__add_object_to_islands(name_object='twowindingtrafo')
            self.__add_object_to_islands(name_object='threewindingtrafo')
            self.__update_all_pos_islands()
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
        return len(self.__data['Islands'])
    
    def get_no_isolated_nodes(self):
        ''' Get number of isolated nodes '''
        return len(self.__data['IsolatedNodes'])

    def __add_nodes_to_island(self, graphs=None, lt=None):
        ''' This class method add the nodes to the islands '''
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
                "List of islands does not have the necessary format"
        if isinstance(graphs, list):
            assert graphs[0].is_multigraph(), "graphs is not a multigraph"
            for aux in range(self.get_no_islands()):
                aux1 = [aux2[1]['obj'] for aux2 in \
                    graphs[aux].nodes(data=True)]
                self.__data['Islands'][aux].set_objects(name='bus', \
                    list_obj=aux1)
                del aux1

    def __add_object_to_islands(self, name_object=None):
        ''' This class method add the transmission lines to the islands '''
        auxp = "List of islands does not have the necessary format"
        assert isinstance(self.__data['Islands'], list) and \
            isinstance(self.__data['Islands'][0], ElectricityNetwork), \
            " ".join(auxp.split())
        auxp = "No valid objects name has been passed to the function \
            __add_object_to_islands \
            in the class {0}".format(self.__class__.__name__)
        assert name_object is not None, " ".join(auxp.split())
        for aux1 in range(self.get_no_islands()):
            aux_objects = []
            aux_objects_list = []
            for aux2 in self.__data['Islands'][aux1].get_objects(name='bus'):
                aux_objects.extend(aux2.get_element(\
                    name=name_object+'_position'))
            aux_objects = list(dict.fromkeys(aux_objects))
            for aux2 in aux_objects:
                aux_objects_list.extend(\
                    self.get_objects(name=name_object, pos=aux2))
            if aux_objects_list != []:
                self.__data['Islands'][aux1].set_objects(name=name_object, \
                    list_obj=aux_objects_list)
    
    def __delete_nodes_graph(self, nodes=None):
        ''' This class method remove the isolated nodes from the graph. The 
        method return a list with all information of the deleted nodes'''
        assert isinstance(nodes, list), "Isolated nodes are not an empty list"
        aux = range(len(nodes))
        isolated_nodes = [Bus() for _ in aux]
        for aux1 in aux:
            isolated_nodes[aux1] = self.get_objects(name='bus', pos=nodes[aux1])
        self.delete_objects(name='bus', pos=nodes)
        return isolated_nodes

    def __update_all_pos_islands(self):
        ''' Update the position of all nodes, transmission lines, etc. on 
        each island '''
        for xisl in range(self.get_no_islands()):
            self.__data['Islands'][xisl].update_all_positions()

class PowerSystemReduction(ElectricityNetwork):
    ''' This class contains all necessary methods to reduce a network depending 
    on the requirements of the user
    
    The options considered are:
    
    1. Simplify generators connected to the same node to an equivalent 
        generator - Not implemented
    2. Simplify loads connected to the same node to an equivalent load - 
        Not implemented
    3. Simplify power system network until a desired voltage level - network 
        characteristics of the reduced network are omitted
    4. Simplify power system network until a desired voltage level - network 
        characteristics of the reduced network are omitted - Not implemented'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        __data2 = {
            'VoltageLevels': [], # Voltage levels in the power system
            'Supernodeinitial' : None,      # Initial node for supernodes
            'Supernodelineinitial' : None,  # Initial artificial line for 
                                            # supernodes
            'SupernodesNetworkInfo': []     # Network info of supernodes
        }
        self._data.update(__data2)
        del __data2
        logging.basicConfig(format='%(asctime)s %(message)s', \
            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
    def network_reduction_voltage_no_characteristics(self, vol_kv=None):
        ''' This class method controls the functions to reduce the network \
        until the desired voltage level (vol_kv) without including 
        electrical characteristics of the eliminated elements

        The vol parameter needs to be in kV
        The electricity_network parameter needs to be an object of the 
            ElectricityNetwork class
        '''
        assert vol_kv is not None, "No voltage passed to reduce the network"
        self.__find_voltage_levels()
        auxp = "The indicated voltage level is not in the list of voltage \
            levels. The valid voltages are: {0}".format(\
            self._data['VoltageLevels'])
        assert vol_kv in self._data['VoltageLevels'], " ".join(auxp.split())
        flags = {
            'bus' : [False for _ in \
            range(self.get_no_objects(name='bus'))], # Boolean list of Bus 
                # objects
            'transmissionline' : [False for _ in \
            range(self.get_no_objects(name='transmissionline'))], # Boolean 
                # list of transmission line objects
            'transformers' : [False for _ in \
            range(self.get_no_objects(name='transformers'))], # Boolean 
                # list of two winding trafo objects
            }
        flag_continue=True
        counter = 0
        supernodes = []
        while flag_continue:
            flag_continue = False
            for xaux in range(self.get_no_objects(name='bus')):
                if not flags['bus'][xaux]:
                    supernode = { # This dictionary contains the positions of the elements
                        # of the network that will be converted into a supernode
                        'bus' : [], # list of Bus objects in the supernode
                        'coupling_buses' : [], # list with the positions of nodes in the 
                            # supernode that interconnect the supernode with the reduced
                            # network
                        'coupling_bus_name' : None, # name of the coupling node  
                            # that interconnect the supernode with the reduced network
                        'transmissionline' : [], # list of transmission line objects in the
                            # supernode
                        'transformers' : [], # list of trafo objects in the
                            # supernode
                        'equivalent_active_power_demand' : None, # Equivalent active power demand
                            # to be connected to the supernode
                        'equivalent_reactive_power_demand' : None, # Equivalent reactive power demand
                            # to be connected to the supernode
                        'thermal_limit_artificial_lines' : [],
                        'equivalent_non_technical_losses_fix' : None,
                        'zone_supernode' : None
                    }
                    flags, supernode, flag_supernode = self.__voltage_track(\
                        bus_position=xaux, vol_kv=vol_kv, \
                        flags=flags, supernode=supernode, flag_supernode=False)
                    if flag_supernode:
                        supernode['coupling_bus_name'] = \
                            'supernode_'+str(counter)
                        counter += 1
                        supernodes.append(supernode)
                    flag_continue = True
        if supernodes != []:
            auxp = 'Network analyser message - the power system under \
                analysis has {0} supernodes'.format(len(supernodes))
            logging.info(" ".join(auxp.split()))
            self.__reduce_network_from_supernodes(supernodes=supernodes)
            self.__add_artificial_elements(supernodes=supernodes)
        else:
            auxp = 'Network analyser message - the power system under \
                analysis has not been reduced - check if the indicated voltage \
                    {0} is correct'.format(vol_kv)
            logging.info(" ".join(auxp.split()))
        return supernodes
                    
    def reduction(self):
        ''' This is the main class method'''
        self.network_reduction_voltage_no_characteristics()
    
    def __add_artificial_elements(self, supernodes=None):
        ''' This class method add the artificial buses and lines to the 
        network '''
        assert isinstance(supernodes, list), "Incorrect \
            object passed for supernodes"
        copy_nodes = self.get_objects(name='bus')
        number_nodes = []
        for xaux in copy_nodes:
            number_nodes.append(xaux.get_element(name='number'))
        for xaux in self._data['SupernodesNetworkInfo']:
            for xaux1 in xaux['bus']:
                number_nodes.append(xaux1.get_element(name='number'))
        aux = 1
        while aux <= max(number_nodes):
            aux *= 10
        
        copy_lines = self.get_objects(name='transmissionline')
        number_lines = []
        for xaux in copy_lines:
            number_lines.append(xaux.get_element(name='number'))
        for xaux in self._data['SupernodesNetworkInfo']:
            for xaux1 in xaux['transmissionline']:
                number_nodes.append(xaux1.get_element(name='number'))
        auxline = 1
        while auxline <= max(number_lines):
            auxline *= 10        

        new_nodes = []
        new_lines = []
        counternodes = 0
        counterlines = 0
        for x in range(len(supernodes)):
            if len(supernodes[x]['coupling_buses']) > 1:
                if counternodes == 0:
                    auxp = 'Network analyser message - artificial nodes with \
                        numbering starting from {0} are added to the \
                        system'.format(aux)
                    logging.info(" ".join(auxp.split()))
                    auxp = 'Network analyser message - artificial lines with \
                        numbering starting from {0} are added to the \
                        system'.format(auxline)
                    if self._data['Supernodeinitial'] == None:
                        self._data['Supernodeinitial'] = aux
                    if self._data['Supernodelineinitial'] == None:
                        self._data['Supernodelineinitial'] = auxline
                    logging.info(" ".join(auxp.split()))
                new_nodes.append(Bus())
                new_nodes = self.__add_artificial_node(supernodes=supernodes, \
                    counter=counternodes, number_node=aux, new_nodes=new_nodes,\
                    pos_supernode=x)
                self.__add_generators_to_artificial_node(supernodes=supernodes,\
                    counter=counternodes, number_node=aux, new_nodes=new_nodes,\
                    pos_supernode=x)
                counterlines, auxline = self.__add_artificial_lines(\
                    supernodes=supernodes, counter=counterlines, \
                    number_line=auxline, new_nodes=new_nodes, \
                    counternode=counternodes, number_node=None, pos_supernode=x,
                    new_lines=new_lines)
                counternodes += 1
                aux += 1
            else:
                for xnode in copy_nodes:
                    if xnode.get_element(name='number') == \
                        self._data['SupernodesNetworkInfo'][x]\
                        ['coupling_buses'][0].get_element(name='number'):
                        xnode.set_element(name='active_power_demand_peak',\
                            val=supernodes[x]\
                                ['equivalent_active_power_demand'] + \
                            xnode.get_element(name='active_power_demand_peak'))
                        xnode.set_element(name='reactive_power_demand_peak',\
                            val=supernodes[x]\
                                ['equivalent_reactive_power_demand'] + \
                            xnode.get_element(\
                                name='reactive_power_demand_peak'))
                        xnode.set_element(name='non_technical_losses_fix',\
                            val=supernodes[x][\
                                'equivalent_non_technical_losses_fix']+
                            xnode.get_element(\
                                name='non_technical_losses_fix'))
                        self.__add_generators_to_artificial_node(\
                            supernodes=supernodes, counter=\
                            xnode.get_element(name='position'), \
                            number_node=xnode.get_element(name='number'), \
                            new_nodes=copy_nodes, \
                            pos_supernode=x)
        
        copy_nodes.extend(new_nodes)
        copy_lines.extend(new_lines)
        self.set_objects(name='bus', list_obj=copy_nodes)
        self.set_objects(name='transmissionline', list_obj=copy_lines)
        self.update_all_positions()

    def __add_artificial_node(self, supernodes=None, counter=None, \
        number_node=None, new_nodes=None, pos_supernode=None):        
        ''' This class method add one artificial bus to the list of artificial 
        buses '''
        new_nodes[counter].set_element(name='voltage_kv', val=0)
        new_nodes[counter].set_element(name='name', \
            val=supernodes[pos_supernode]['coupling_bus_name'])
        new_nodes[counter].set_element(name='number', val=number_node)
        new_nodes[counter].set_element(name='type', val=1)
        new_nodes[counter].set_element(\
            name='maximum_voltage_magnitude', val=10)
        new_nodes[counter].set_element(\
            name='minimum_voltage_magnitude', val=0)
        new_nodes[counter].set_element(name='non_technical_losses_fix',\
            val=supernodes[pos_supernode][\
            'equivalent_non_technical_losses_fix'])
        new_nodes[counter].set_element(name='active_power_demand_peak',\
            val=supernodes[pos_supernode]['equivalent_active_power_demand'])
        new_nodes[counter].set_element(\
            name='reactive_power_demand_peak',\
            val=supernodes[pos_supernode]['equivalent_reactive_power_demand'])
        new_nodes[counter].set_element(\
            name='initial_voltage_magnitude', val=1)
        new_nodes[counter].set_element(\
            name='initial_voltage_angle', val=0)
        new_nodes[counter].set_element(\
            name='zone', val=supernodes[pos_supernode]['zone_supernode'])        
        return new_nodes

    def __add_artificial_lines(self, supernodes=None, counter=None, \
        number_line=None, new_nodes=None, counternode=None, number_node=None,
        pos_supernode=None, new_lines=None):
        ''' This class method adds the artificial transmission lines to the 
        electricity network '''
        xcounter = 0
        number_lines = []
        for xnodespos in supernodes[pos_supernode]['coupling_buses']:
            new_lines.append(TransmissionLine())
            new_lines[counter].set_element(name='shunt_susceptance', val=0)
            new_lines[counter].set_element(name='resistance', val=0)
            new_lines[counter].set_element(name='reactance', val=0.01)
            new_lines[counter].set_element(name='number', val=number_line)
            new_lines[counter].set_element(name='long_term_thermal_limit', \
                val=supernodes[pos_supernode]['thermal_limit_artificial_lines'][xcounter])
            new_lines[counter].set_element(name='non_technical_losses_fix', \
                val=0)
            new_lines[counter].set_element(name='non_technical_losses_fix', \
                val=0)
            new_lines[counter].set_element(name='bus_number', \
                val=[self._data['SupernodesNetworkInfo'][pos_supernode]\
                    ['coupling_buses'][xcounter].get_element(name='number'), \
                    number_node])
            number_lines.append(number_line)
            xcounter += 1
            counter += 1
            number_line += 1
        new_nodes[counternode].set_element(name='transmissionline_number', \
            val=number_lines)

        return counter, number_line

    def __add_generators_to_artificial_node(self, supernodes=None, counter=None, \
        number_node=None, new_nodes=None, pos_supernode=None):
        ''' This class method adds generators to the artificial node '''
        for xgentype in self.get_generation_types_names():
            for xgen in self.get_objects(name=xgentype):
                if xgen.get_element(name='bus_position') in \
                    supernodes[pos_supernode]['bus']:
                    xgen.set_element(name='bus_number', val=number_node)
                    new_nodes[counter].set_element(\
                    name=xgentype+'_number', val=number_node)

    def __check_if_empty_list(self, name=None, bus_position=None):
        ''' This method checks if the bus object is connected to the network '''
        assert bus_position is not None, "No bus position passed to check its \
            connectivity"
        assert name is not None, "No name of branch element pass"
        aux = self.get_object_elements(name_object='bus', \
            name_element=name+'_position', pos_object=bus_position)
        if aux != []:
            return False
    
    def __check_positions_branch_objects(self, name=None, bus_position=None, \
        vol_kv=None, flags=None, supernode=None, flag_supernode=None):
        assert bus_position is not None, "No bus position passed to reduce \
            the network"
        assert flags is not None, "No flag dictionary passed to reduce the \
            network"
        assert supernode is not None, "No supernode dictionary passed to \
            reduce the network"
        assert flag_supernode is not None, "No flag for the supernode passed to \
            reduce the network"
        assert name is not None, "No name of branch element pass"

        # Checking if the object exist
        aux = self.get_object_elements(name_object=name, \
            name_element='position', pos_object=0)
        if aux == None:
            return flags, supernode, flag_supernode
        
        auxlist = self.get_object_elements(name_object='bus', \
            name_element=name+'_position', pos_object=bus_position)
        if auxlist == None:
            return flags, supernode, flag_supernode

        for xauxlist in auxlist:
            if not flags[name][xauxlist]:
                list_pos_buses = self.get_object_elements(\
                    name_object=name, \
                    name_element='bus_position', pos_object=xauxlist)
                assert bus_position in list_pos_buses , \
                    "The node should be on the list list_pos_buses"
                flags, supernode, flag_supernode = self.__check_ends_lines(\
                    bus_position=bus_position, vol_kv=vol_kv, \
                    element_position=xauxlist, element_name=name, \
                    list_pos_buses=list_pos_buses, flags=flags, \
                    supernode=supernode, flag_supernode=flag_supernode)                
                if not flags[name][xauxlist]:
                    flags[name][xauxlist] = True
                    supernode[name].append(xauxlist)
        return flags, supernode, flag_supernode

    def __check_ends_lines(self, bus_position=None, \
        vol_kv=None, element_position=None, element_name=None, \
        list_pos_buses=None, flags=None, supernode=None, flag_supernode=None):
        ''' This class method check the nodes that are not connected to a three 
        winding transformer'''
        for xauxlisbuses in list_pos_buses:
            auxvoltage = self.get_object_elements(name_object='bus', \
                name_element='voltage_kv', pos_object=xauxlisbuses)
            if xauxlisbuses != bus_position and \
                not flags['bus'][xauxlisbuses]:
                flags, supernode, flag_supernode = \
                    self.__voltage_track(bus_position=xauxlisbuses, \
                    vol_kv=vol_kv, flags=flags, supernode=supernode, \
                    flag_supernode=flag_supernode)
                if  auxvoltage >= vol_kv and auxvoltage != \
                    self.get_element(name='voltagethreewindingtrafos'):
                    supernode['thermal_limit_artificial_lines'].append(\
                        self.get_object_elements(name_object=element_name, \
                        name_element='long_term_thermal_limit', \
                        pos_object=element_position))
            elif xauxlisbuses != bus_position and \
                flags['bus'][xauxlisbuses] and \
                auxvoltage >= vol_kv and \
                xauxlisbuses not in supernode['bus'] \
                and auxvoltage != self.get_element(\
                name='voltagethreewindingtrafos'):
                    supernode['coupling_buses'].append(xauxlisbuses)
                    supernode['thermal_limit_artificial_lines'].append(\
                        self.get_object_elements(name_object=element_name, \
                        name_element='long_term_thermal_limit', \
                        pos_object=element_position))
        return flags, supernode, flag_supernode

    def __find_voltage_levels(self):
        ''' This method finds all voltage levels in the system '''
        aux = []
        for xaux in range(self.get_no_objects(name='bus')):
            aux.append(self.get_object_elements(name_object='bus', \
                name_element='voltage_kv', pos_object=xaux))
        aux1 = []
        for xn in aux:
            flag_vol = False
            for xvol in aux1:
                if xn == xvol:
                    flag_vol = True
                    break
            if not flag_vol:
                aux1.append(xn)
        aux1.sort(reverse=True)
        self._data['VoltageLevels'] = aux1
    
    def __list_elements_to_remove(self, vol_kv=None, electricity_network=None):
        ''' This function return the list of elements (nodes, lines, trafos, 
        etc) to be eliminated in the network reduction'''
        copy_electricity_network = ElectricityNetwork()
        copy_electricity_network.set_electricity_network_data(\
            electricity_network)
    
    def __reduce_network_from_supernodes(self, supernodes=None):
        ''' This class method reduce the network based on the information of 
            supernodes '''
        assert isinstance(supernodes, list), "Incorrect \
            object passed for supernodes"
        
        self.__save_network_info_in_supernodes(supernodes=supernodes)
        busestoerase = []
        for xsuper in supernodes:
            print(xsuper['coupling_buses'])
            print(xsuper['bus'])
            xsuper['zone_supernode'] = self.get_object_elements(\
                    name_object='bus', \
                    name_element='zone', pos_object=\
                    xsuper['coupling_buses'][0])
            busestoerase.extend(xsuper['bus'])
            xsuper['equivalent_active_power_demand'] = 0
            xsuper['equivalent_reactive_power_demand'] = 0
            xsuper['equivalent_non_technical_losses_fix'] = 0
            for xbus in xsuper['bus']:
                xsuper['equivalent_active_power_demand'] += \
                    self.get_object_elements(\
                    name_object='bus', \
                    name_element='active_power_demand_peak', pos_object=xbus)
                xsuper['equivalent_reactive_power_demand'] += \
                    self.get_object_elements(\
                    name_object='bus', \
                    name_element='reactive_power_demand_peak', pos_object=xbus)
                xsuper['equivalent_non_technical_losses_fix'] += \
                    self.get_object_elements(\
                    name_object='bus', \
                    name_element='non_technical_losses_fix', pos_object=xbus)
        self.delete_objects(name='bus', pos=busestoerase)
        for xseries in self.get_series_elements_names():
            elementstoerase = []
            for xsuper in supernodes:
                elementstoerase.extend(xsuper[xseries])
                for xserieselements in xsuper[xseries]:
                    auxlossfix = self.get_object_elements(\
                        name_object=xseries, \
                        name_element='non_technical_losses_fix', \
                        pos_object=xserieselements)
                    if auxlossfix != None:
                        xsuper['equivalent_non_technical_losses_fix'] += \
                            auxlossfix
            self.delete_objects(name=xseries, pos=elementstoerase)

    def __save_network_info_in_supernodes(self, supernodes=None):
        ''' This method class save the network information for the supernodes 
        '''
        self._data['SupernodesNetworkInfo'] = []
        for xsuper in supernodes:
            supernode = { # This dictionary contains the positions of the elements
                # of the network that will be converted into a supernode
                'bus' : self.get_objects(name='bus', pos=xsuper['bus']),
                'coupling_buses' : self.get_objects(name='bus', \
                    pos=xsuper['coupling_buses']),
                'coupling_bus_name' : xsuper['coupling_bus_name'], # name of the coupling node  
                    # that interconnect the supernode with the reduced network
                'transmissionline' : self.get_objects(name='transmissionline', \
                    pos=xsuper['transmissionline']),
                'transformers' : self.get_objects(name='transformers', \
                    pos=xsuper['transformers'])
            }
            self._data['SupernodesNetworkInfo'].append(supernode)

    def __voltage_track(self, bus_position=None, vol_kv=None, flags=None, \
        supernode=None, flag_supernode=None):
        ''' Recursive function for finding next voltage level '''
        assert bus_position is not None, "No bus position passed to reduce \
            the network"
        assert flags is not None, "No flag dictionary passed to reduce the \
            network"
        assert supernode is not None, "No supernode dictionary passed to \
            reduce the network"
        assert flag_supernode is not None, "No flag for the supernode passed to \
            reduce the network"
        
        if not flag_supernode:
            aux = self.get_object_elements(name_object='bus', \
                name_element='voltage_kv', pos_object=bus_position)
            if aux >= vol_kv and aux != self.get_element(\
                name='voltagethreewindingtrafos'):
                print(bus_position)
                print(aux)
                aux[10000]
                flags['bus'][bus_position] = True
                return flags, supernode, flag_supernode
        else:
            aux = self.get_object_elements(name_object='bus', \
                name_element='voltage_kv', pos_object=bus_position)
            if aux >= vol_kv and aux != self.get_element(\
                name='voltagethreewindingtrafos'):
                # If a supernode exist add the node to the list of coupling 
                # nodes
                flags['bus'][bus_position] = True
                print(bus_position)
                aux[10000]
                supernode['bus'].append(bus_position)
                supernode['coupling_buses'].append(bus_position)
                return flags, supernode, flag_supernode
        
        flags['bus'][bus_position] = True
        is_disconnected = True
        for key in flags.keys():
            if key != 'bus':
                is_disconnected = self.__check_if_empty_list(\
                    name=key, bus_position=bus_position)
            if not is_disconnected:
                break
        if is_disconnected:
            return flags, supernode, flag_supernode
        
        flag_supernode = True
        supernode['bus'].append(bus_position)
        
        for xlist in self.get_series_elements_names():
            flags, supernode, flag_supernode = \
                self.__check_positions_branch_objects(name=xlist, \
                bus_position=bus_position, vol_kv=vol_kv, \
                flags=flags, supernode=supernode, flag_supernode=flag_supernode)
        
        return flags, supernode, flag_supernode
    
class TemporalTree():
    
    G = nx.DiGraph()