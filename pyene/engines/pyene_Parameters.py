"""
A common template of parameters for power systems and temporal trees is provided
in this file. Most of the initial functions have been taken from the file pyeneD
created by Dr Eduardo Alejandro Martínez Ceseña.

@author: Dr. Jose Nicolas Melchor Gutierrez
         Dr Eduardo Alejandro Martínez Ceseña
"""

import numpy as np

'''                               DEVICE CLASSES                            '''

class Branch:
    ''' Electricity branch - This could be information for either transmission \
        lines or two winding transformers '''
    def __init__(self):
        # Basic __data
        aux = ['BR_B', 'BR_R', 'BR_STATUS', 'BR_X',
               'Number', 'F_BUS', 'RATE_A', 'RATE_B', 'RATE_C',
               'T_BUS', 'Loss_Fix', 'Position', 'F_Position', 'T_Position',
               'N-1']
        self.__data = {}
        for x in aux:
            self.__data[x] = None

    def get_bus_from(self):
        ''' Get bus number at beginning (from) of the branch '''
        return self.__data['F_BUS']

    def get_bus_to(self):
        ''' Get bus number at end (to) of the branch '''
        return self.__data['T_BUS']

    def get_charging_susceptance(self):
        ''' Get Susceptance '''
        return self.__data['BR_B']

    def get_contingency(self, xs=':'):
        ''' Get values for a single N-1 condition '''
        return self.__data['N-1'][xs]

    def get_long_rate(self):
        ''' Get Rate A (long term rating) for normal operation conditions'''
        return self.__data['RATE_A']

    def get_loss(self):
        ''' Return non technical losses in the bus '''
        return self.__data['Loss_Fix']

    def get_number(self):
        ''' Get branch number '''
        return self.__data['Number']

    def get_pos(self):
        ''' Get position of the branch - starting from zero'''
        return self.__data['Position']

    def get_pos_from(self):
        ''' Get bus position at beginning (from) of the branch '''
        return self.__data['F_Position']

    def get_pos_to(self):
        ''' Get bus position at end (to) of the branch '''
        return self.__data['T_Position']

    def get_reactance(self):
        ''' Get Reactance '''
        return self.__data['BR_X']

    def get_resistance(self):
        ''' Get Resistance '''
        return self.__data['BR_R']

    def is_active(self, xs):
        ''' Is the line connected in this scenario? '''
        return self.__data['N-1'][xs] is not None

    def set_bus_from(self, val=None):
        ''' Set bus number at beginning (from) of the branch '''
        assert val is not None, "No value passed for the number of the bus from"
        self.__data['F_BUS'] = val

    def set_bus_to(self, val=None):
        ''' Set bus number at end (to) of the branch '''
        assert val is not None, "No value passed for the number of the bus to"
        self.__data['T_BUS'] = val

    def set_charging_susceptance(self, val=None):
        ''' Set Susceptance'''
        assert val is not None, "No value passed for the line charging \
        susceptance"
        self.__data['BR_B'] = val

    def set_long_rate(self, val=None):
        ''' Set Rate A (long term rating) for normal operation conditions'''
        assert val is not None, "No value passed for the Rate A"
        self.__data['RATE_A'] = val

    def set_number(self, val=None):
        ''' Set branch number '''
        assert val is not None, "No value passed for the number of the line"
        self.__data['Number'] = val
    
    def set_pos(self, val=None):
        ''' Set Position of Bus on the list of branches - starting from zero'''
        assert val is not None, "No value passed for the position of the branch"
        self.__data['Position'] = val


    def set_pos_from(self, val=None):
        ''' Set bus position at beginning (from) of the branch '''
        assert val is not None, "No value passed for the position of the bus \
        from"
        self.__data['F_Position'] = val

    def set_pos_to(self, val=None):
        ''' Set bus position at end (to) of the branch '''
        assert val is not None, "No value passed for the position of the bus \
        to"
        self.__data['T_Position'] = val

    def set_contingency(self, val=None, x=None):
        ''' Set values for all conditions '''
        assert val is not None, "No value passed for the contingency"
        if x is None:
            self.__data['N-1'] = val
        else:
            self.__data['N-1'][x] = val

    def set_reactance(self, val=None):
        ''' Set Reactance'''
        assert val is not None, "No value passed for the reactance"
        self.__data['BR_X'] = val

    def set_resistance(self, val=None):
        ''' Set Resistance'''
        assert val is not None, "No value passed for the resistance"
        self.__data['BR_R'] = val

class TwoWindingTrafo(Branch):
    ''' Two winding transformer class '''
    def __init__(self):
        super().__init__()
        # Basic __data
        aux = ['ANGMAX', 'ANGMIN', 'TAP', 'BASE_KV_FROM', 'BASE_KV_TO']
        self.__data.update(aux)
        for x in aux:
            self.__data[x] = None
    def get_Tap(self):
        ''' Get tap position '''
        return self.__data['TAP']

    def set_Tap(self, val=None):
        ''' Set tap position - float number '''
        assert val is not None, "No value passed to set the Tap position"
        self.__data['TAP'] = val

class TransmissionLine(Branch):
    ''' Transmission Line class '''
    def __init__(self):
        super().__init__()

class Bus:
    ''' Electricity bus '''
    def __init__(self):
        # Basic __data
        aux = ['BASE_KV', 'BS', 'BUS_AREA', 'BUS_TYPE', 'BUS_X', 'BUS_Y',
               'Demand', 'GS', 'PeakP', 'PeakQ', 'Position', 'Name', 'Number',
               'VM', 'VA', 'VMAX', 'VMIN', 'ZONE', 'Load_Type', 'Loss_Fix',
               'NoFB', 'NoTB', 'GenType', 'NoFTW', 'NoTTW', 'NoThWEnd1',
               'NoThWEnd2', 'NoThWEnd3']
        self.__data = {}
        for x in aux:
            self.__data[x] = None

        aux =  ['N-1', 'F_TLines', 'T_TLines', 'GenType','GenPosition',
                'F_TWTrafo', 'T_TWTrafo', 'End1_ThWTrafo', 'End2_ThWTrafo',
                'End3_ThWTrafo']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self.__data.update(__data2)
        del __data2

    def add_transmission_line_from(self, val):
        ''' Append value to F_TLines - Transmission lines connected to the node
        at the end from'''
        self.__data['F_TLines'].append(val)
        self.__data['NoFB'] += 1

    def add_transmission_line_to(self, val):
        ''' Append value to T_TLines - Transmission lines connected to the node
        at the end to'''
        self.__data['T_TLines'].append(val)
        self.__data['NoTB'] += 1

    def add_gen(self, xt, xp):
        ''' Append generator type and position '''
        self.__data['GenType'].append(xt)
        self.__data['GenPosition'].append(xp)

    def get_base_voltage(self):
        ''' Get base voltage kV '''
        return self.__data['BASE_KV']

    def get_gen_pos(self):
        ''' Get list of generator positions for each type connected to
        the bus '''
        return self.__data['GenPosition']

    def get_gen_type(self):
        ''' Get list of generator types connected to the bus '''
        return self.__data['GenType']

    def get_max_voltage(self):
        ''' Get max voltage limit (pu) '''
        return self.__data['VMAX']

    def get_min_voltage(self):
        ''' Get max voltage limit (pu) '''
        return self.__data['VMIN']

    def get_latitude(self):
        ''' Get Y coordinates '''
        return self.__data['BUS_Y']

    def get_load_type(self):
        ''' Get load type (1:Urban, 2:Rural) '''
        return self.__data['Load_Type']

    def get_longitude(self):
        ''' Get X coordinates '''
        return self.__data['BUS_X']

    def get_loss(self):
        ''' Return non technical losses in the bus '''
        return self.__data['Loss_Fix']

    def get_number(self):
        ''' Get Bus number '''
        return self.__data['Number']

    def get_pos(self):
        ''' Get Bus position - beginning from zero '''
        return self.__data['Position']

    def get_security(self, xs=':'):
        ''' Get position of variable in N-1 scenario '''
        return self.__data['N-1'][xs]
    
    def get_three_trafo_end1(self):
        ''' Get list of three winding transformers connected the bus at end 1 '''
        return self.__data['End1_ThWTrafo']
    
    def get_three_trafo_end2(self):
        ''' Get list of three winding transformers connected the bus at end 2 '''
        return self.__data['End2_ThWTrafo']
    
    def get_three_trafo_end3(self):
        ''' Get list of three winding transformers connected the bus at end 3 '''
        return self.__data['End3_ThWTrafo'] 

    def get_transmission_line_from(self):
        ''' Get list of transmission lines connected from the bus '''
        return self.__data['F_TLines']

    def get_transmission_line_to(self):
        ''' Get list of transmission lines connected to the bus '''
        return self.__data['T_TLines']   

    def get_two_trafo_from(self):
        ''' Get list of two winding transformers connected from the bus '''
        return self.__data['F_TWTrafo']

    def get_two_trafo_to(self):
        ''' Get list of two winding transformers connected to the bus '''
        return self.__data['T_TWTrafo']

    def get_type(self):
        ''' Get bus type '''
        return self.__data['BUS_TYPE']

    def get_voltage_magnitude(self):
        ''' Get Voltege magnitude (pu) '''
        return self.__data['VM']
    
    def set_gen_pos(self, lt=None):
        ''' Set list of generator positions for each type connected to
        the bus '''
        assert isinstance(lt, list), "The passed value is \
            not an list for GenPosition"
        self.__data['GenPosition'] = lt
    
    def set_gen_type(self, lt=None):
        ''' Set list of generator types connected to the bus '''
        assert isinstance(lt, list), "The passed value is \
            not an list for GenPosition"
        self.__data['GenType'] = lt

    def set_load_type(self, val):
        ''' Set load type (0:Urban, 1:Rural) '''
        self.__data['Load_Type'] = val

    def set_name(self, val=None):
        ''' Set Name of Bus '''
        assert val is not None, "No value passed for the name of the bus"
        self.__data['Name'] = val

    def set_number(self, val=None):
        ''' Set original Number of Bus '''
        assert val is not None, "No value passed for the Number of the bus"
        self.__data['Number'] = val
    
    def set_pos(self, val=None):
        ''' Set Position of Bus on the list of buses - starting from zero'''
        assert val is not None, "No value passed for the position of the bus"
        self.__data['Position'] = val

    def set_security(self, val, x=None):
        ''' Set values for all conditions '''
        if x is None:
            self.__data['N-1'] = val
        else:
            self.__data['N-1'][x] = val
    
    def set_transmission_line_from(self, lt=None):
        ''' Set list of transmission lines connected from the bus '''
        assert isinstance(lt, list), "The passed value is \
            not an list for F_TLines"
        self.__data['F_TLines'] = lt

    def set_transmission_line_to(self, lt=None):
        ''' Set list of transmission lines connected to the bus '''
        assert isinstance(lt, list), "The passed value is \
            not an list for T_TLines"
        self.__data['T_TLines'] = lt
    
    def set_three_trafo_end1(self, lt=None):
        ''' Set list of three winding transformers connected the bus at end 1 '''
        assert isinstance(lt, list), "The passed value is \
            not an list for End1_ThWTrafo"
        self.__data['End1_ThWTrafo'] = lt
    
    def set_three_trafo_end2(self, lt=None):
        ''' Set list of three winding transformers connected the bus at end 2 '''
        assert isinstance(lt, list), "The passed value is \
            not an list for End2_ThWTrafo"
        self.__data['End2_ThWTrafo'] = lt
    
    def set_three_trafo_end3(self, lt=None):
        ''' Set list of three winding transformers connected the bus at end 3 '''
        assert isinstance(lt, list), "The passed value is \
            not an list for End3_ThWTrafo"
        self.__data['End3_ThWTrafo'] = lt

    def set_two_trafo_from(self, lt=None):
        ''' Set list of two winding transformers connected from the bus '''
        assert isinstance(lt, list), "The passed value is \
            not an list for F_TWTrafo"
        self.__data['F_TWTrafo'] = lt

    def set_two_trafo_to(self, lt=None):
        ''' Set list of two winding transformers connected to the bus '''
        assert isinstance(lt, list), "The passed value is \
            not an list for T_TWTrafo"
        self.__data['T_TWTrafo'] = lt
    
    def update_gen_pos(self, poslt=None, newpos=None):
        ''' Update list of generator positions for each type connected to
        the bus '''
        assert poslt != None and newpos!=None, "Some values have not been \
            passed to set the GenPosition"
        self.__data['GenPosition'][poslt] = newpos

class ElectricityNetwork:
    ''' Electricity network '''
    def __init__(self, **kwargs):
        ''' General electricity network settings '''
        self.__data = {
                'baseMVA': None,
                'Slack': None,
                'NoBuses': 0,  # Number of buses
                'bus' : [], # list of Bus objects
                'NoTLines': 0,  # Number of transmission lines
                'transmissionline' : [], # list of transmission line objects
                'Security': [],  # list of N-1 cases to consider
                'SecurityNo': 0,  # Number of N-1 cases
                'NoConv' : 0, # Number of conventional generators
                'conv' : [], # list of conventional generator objects
                'NoHydro' : 0, # Number of hydro generators
                'hydro' : [], # list of hydro generator objects
                'NoRES' : 0, # Number of RES generators
                'RES' : [], # list of RES generator objects
                'NoGen': 0,
                'GenTypes': [],  # Types of generators to be considered
                                # (i.e. conv, RES, hydro, etc)
                'TWtrafos': 0, # Number of two winding transformers
                'twowindingtrafo' : [], # list of two winding trafo objects
                'ThWtrafos': 0, # Number of three winding transformers
                'threewindingtrafo' : [] # list of two winding trafo objects
                }
        if 'NoBus' in kwargs.keys():
            self.__data['NoBuses'] = kwargs.pop('NoBus')
        if 'NoTLines' in kwargs.keys():
            self.__data['NoTLines'] = kwargs.pop('NoTLines')
        if 'NoConv' in kwargs.keys():
            self.__data['NoConv'] = kwargs.pop('NoConv')
        if 'NoHydro' in kwargs.keys():
            self.__data['NoHydro'] = kwargs.pop('NoHydro')
        if 'NoRES' in kwargs.keys():
            self.__data['NoRES'] = kwargs.pop('NoRES')
        if 'NoThWtrafos' in kwargs.keys():
            self.__data['ThWtrafos'] = kwargs.pop('NoThWtrafos')
        if 'NoTWTrafos' in kwargs.keys():
            self.__data['TWtrafos'] = kwargs.pop('NoTWTrafos')

        self.__data['NoGen'] = self.__data['NoConv'] + self.__data['NoHydro'] + \
            self.__data['NoRES']

        # Initialise bus object
        self.__data['bus'] = [Bus() for _ in
                    range(self.__data['NoBuses'])]

        # Initialise Conventional generator object
        self.__data['conv'] = [Conventional() for _ in
                       range(self.__data['NoConv'])]
        if self.__data['NoConv'] > 0: self.__data['GenTypes'].append('conv')

        # Initialise hydro-electrical generator object
        self.__data['hydro'] = [Hydropower() for _ in
                       range(self.__data['NoHydro'])]
        if self.__data['NoHydro'] > 0: self.__data['GenTypes'].append('hydro')

        # Initialise RES generator object
        self.__data['RES'] = [RES() for _ in
                       range(self.__data['NoRES'])]
        if self.__data['NoRES'] > 0: self.__data['GenTypes'].append('RES')

        # Initialise transmission line object
        self.__data['transmissionline'] = [TransmissionLine() for _ in
                       range(self.__data['NoTLines'])]

        # Initialise three winding transformer object
        self.__data['threewindingtrafo'] = [ThreeWindingTrafo() for _ in
                    range(self.__data['ThWtrafos'])]

        # Initialise two winding transformer object
        self.__data['twowindingtrafo'] = [TwoWindingTrafo() for _ in
                    range(self.__data['TWtrafos'])]

    # TODO: new functions to add and delete elements

    def add_new_security(self, val=None):
        ''' Add a new N-1 case to the list of cases '''
        assert val is not None, "No value passed to add a new security \
            (contingency) to the list of contingencies"
        self.__data['Security'].append(val)
        self.__set_no_security(1)

    def add_new_gen_type(self, val=None):
        ''' Add a generator type to the list '''
        assert val is not None, "No value passed to add a generation \
            type to the list of generation types"
        self.__data['GenTypes'].append(val)
    
    def add_nodes(self, obj=None):
        ''' Add nodes into the list'''
        assert obj is not None, "No list of nodes to add has been passed"
        if isinstance(obj, list):
            if isinstance(obj, Bus):
                self.__data['bus'].extend(obj)
                self.__data['NoBuses'] = len(self.__data['bus'])
            else:
                assert "No valid object to extend the list of nodes"
        elif isinstance(obj, Bus):
            self.__data['bus'].append(obj)
            self.__data['NoBuses'] += 1
        else:
            assert "No valid object to extend the list of nodes"
        # Updating positions in bus list
        self.__update_pos_nodes()

    def del_nodes(self, lt=None):
        ''' Delete nodes of the list '''
        assert lt is not None, "No list of nodes to delete has been passed"
        if isinstance(lt, list):
            for aux in lt:
                for aux1 in range(len(self.__data['bus'])):
                    if aux == self.__data['bus'][aux1].get_pos():
                        self.__data['bus'].pop(aux1)
                        break
            self.__data['NoBuses'] -= len(lt)
        else:
            self.__data['bus'].pop(lt)
            self.__data['NoBuses'] -= 1
        # Updating positions in bus list
        self.__update_pos_nodes()

    def get_base(self):
        ''' Provide base MVA rating '''
        return self.__data['baseMVA']
    
    def get_objects(self, obj=None, pos=':'):
        ''' Get the object indicated in "obj" and "pos" 
            
            obj: Name of object to be returned. The options are:
                'bus' -> list of Bus objects
                'transmissionline' -> list of transmission line objects
                'Security' -> list of N-1 cases to consider
                'conv' -> list of conventional generator objects
                'hydro' -> list of hydro generator objects
                'RES' -> list of RES generator objects
                'GenTypes' -> Types of generators to be considered
                                 (i.e. conv, RES, hydro, etc)
                'twowindingtrafo' -> list of two winding trafo objects
                'threewindingtrafo' -> list of two winding trafo objects
                
            pos: Position or list of positions of the object that will be 
            returned. The function will return the whole list of the object by 
            default but the user can pass a list or an integer value '''
        assert obj is not None, "No valid name has been passed"
        if obj in self.__data:
            if pos == ':':
                return self.__data[obj]
            elif isinstance(pos, list):
                aux = []
                for aux1 in pos:
                    aux.append(self.__data[obj][aux1])
                return aux
            else:
                return self.__data[obj][pos]        

    def get_flow_from(self, xn, xs):
        ''' Get transmission lines connected from bus per scenario '''
        aux = []
        for xb in self.__data['bus'][xn].get_transmission_line_from(): # Transmission line  
            # connected to the bus
            # Is the transmission line active in the scenario?
            if self.__data['transmissionline'][xb].is_active(xs):
                aux.append(self.__data['transmissionline'][xb].get_N1(xs))
        return aux

    def get_flow_to(self, xn, xs):
        ''' Get transmission line connected to bus per scenario '''
        aux = []
        for xb in self.__data['bus'][xn].get_transmission_line_to():  # Transmission lines
            # connected to the bus
            # Is the transmission line active in the scenario?
            if self.__data['transmissionline'][xb].get_N1(xs) is not None:
                aux.append(self.__data['transmissionline'][xb].get_N1(xs))
        return aux

    def get_gen_in_bus(self, bus):
        ''' Get list of generators connected to a bus '''
        aux = []
        for xt, xp in zip(bus.__data['GenType'], bus.__data['GenPosition']):
            aux.append(getattr(self, self.__data['GenTypes'][xt])[xp].get_GenNumber())
        return aux
    
    def get_gen_types(self):
        ''' Get types of generator in the network '''
        return self.__data['GenTypes']

    def get_no_buses(self):
        ''' Get total number of buses in the network '''
        return self.__data['NoBuses']

    def get_no_conv(self):
        ''' Get Number of conv units '''
        return self.__data['NoConv']
    
    def get_no_gen(self):
        ''' Get Number of generation units '''
        return self.__data['NoGen']

    def get_no_hydro(self):
        ''' Get Number of hydro units '''
        return self.__data['NoHydro']

    def get_no_renewable(self):
        ''' Get Number of RES units '''
        return self.__data['NoRES']
    
    def get_no_three_winding_trafos(self):
        ''' Get total number of three winding transformers in the network '''
        return self.__data['ThWtrafos']
    
    def get_no_transmission_lines(self):
        ''' Get total number of transmission lines in the network '''
        return self.__data['NoTLines']
    
    def get_no_two_winding_trafos(self):
        ''' Get total number of two winding transformers in the network '''
        return self.__data['TWtrafos']
    
    def get_pos_all_nodes(self):
        ''' Get the position of all nodes in the power system'''
        aux = []
        for xn in range(self.__data['NoBuses']):
            aux.append(self.__data['bus'][xn].get_pos())
        return aux

    def set_base_power(self, val=None):
        ''' Set Base Power '''
        assert val is not None, "No value passed to set the Base Power"
        self.__data['baseMVA'] = val

    def set_bus_data(self, ob=None):
        ''' set the __data of the Bus object '''
        assert isinstance(ob, list), "Bus __data is not an empty list"
        assert isinstance(ob[0], Bus), "Incorrect object passed to set the \
            Bus __data"
        self.__data['bus'] = ob
        self.__set_no_buses(len(self.__data['bus']))

    def set_conv_data(self, ob=None):
        ''' set the __data of the conventional generator object '''
        assert isinstance(ob, list), "Conventional generator __data is not \
            an empty list"
        assert isinstance(ob[0], Conventional), "Incorrect object passed \
            to set the conventional generator __data"
        self.__data['conv'] = ob
        self.__set_no_conv(len(self.__data['conv']))

    def set_electricity_network_data(self, ob=None):
        ''' set the __data of the electricity network object '''
        assert isinstance(ob, ElectricityNetwork), "Incorrect object \
            passed to set the Electricity Network __data"
        self.set_bus_data(ob.__data['bus'])
        self.set_transmission_line_data(ob.__data['transmissionline'])
        if len(ob.__data['conv']) > 0:
            self.set_conv_data(ob.__data['conv'])
        if len(ob.__data['hydro']) > 0:
            self.set_hydro_data(ob.__data['hydro'])
        if len(ob.__data['RES']) > 0:
            self.set_renewable_data(ob.__data['RES'])

    def set_hydro_data(self, ob=None):
        ''' set the data of the conventional generator object '''
        assert isinstance(ob, list), "Hydroelectrical generator data is \
            not an empty list"
        assert isinstance(ob[0], Hydropower), "Incorrect object passed to \
            set the hydro generator data"
        self.__data['hydro'] = ob
        self.__set_no_hydro(len(self.__data['hydro']))

    def set_list_gen_types(self, lt=None):
        ''' Set list of generator types '''
        assert isinstance(lt, list), "The passed value is \
            not an list for GenTypes"
        self.__data['GenTypes'] = lt

    def set_list_security(self, lt=None):
        ''' Set list of N-1 cases to consider '''
        assert isinstance(lt, list), "The passed value is \
            not an list for Security"
        self.__data['Security'] = lt
        self.__set_no_security(len(self.__data['Security']))

    def set_renewable_data(self, ob=None):
        ''' set the data of the conventional generator object '''
        assert isinstance(ob, list), "RES generator data is not an empty \
            list"
        assert isinstance(ob[0], RES), "Incorrect object passed to set \
            the RES generator data"
        self.__data['RES'] = ob
        self.__set_no_renewable(len(self.__data['RES']))

    def set_slack_bus(self, val=None):
        ''' Set Slack Bus '''
        assert val is not None, "No value passed to set the Slack Node"
        self.__data['Slack'] = val
    
    def set_three_winding_trafos_data(self, ob=None):
        ''' set the data of the three winding transformer object '''
        assert isinstance(ob, list), "Three winding transformer data is not an \
            empty list"
        assert isinstance(ob[0], ThreeWindingTrafo), "Incorrect object passed \
            to set the three winding transformer data"
        self.__data['threewindingtrafo'] = ob
        self.__set_no_three_winding_trafos(len(self.__data['threewindingtrafo']))
    
    def set_transmission_line_data(self, ob=None):
        ''' set the data of the transmission line object '''
        assert isinstance(ob, list), "Transmission line data is not an empty \
            list"
        assert isinstance(ob[0], TransmissionLine), "Incorrect object passed \
            to set the transmission line data"
        self.__data['transmissionline'] = ob
        self.__set_no_transmission_lines(len(self.__data['transmissionline']))
    
    def set_two_winding_trafos_data(self, ob=None):
        ''' set the data of the two winding transformer object '''
        assert isinstance(ob, list), "Two winding transformer data is not an \
            empty list"
        assert isinstance(ob[0], TwoWindingTrafo), "Incorrect object passed \
            to set the two winding transformer data"
        self.__data['twowindingtrafo'] = ob
        self.__set_no_two_winding_trafos(len(self.__data['twowindingtrafo']))
    
    def update_all_positions(self):
        ''' Update the position of all nodes, transmission lines, etc. '''
        self.__update_pos_nodes()
        self.__update_pos_generators()
        self.__update_pos_transmission_lines()
        self.__update_pos_two_winding_trafos()
        self.__update_pos_three_winding_trafos()

    def __set_no_buses(self, val=None):
        ''' Set Number of NoBuses '''
        assert val is not None, "No value passed to set the Number of buses"
        self.__data['NoBuses'] += val

    def __set_no_conv(self, val=None):
        ''' Set Number of conv units '''
        assert val is not None, "No value passed to set the Number of \
            Conventional generators"
        self.__data['NoConv'] += val
        self.__data['NoGen'] += val

    def __set_no_hydro(self, val=None):
        ''' Set Number of hydro units '''
        assert val is not None, "No value passed to set the Number of \
            Hydroelectrical generators"
        self.__data['NoHydro'] += val
        self.__data['NoGen'] += val

    def __set_no_renewable(self, val=None):
        ''' Set Number of RES units '''
        assert val is not None, "No value passed to set the Number of \
            RES generators"
        self.__data['NoRES'] += val
        self.__data['NoGen'] += val

    def __set_no_security(self, val=None):
        ''' Set Number of RES units '''
        assert val is not None, "No value passed to set the Number of \
            securities (contingencies)"
        self.__data['SecurityNo'] += val
    
    def __set_no_three_winding_trafos(self, val=None):
        ''' Set Number of three winding transformers '''
        assert val is not None, "No value passed to set the Number of \
            three winding transformers"
        self.__data['ThWtrafos'] += val
    
    def __set_no_transmission_lines(self, val=None):
        ''' Set Number of transmission lines '''
        assert val is not None, "No value passed to set the Number of \
            transmission lines"
        self.__data['NoTLines'] += val
    
    def __set_no_two_winding_trafos(self, val=None):
        ''' Set Number of two winding transformers '''
        assert val is not None, "No value passed to set the Number of \
            two winding transformers"
        self.__data['TWtrafos'] += val
    
    def __update_bus_pos_generators(self):
        '''Update the position of the nodes in all generators'''
        for xn in self.__data['bus']:
            for xgt in self.__data['GenTypes']:
                for xg in self.__data[xgt]:
                    if xn.get_number() == xg.get_bus():
                        xg.set_pos(xn.get_pos())
    
    def __update_generator_pos_buses(self):
        '''Update the position of the generators in all buses'''
        for xn in self.__data['bus']:
            aux = []
            aux1 = []
            for xgt in self.get_gen_types():
                for xg in self.__data[xgt]:
                    if xn.get_pos() == xg.get_bus_pos():
                        aux.append(xgt)
                        aux1.append(xg.get_pos())
            xn.set_gen_type(aux)
            xn.set_gen_pos(aux1)
    
    def __update_three_winding_trafos_pos_buses(self):
        '''Update the position of three winding transformers in all buses'''
        for xn in self.__data['bus']:
            aux = []
            aux1 = []
            aux2 = []
            for xthwt in self.__data['threewindingtrafo']:
                if xn.get_pos() == xthwt.get_pos_bus1():
                    aux.append(xthwt.get_pos())
                elif xn.get_pos() == xthwt.get_pos_bus2():
                    aux1.append(xthwt.get_pos())
                elif xn.get_pos() == xthwt.get_pos_bus3():
                    aux2.append(xthwt.get_pos())
            xn.set_three_trafo_end1(aux)
            xn.set_three_trafo_end2(aux1)
            xn.set_three_trafo_end3(aux2)
    
    def __update_transmission_lines_pos_buses(self):
        '''Update the position of  transmission lines in all buses'''
        for xn in self.__data['bus']:
            aux = []
            aux1 = []
            for xl in self.__data['transmissionline']:
                if xn.get_pos() == xl.get_pos_from():
                    aux.append(xl.get_pos())
                elif xn.get_pos() == xl.get_pos_to():
                    aux1.append(xl.get_pos())
            xn.set_transmission_line_from(aux)
            xn.set_transmission_line_to(aux1)
    
    def __update_two_winding_trafos_pos_buses(self):
        '''Update the position of two winding trafos in all buses'''
        for xn in self.__data['bus']:
            aux = []
            aux1 = []
            for xtwt in self.__data['twowindingtrafo']:
                if xn.get_pos() == xtwt.get_pos_from():
                    aux.append(xtwt.get_pos())
                elif xn.get_pos() == xtwt.get_pos_to():
                    aux1.append(xtwt.get_pos())
            xn.set_two_trafo_from(aux)
            xn.set_two_trafo_to(aux1)
        
    def __update_pos_ends_three_winding_trafos(self):
        '''Update the position of the nodes in both ends of the three winding
        transformer'''
        for xn in self.__data['bus']:
            for xthwt in self.__data['threewindingtrafo']:
                if xn.get_number() == xthwt.get_number_bus1():
                    xthwt.set_pos_bus1(xn.get_pos())
                elif xn.get_number() == xthwt.get_number_bus2():
                    xthwt.set_pos_bus2(xn.get_pos())
                elif xn.get_number() == xthwt.get_number_bus3():
                    xthwt.set_pos_bus3(xn.get_pos())
    
    def __update_pos_ends_transmission_lines(self):
        '''Update the position of the nodes in both ends of the transmission 
        line'''
        for xn in self.__data['bus']:
            for xl in self.__data['transmissionline']:
                if xn.get_number() == xl.get_bus_from():
                    xl.set_pos_from(xn.get_pos())
                elif xn.get_number() == xl.get_bus_to():
                    xl.set_pos_to(xn.get_pos())
    
    def __update_pos_ends_two_winding_trafos(self):
        '''Update the position of the nodes in both ends of the two winding
        transformer'''
        for xn in self.__data['bus']:
            for xtwt in self.__data['twowindingtrafo']:
                if xn.get_number() == xtwt.get_bus_from():
                    xtwt.set_pos_from(xn.get_pos())
                elif xn.get_number() == xtwt.get_bus_to():
                    xtwt.set_pos_to(xn.get_pos())

    def __update_pos_generators(self):
        ''' Update the position of the generators - starting from zero '''
        for xgt in self.get_gen_types():
            aux=0
            for xg in self.__data[xgt]:
                xg.set_pos(aux)
                aux += 1
        self.__update_generator_pos_buses()

    def __update_pos_nodes(self):
        '''Update the position of the nodes - starting from zero'''
        aux=0
        for xn in self.__data['bus']:
            xn.set_pos(aux)
            aux += 1
        self.__update_pos_ends_transmission_lines()
        self.__update_pos_ends_two_winding_trafos()
        self.__update_pos_ends_three_winding_trafos()
        self.__update_bus_pos_generators()
    
    def __update_pos_three_winding_trafos(self):
        ''' Update the position of three winding trafos - starting from zero '''
        aux=0
        for xl in self.__data['threewindingtrafo']:
            xl.set_pos(aux)
            aux += 1
        self.__update_three_winding_trafos_pos_buses()
    
    def __update_pos_transmission_lines(self):
        ''' Update the position of transmission lines - starting from zero '''
        aux=0
        for xl in self.__data['transmissionline']:
            xl.set_pos(aux)
            aux += 1
        self.__update_transmission_lines_pos_buses()
    
    def __update_pos_two_winding_trafos(self):
        ''' Update the position of two winding trafos - starting from zero '''
        aux=0
        for xl in self.__data['twowindingtrafo']:
            xl.set_pos(aux)
            aux += 1
        self.__update_two_winding_trafos_pos_buses()

class GenClass:
    ''' Core generation class '''
    def __init__(self):
        # Basic __data
        aux = ['Ancillary', 'APF' , 'Bus', 'MBASE', 'PC1', 'PC2',
               'PG', 'PMAX', 'PMIN', 'QC1MIN', 'QC1MAX', 'QC2MIN', 'QC2MAX',
               'QG', 'QMAX', 'QMIN', 'Ramp', 'RAMP_AGC', 'RAMP_10', 'RAMP_30',
               'RAMP_Q', 'RES', 'VG', 'MDT', 'MUT', 'Baseload',
               'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP', 'Position', 'NoPieces',
               'BusPosition', 'UniCost', 'Uncertainty', 'GenNumber']
        self.__data = {}
        for x in aux:
            self.__data[x] = None
        
        aux =  ['COST', 'LCost']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self.__data.update(__data2)
        del __data2

    def get_active_power(self):
        ''' Get power output '''
        return self.__data['PG']

    def get_bus(self):
        ''' Get bus number '''
        return self.__data['Bus']

    def get_bus_pos(self):
        ''' Get bus position '''
        return self.__data['BusPosition']

    def get_gen_number(self):
        ''' Get generator number among all generators'''
        return self.__data['GenNumber']

    def get_max_active_power(self):
        ''' Get maximum capacity (MW) '''
        return self.__data['PMAX']

    def get_min_active_power(self):
        ''' Get minimum capacity (MW) '''
        return self.__data['PMIN']

    def get_no_pieces(self):
        ''' Get number of pieces used for piece-wise cost estimations '''
        return self.__data['NoPieces']

    def get_pos(self):
        ''' Get generator position '''
        return self.__data['Position']

    def get_reactive_power(self):
        ''' Get reactive power output '''
        return self.__data['QG']

    def get_uni_cost(self):
        ''' Return coefficient cost of linear generation cost'''
        return self.__data['UniCost']

    def get_voltage_generator(self):
        ''' Get voltage magnitude'''
        return self.__data['VG']
    
    def set_bus_pos(self, xb):
        ''' Set position of the bus '''
        self.__data['BusPosition'] = xb

    def set_cost_curve(self, nopieces, a, b):
        ''' Set parameters for piece wise cost curve approximation '''
        self.__data['LCost'] = np.zeros((nopieces, 2), dtype=float)
        for xv in range(nopieces):
            self.__data['LCost'][xv][0] = a[xv]
            self.__data['LCost'][xv][1] = b[xv]
        self.__data['NoPieces'] = nopieces
    
    def set_gen_number(self, val):
        ''' Set generator number among all generators'''
        self.__data['GenNumber'] = val

    def set_max_active_power(self, val):
        ''' Set maximum capacity (MW) '''
        self.__data['PMAX'] = val

    def set_min_active_power(self, val):
        ''' Set minimum capacity (MW) '''
        self.__data['PMIN'] = val
    
    def set_pos(self, xg):
        ''' Set generator position '''
        self.__data['Position'] = xg

    def set_ramp(self, val):
        ''' Set Ramps'''
        self.__data['Ramp'] = val

    def set_uni_cost(self, val):
        ''' Set coefficient cost of linear generation cost'''
        self.__data['UniCost'] = val

class Conventional(GenClass):
    ''' Conventional generator '''
    def __init__(self):
        self.cooling = {}
        self.cooling['Flag'] = False
        self.cooling['Gen2Der'] = []
        self.cooling['Derate'] = None
        super().__init__()

class Hydropower(GenClass):
    ''' Hydropower generator '''
    def __init__(self):
        super().__init__()

class RES(GenClass):
    ''' RES generation '''
    def __init__(self):
        super().__init__()

class ThreeWindingTrafo():
    ''' Three winding transformer object '''
    def __init__(self):
        aux = ['Bus1', 'Bus2', 'Bus3', 'Bus1_Position', 'Bus2_Position',
            'Bus3_Position', 'R1-2', 'X1-2', 'SBase1-2', 'R2-3', 'X2-3',
            'SBase2-3', 'R3-1', 'X3-1', 'SBase3-1', "TAP1", 'VBase1', 'ANG1',
            'ANGMAX1', 'ANGMIN1', "TAP2", 'VBase2', 'ANG2', 'ANGMAX2',
            'ANGMIN2', "TAP3", 'VBase3', 'ANG3', 'ANGMAX3', 'ANGMIN3',
            'RATE_A1', 'RATE_B1', 'RATE_C1', 'RATE_A2', 'RATE_B2', 'RATE_C2',
            'RATE_A3', 'RATE_B3', 'RATE_C3', 'Loss_Fix', 
            'N-1', 'STATUS1', 'STATUS2', 'STATUS3', 'YMag', 'Position']
        self.__data = {}
        for x in aux:
            self.__data[x] = None
    
    def get_number_bus1(self):
        ''' Get number position at end 1 of the trafo '''
        return self.__data['Bus1']
    
    def get_number_bus2(self):
        ''' Get number position at end 2 of the trafo '''
        return self.__data['Bus2']
    
    def get_number_bus3(self):
        ''' Get number position at end 3 of the trafo '''
        return self.__data['Bus3']
    
    def get_pos(self):
        ''' Get position of the trafo - starting from zero'''
        return self.__data['Position']
    
    def get_pos_bus1(self):
        ''' Get bus position at end 1 of the trafo '''
        return self.__data['Bus1_Position']
    
    def get_pos_bus2(self):
        ''' Get bus position at end 2 of the trafo '''
        return self.__data['Bus2_Position']
    
    def get_pos_bus3(self):
        ''' Get bus position at end 3 of the trafo '''
        return self.__data['Bus3_Position']
    
    def set_pos(self, val=None):
        ''' Set Position of the trafo on the list of trafos - starting from zero'''
        assert val is not None, "No value passed for the position of the \
            three winding trafo"
        self.__data['Position'] = val
    
    def set_pos_bus1(self, val=None):
        ''' Set bus position at end 1 of the trafo '''
        assert val is not None, "No value passed for the position of the bus \
        at end 1"
        self.__data['Bus1_Position'] = val
    
    def set_pos_bus2(self, val=None):
        ''' Set bus position at end 2 of the trafo '''
        assert val is not None, "No value passed for the position of the bus \
        at end 2"
        self.__data['Bus2_Position'] = val
    
    def set_pos_bus3(self, val=None):
        ''' Set bus position at end 3 of the trafo '''
        assert val is not None, "No value passed for the position of the bus \
        at end 3"
        self.__data['Bus3_Position'] = val
