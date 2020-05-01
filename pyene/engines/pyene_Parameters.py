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
               'NoFB', 'NoTB', 'GenType', 'NoFTW', 'NoTTW', 'NoF1ThW',
               'NoT1ThW', 'NoF2ThW', 'NoT2ThW', 'NoF3ThW', 'NoT3ThW']
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
    
    def get_transmission_line_from(self):
        ''' Get list of transmission lines connected from the bus '''
        return self.__data['F_TLines']

    def get_transmission_line_to(self):
        ''' Get list of transmission lines connected to the bus '''
        return self.__data['T_TLines']
    
    def get_three_trafo_end1(self):
        ''' Get list of three winding transformers connected the bus at end 1 '''
        return self.__data['End1_ThWTrafo']
    
    def get_three_trafo_end2(self):
        ''' Get list of three winding transformers connected the bus at end 2 '''
        return self.__data['End2_ThWTrafo']
    
    def get_three_trafo_end3(self):
        ''' Get list of three winding transformers connected the bus at end 3 '''
        return self.__data['End3_ThWTrafo']    

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

    def set_security(self, val, x=None):
        ''' Set values for all conditions '''
        if x is None:
            self.__data['N-1'] = val
        else:
            self.__data['N-1'][x] = val

    def set_pos(self, val=None):
        ''' Set Position of Bus on the list of Buses - starting from zero'''
        assert val is not None, "No value passed for the position of the bus"
        self.__data['Position'] = val


class ElectricityNetwork:
    ''' Electricity network '''
    def __init__(self, **kwargs):
        ''' General electricity network settings '''
        self.__data = {
                'baseMVA': None,
                'Slack': None,
                'Buses': 0,  # Number of buses
                'TLines': 0,  # Number of transmission lines
                'Security': [],  # list of N-1 cases to consider
                'SecurityNo': 0,  # Number of N-1 cases
                'Conv' : 0, # Number of conventional generators
                'Hydro' : 0, # Number of Hydro generators
                'RES' : 0, # Number of RES generators
                'NoGen': 0,
                'GenTypes': [],  # Types of generators to be considered
                                # (i.e. conv, RES, hydro, etc)
                'TWtrafos': 0, # Number of two winding transformers
                'ThWtrafos': 0 # Number of three winding transformers
                }
        if 'NoBus' in kwargs.keys():
            self.__data['Buses'] = kwargs.pop('NoBus')
        if 'NoTLines' in kwargs.keys():
            self.__data['TLines'] = kwargs.pop('NoTLines')
        if 'NoConv' in kwargs.keys():
            self.__data['Conv'] = kwargs.pop('NoConv')
        if 'NoHydro' in kwargs.keys():
            self.__data['Hydro'] = kwargs.pop('NoHydro')
        if 'NoRES' in kwargs.keys():
            self.__data['RES'] = kwargs.pop('NoRES')
        if 'NoThWtrafos' in kwargs.keys():
            self.__data['ThWtrafos'] = kwargs.pop('NoThWtrafos')
        if 'NoTWTrafos' in kwargs.keys():
            self.__data['TWtrafos'] = kwargs.pop('NoTWTrafos')

        self.__data['NoGen'] = self.__data['Conv'] + self.__data['Hydro'] + \
            self.__data['RES']

        # Initialise bus object
        self.bus = [Bus() for _ in
                    range(self.__data['Buses'])]

        # Initialise transmission line object
        self.transmissionline = [TransmissionLine() for _ in
                       range(self.__data['TLines'])]

        # Initialise Conventional generator object
        self.conv = [Conventional() for _ in
                       range(self.__data['Conv'])]
        if self.__data['Conv'] > 0: self.__data['GenTypes'].append('Conv')

        # Initialise Hydro-electrical generator object
        self.hydro = [Hydropower() for _ in
                       range(self.__data['Hydro'])]
        if self.__data['Hydro'] > 0: self.__data['GenTypes'].append('Hydro')

        # Initialise RES generator object
        self.RES = [RES() for _ in
                       range(self.__data['RES'])]
        if self.__data['RES'] > 0: self.__data['GenTypes'].append('RES')

        # Initialise three winding transformer object
        self.threewindingtrafo = [ThreeWindingTrafo() for _ in
                    range(self.__data['ThWtrafos'])]

        # Initialise two winding transformer object
        self.twowindingtrafo = [TwoWindingTrafo() for _ in
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
                self.bus.extend(obj)
                self.__data['Buses'] = len(self.bus)
            else:
                assert "No valid object to extend the list of nodes"
        elif isinstance(obj, Bus):
            self.bus.append(obj)
            self.__data['Buses'] += 1
        else:
            assert "No valid object to extend the list of nodes"
        # Updating positions in bus list
        self.__update_pos_nodes()

    def del_nodes(self, lt=None):
        ''' Delete nodes of the list '''
        assert lt is not None, "No list of nodes to delete has been passed"
        if isinstance(lt, list):
            for aux in lt:
                for aux1 in range(len(self.bus)):
                    if aux == self.bus[aux1].get_pos():
                        self.bus.pop(aux1)
                        break
            self.__data['Buses'] -= len(lt)
        else:
            self.bus.pop(lt)
            self.__data['Buses'] -= 1
        # Updating positions in bus list
        self.__update_pos_nodes()

    def get_base(self):
        ''' Provide base MVA rating '''
        return self.__data['baseMVA']

    def get_flow_from(self, xn, xs):
        ''' Get transmission lines connected from bus per scenario '''
        aux = []
        for xb in self.bus[xn].get_transmission_line_from(): # Transmission line  
            # connected to the bus
            # Is the transmission line active in the scenario?
            if self.transmissionline[xb].is_active(xs):
                aux.append(self.transmissionline[xb].get_N1(xs))
        return aux

    def get_flow_to(self, xn, xs):
        ''' Get transmission line connected to bus per scenario '''
        aux = []
        for xb in self.bus[xn].get_transmission_line_to():  # Transmission lines
            # connected to the bus
            # Is the transmission line active in the scenario?
            if self.transmissionline[xb].get_N1(xs) is not None:
                aux.append(self.transmissionline[xb].get_N1(xs))
        return aux

    def get_pos_all_nodes(self):
        ''' Get the position of all nodes in the power system'''
        aux = []
        for xn in range(self.__data['Buses']):
            aux.append(self.bus[xn].get_pos())
        return aux

    def get_gen_in_bus(self, bus):
        ''' Get list of generators connected to a bus '''
        aux = []
        for xt, xp in zip(bus.__data['GenType'], bus.__data['GenPosition']):
            aux.append(getattr(self, self.__data['GenTypes'][xt])[xp].get_GenNumber())
        return aux

    def get_no_buses(self):
        ''' Get total number of buses in the network '''
        return self.__data['Buses']

    def get_no_conv(self):
        ''' Get Number of Conv units '''
        return self.__data['Conv']
    
    def get_no_gen(self):
        ''' Get Number of generation units '''
        return self.__data['NoGen']

    def get_no_hydro(self):
        ''' Get Number of Hydro units '''
        return self.__data['Hydro']

    def get_no_renewable(self):
        ''' Get Number of RES units '''
        return self.__data['RES']
    
    def get_no_three_winding_trafos(self):
        ''' Get total number of three winding transformers in the network '''
        return self.__data['ThWtrafos']
    
    def get_no_transmission_lines(self):
        ''' Get total number of transmission lines in the network '''
        return self.__data['TLines']
    
    def get_no_two_winding_trafos(self):
        ''' Get total number of two winding transformers in the network '''
        return self.__data['TWtrafos']

    def set_base_power(self, val=None):
        ''' Set Base Power '''
        assert val is not None, "No value passed to set the Base Power"
        self.__data['baseMVA'] = val

    def set_bus_data(self, ob=None):
        ''' set the __data of the Bus object '''
        assert isinstance(ob, list), "Bus __data is not an empty list"
        assert isinstance(ob[0], Bus), "Incorrect object passed to set the \
            Bus __data"
        self.bus = ob
        self.__set_no_buses(len(self.bus))

    def set_conv_data(self, ob=None):
        ''' set the __data of the conventional generator object '''
        assert isinstance(ob, list), "Conventional generator __data is not \
            an empty list"
        assert isinstance(ob[0], Conventional), "Incorrect object passed \
            to set the conventional generator __data"
        self.conv = ob
        self.__set_no_conv(len(self.conv))

    def set_electricity_network_data(self, ob=None):
        ''' set the __data of the electricity network object '''
        assert isinstance(ob, ElectricityNetwork), "Incorrect object \
            passed to set the Electricity Network __data"
        self.set_bus_data(ob.bus)
        self.set_transmission_line_data(ob.transmissionline)
        if len(ob.conv) > 0:
            self.set_conv_data(ob.conv)
        if len(ob.hydro) > 0:
            self.set_hydro_data(ob.hydro)
        if len(ob.RES) > 0:
            self.set_renewable_data(ob.RES)

    def set_hydro_data(self, ob=None):
        ''' set the data of the conventional generator object '''
        assert isinstance(ob, list), "Hydroelectrical generator data is \
            not an empty list"
        assert isinstance(ob[0], Hydropower), "Incorrect object passed to \
            set the Hydro generator data"
        self.hydro = ob
        self.__set_no_hydro(len(self.hydro))

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
        self.RES = ob
        self.__set_no_renewable(len(self.RES))

    def set_slack_bus(self, val=None):
        ''' Set Slack Bus '''
        assert val is not None, "No value passed to set the Slack Node"
        self.__data['Slack'] = val
    
    def set_transmission_line_data(self, ob=None):
        ''' set the data of the transmission line object '''
        assert isinstance(ob, list), "Transmission line data is not an empty \
            list"
        assert isinstance(ob[0], TransmissionLine), "Incorrect object passed \
            to set the transmission line data"
        self.transmissionline = ob
        self.__set_no_transmission_lines(len(self.transmissionline))

    def __set_no_transmission_lines(self, val=None):
        ''' Set Number of transmission lines '''
        assert val is not None, "No value passed to set the Number of \
            transmission lines"
        self.__data['TLines'] += val

    def __set_no_buses(self, val=None):
        ''' Set Number of Buses '''
        assert val is not None, "No value passed to set the Number of Buses"
        self.__data['Buses'] += val

    def __set_no_conv(self, val=None):
        ''' Set Number of Conv units '''
        assert val is not None, "No value passed to set the Number of \
            Conventional generators"
        self.__data['Conv'] += val
        self.__data['NoGen'] += val

    def __set_no_hydro(self, val=None):
        ''' Set Number of Hydro units '''
        assert val is not None, "No value passed to set the Number of \
            Hydroelectrical generators"
        self.__data['Hydro'] += val
        self.__data['NoGen'] += val

    def __set_no_renewable(self, val=None):
        ''' Set Number of RES units '''
        assert val is not None, "No value passed to set the Number of \
            RES generators"
        self.__data['RES'] += val
        self.__data['NoGen'] += val

    def __set_no_security(self, val=None):
        ''' Set Number of RES units '''
        assert val is not None, "No value passed to set the Number of \
            securities (contingencies)"
        self.__data['SecurityNo'] += val
    
    def __update_bus_pos_generators(self):
        '''Update the position of the nodes in all generators'''
        for xn in self.bus:
            for xg, xp in zip(xn.get_gen_type(), xn.get_gen_pos()):
                print(xg)
                getattr(self, xg)[xp].set_bus_pos(xn.get_pos())
    
    def __update_pos_ends_three_winding_trafos(self):
        '''Update the position of the nodes in both ends of the three winding
        transformer'''
        for xn in self.bus:
            for xthwt in xn.get_three_trafo_end1():
                self.threewindingtrafo[xthwt].set_pos_bus1(xn.get_pos())
            for xthwt in xn.get_three_trafo_end2():
                self.threewindingtrafo[xthwt].set_pos_bus2(xn.get_pos())
            for xthwt in xn.get_three_trafo_end3():
                self.threewindingtrafo[xthwt].set_pos_bus3(xn.get_pos())
    
    def __update_pos_ends_transmission_lines(self):
        '''Update the position of the nodes in both ends of the transmission 
        line'''
        for xn in self.bus:
            for xb in xn.get_transmission_line_from():
                self.transmissionline[xb].set_pos_from(xn.get_pos())
            for xb in xn.get_transmission_line_to():
                self.transmissionline[xb].set_pos_to(xn.get_pos())
    
    def __update_pos_ends_two_winding_trafos(self):
        '''Update the position of the nodes in both ends of the two winding
        transformer'''
        for xn in self.bus:
            for xtwt in xn.get_two_trafo_from():
                self.twowindingtrafo[xtwt].set_pos_from(xn.get_pos())
            for xtwt in xn.get_two_trafo_to():
                self.twowindingtrafo[xtwt].set_pos_to(xn.get_pos())

    def __update_pos_nodes(self):
        '''Update the position of the nodes - starting from zero'''
        oldpos = []
        aux=0
        for xn in self.bus:
            oldpos.append(xn.get_pos())
            xn.set_pos(aux)
            aux += 1
        self.__update_pos_ends_transmission_lines()
        self.__update_pos_ends_two_winding_trafos()
        self.__update_pos_ends_three_winding_trafos()
        self.__update_bus_pos_generators()            


class GenClass:
    ''' Core generation class '''
    def __init__(self):
        # Basic __data
        aux = ['Ancillary', 'APF', 'GEN', 'Bus', 'MBASE', 'PC1', 'PC2',
               'PG', 'PMAX', 'PMIN', 'QC1MIN', 'QC1MAX', 'QC2MIN', 'QC2MAX',
               'QG', 'QMAX', 'QMIN', 'Ramp', 'RAMP_AGC', 'RAMP_10', 'RAMP_30',
               'RAMP_Q', 'RES', 'VG', 'MDT', 'MUT', 'Baseload', 'COST',
               'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP', 'Position', 'NoPieces',
               'LCost', 'BusPosition', 'UniCost', 'Uncertainty', 'GenNumber']
        self.__data = {}
        for x in aux:
            self.__data[x] = None

    def get_Bus(self):
        ''' Get bus number '''
        return self.__data['Bus']

    def get_bus_pos(self):
        ''' Get bus position '''
        return self.__data['BusPosition']

    def get_GenNumber(self):
        ''' Get generator number among all generators'''
        return self.__data['GenNumber']

    def get_PMax(self):
        ''' Get maximum capacity (MW) '''
        return self.__data['PMAX']

    def get_PMin(self):
        ''' Get minimum capacity (MW) '''
        return self.__data['PMIN']

    def get_NoPieces(self):
        ''' Get number of pieces used for piece-wise cost estimations '''
        return self.__data['NoPieces']

    def get_P(self):
        ''' Get power output '''
        return self.__data['PG']

    def get_Q(self):
        ''' Get reactive power output '''
        return self.__data['QG']

    def get_Pos(self):
        ''' Get generator position '''
        return self.__data['Position']

    def get_UniCost(self):
        ''' Return coefficient cost of linear generation cost'''
        return self.__data['UniCost']

    def get_VG(self):
        ''' Get voltage magnitude'''
        return self.__data['VG']

    def set_GenNumber(self, val):
        ''' Set generator number among all generators'''
        self.__data['GenNumber'] = val

    def set_CostCurve(self, NoPieces, A, B):
        ''' Set parameters for piece wise cost curve approximation '''
        self.__data['LCost'] = np.zeros((NoPieces, 2), dtype=float)
        for xv in range(NoPieces):
            self.__data['LCost'][xv][0] = A[xv]
            self.__data['LCost'][xv][1] = B[xv]

        self.__data['NoPieces'] = NoPieces

    def set_PMax(self, val):
        ''' Set maximum capacity (MW) '''
        self.__data['PMAX'] = val

    def set_PMin(self, val):
        ''' Set minimum capacity (MW) '''
        self.__data['PMIN'] = val

    def set_bus_pos(self, xb):
        ''' Set position of the bus '''
        self.__data['BusPosition'] = xb

    def set_Ramp(self, val):
        ''' Set Ramps'''
        self.__data['Ramp'] = val

    def set_UniCost(self, val):
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
            'RATE_A3', 'RATE_B3', 'RATE_C3', 'Loss_Fix1', 'Loss_Fix2',
            'Loss_Fix3', 'N-1', 'STATUS1', 'STATUS2', 'STATUS3', 'YMag']
        self.__data = {}
        for x in aux:
            self.__data[x] = None
    
    def get_pos_bus1(self):
        ''' Get bus position at end 1 of the trafo '''
        return self.__data['Bus1_Position']
    
    def get_pos_bus2(self):
        ''' Get bus position at end 2 of the trafo '''
        return self.__data['Bus2_Position']
    
    def get_pos_bus3(self):
        ''' Get bus position at end 3 of the trafo '''
        return self.__data['Bus3_Position']
    
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
