"""
A common template of parameters for power systems and temporal trees is provided
in this file. Most of the initial functions have been taken from the file pyeneD
created by Dr Eduardo Alejandro Martínez Ceseña.

@author: Dr. Jose Nicolas Melchor Gutierrez
         Dr Eduardo Alejandro Martínez Ceseña
"""

import numpy as np

'''                               DEVICE CLASSES                            '''

class _CommonMethods():
    def __init__(self):
        ''' This Class contains the methods that are common to some classes '''
        # Assign random value to self.__data in this function. The value will 
        # be overwritten in the child classes
        self.__data = {}

    def get_element(self, name=None, pos=':'):
        ''' This function returns the value or a list of values for the 
        requested parameter "name". 
        - If the parameter is a list then the function returns either a list 
        with the positions indicated in "pos" or the value of the position "pos"
        - "pos" needs to indicate a position between zero and the size of the 
        list
        - If the parameters is a value then "pos" can be ignored'''
        auxp = "No valid name has been passed to the function get_element in \
            the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self.__data:
            if isinstance(self.__data[name], list):
                if pos == ':':
                    return self.__data[name]
                elif isinstance(pos, list):
                    aux = []
                    for aux1 in pos:
                        aux.append(self.__data[name][aux1])
                    return aux
                else:
                    return self.__data[name][pos]
            else:
                return self.__data[name]
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_element in the class {1}. The valid keys for this class are as \
            follows: \n {2}".format(name, self.__class__.__name__, \
            self.__data.keys()) 
            assert " ".join(auxp.split())
    
    def get_no_elements(self, name=None):
        ''' This function returns the number of elements in a list '''
        auxp = "No valid name has been passed to the function get_no_elements \
            in the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self.__data:
            if isinstance(self.__data[name], list):
                return len(self.__data[name])
            else:
                auxp = "the key {0} in the class {1} is not a list".format(\
                    name, self.__class__.__name__)
                assert name is not None, " ".join(auxp.split())
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_no_elements in the class {1}. The valid keys for this class are \
            as follows: \n {2}".format(name, self.__class__.__name__, \
            self.__data.keys()) 
            assert " ".join(auxp.split())
    
    def set_element(self, name=None, val=None):
        ''' This function set the value or a list of values to the 
        requested parameter "name". 
        - This function rewrite the existing values or value stored in the 
        parameter'''
        auxp = "No valid name has been passed to the function get_element in \
            the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self.__data:
            if isinstance(self.__data[name], list):
                auxp = "You are trying to set the \
                parameter {0} that have to be a list with different type of \
                data element in the class {1}".format(name, \
                self.__class__.__name__)
                assert isinstance(val, list), " ".join(auxp.split())
                self.__data[name] = val
            else:
                auxp = "You are trying to set the \
                parameter {0} that have to be a single number or other single type \
                of data with a list in the class {1}".format(name, \
                self.__class__.__name__)
                assert not isinstance(val, list), " ".join(auxp.split())
                self.__data[name] = val
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_element in the class {1}. The valid keys for this class are as \
            follows \n {2}".format(name, self.__class__.__name__, \
            self.__data.keys()) 
            assert " ".join(auxp.split())

class Branch(_CommonMethods):
    ''' Electricity branch - This could be information for either transmission \
        lines or two winding transformers '''
    def __init__(self):
        # Data that have to be single numbers or names
        aux = ['shunt_susceptance', 'resistance', 'status', 'reactance',
               'number', 'long_term_thermal_limit', 
               'short_term_thermal_limit', 'emergency_thermal_limit',
               'non_technical_losses_fix', 'position',
               'contingency_n-1']
        self.__data = {}
        for x in aux:
            self.__data[x] = None

        # Data that have to be a list of numbers or names
        aux =  ['bus_number', 'bus_position']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self.__data.update(__data2)
        del __data2

class TwoWindingTrafo(Branch):
    ''' Two winding transformer class '''
    def __init__(self):
        super().__init__()
        # Basic __data
        aux = ['max_phase_shift_angle', 'min_phase_shift_angle',
        'tap']
        self.__data.update(aux)
        for x in aux:
            self.__data[x] = None

class TransmissionLine(Branch):
    ''' Transmission Line class '''
    def __init__(self):
        super().__init__()

class Bus(_CommonMethods):
    ''' Electricity bus '''
    def __init__(self):
        # Data that have to be single numbers or names
        aux = ['voltage_kv', 'shunt_susceptance', 'area', 'type', 'position_x', 
               'position_y', 'active_power_demand', 'shunt_conductance', 
               'active_power_demand_peak', 'reactive_power_demand_peak', 
               'position', 'name', 'number','initial_voltage_magnitude', 
               'initial_voltage_angle', 'maximum_voltage_magnitude', 
               'minimum_voltage_magnitude', 'zone', 'non_technical_losses_fix']
        self.__data = {}
        for x in aux:
            self.__data[x] = None

        # Data that have to be a list of numbers or names
        aux =  ['contingency_n-1', 'load_type', 'from_end_transmission_lines',
                'to_end_transmission_lines', 'gen_type','gen_position',
                'from_end_two_winding_trafo', 'to_end_two_winding_trafo',
                'high_voltage_end_three_winding_trafo', 
                'medium_voltage_end_three_winding_trafo',
                'low_voltage_end_three_winding_trafo']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self.__data.update(__data2)
        del __data2
    
    def update_gen_pos(self, poslt=None, newpos=None):
        ''' Update list of generator positions for each type connected to
        the bus '''
        assert poslt != None and newpos!=None, "Some values have not been \
            passed to set the GenPosition"
        self.__data['GenPosition'][poslt] = newpos

class ElectricityNetwork(_CommonMethods):
    ''' Electricity network '''
    def __init__(self, **kwargs):
        ''' General electricity network settings 
        In case the user wants to initialise some or all the objects then \
        the some or all the following arguments have to be passed in kwargs:
        'nobuses' -> Number of buses
        'noconv' -> Number of conventional generators
        'nohydro' -> Number of hydro generators
        'nores' -> Number of RES generators
        'notlines' -> Number of transmission lines
        'notwtrafos' -> Number of two winding transformers
        'nothwtrafos' -> Number of three winding transformers
        '''
        self.__data = {
            'baseMVA': None,
            'Slack': None,
            'Security': [],  # list of N-1 cases to consider
            }
        
        self.__objects = {
            'bus' : [], # list of Bus objects
            'transmissionline' : [], # list of transmission line objects
            'conv' : [], # list of conventional generator objects
            'hydro' : [], # list of hydro generator objects
            'RES' : [], # list of RES generator objects
            'twowindingtrafo' : [], # list of two winding trafo objects
            'threewindingtrafo' : [] # list of two winding trafo objects
            }

        # Initialise bus object
        if 'nobuses' in kwargs.keys():
            self.__objects['bus'] = [Bus() for _ in
                range(kwargs.pop('nobuses'))]
        else:
            self.__objects['bus'] = [Bus()]

        # Initialise Conventional generator object
        if 'noconv' in kwargs.keys():
            self.__objects['conv'] = [Conventional() for _ in
                range(kwargs.pop('noconv'))]
        else:
            self.__objects['conv'] = [Conventional()]

        # Initialise hydro-electrical generator object
        if 'nohydro' in kwargs.keys():
            self.__objects['hydro'] = [Hydropower() for _ in
                range(kwargs.pop('nohydro'))]
        else:
            self.__objects['hydro'] = [Hydropower()]

        # Initialise RES generator object
        if 'nores' in kwargs.keys():
            self.__objects['RES'] = [RES() for _ in
                range(kwargs.pop('nores'))]
        else:
            self.__objects['RES'] = [RES()]
                
        # Initialise transmission line object
        if 'notlines' in kwargs.keys():
            self.__objects['transmissionline'] = [TransmissionLine() for _ in
                    range(kwargs.pop('notlines'))]
        else:
            self.__objects['transmissionline'] = [TransmissionLine()]

        # Initialise three winding transformer object
        if 'nothwtrafos' in kwargs.keys():
            self.__objects['threewindingtrafo'] = [ThreeWindingTrafo() for _ in
                range(kwargs.pop('nothwtrafos'))]
        else:
            self.__objects['threewindingtrafo'] = [ThreeWindingTrafo()]

        # Initialise two winding transformer object
        if 'notwtrafos' in kwargs.keys():
            self.__objects['twowindingtrafo'] = [TwoWindingTrafo() for _ in
                range(kwargs.pop('notwtrafos'))]
        else:
            self.__objects['twowindingtrafo'] = [TwoWindingTrafo()]
    
    def get_objects(self, name=None, pos=':'):
        ''' This function returns an object or a list of objects for the 
        requested object "name". 
        - The function returns either a list with the positions indicated 
        in "pos" or the object of the position "pos"
        - "pos" needs to indicate a position between zero and the size of the 
        list
        - The function returns by default the whole list of objects'''
        auxp = "No valid name has been passed to the function get_objects in \
            the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self.__objects:
            if pos == ':':
                return self.__objects[name]
            elif isinstance(pos, list):
                aux = []
                for aux1 in pos:
                    aux.append(self.__objects[name][aux1])
                return aux
            else:
                return self.__objects[name][pos]
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_objects in the class {1}. The valid keys for this class are as \
            follows: \n {2}".format(name, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())
    
    def get_objects_elements(self, name_object=None, name_element, pos=':'):
        ''' This function returns the value or a list of values for the 
        requested parameter "name_element" in the object "name_obj". 
        - If the parameter is a list then the function returns either a list 
        with the positions indicated in "pos" or the value of the position "pos"
        - "pos" needs to indicate a position between zero and the size of the 
        list
        - If the parameters is a value then "pos" can be ignored

        - The function returns either a list with the positions indicated 
        in "pos" or the object of the position "pos"
        - "pos" needs to indicate a position between zero and the size of the 
        list
        - The function returns by default the whole list of objects'''
        auxp = "No valid name has been passed to the function get_objects in \
            the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self.__objects:
            if pos == ':':
                return self.__objects[name]
            elif isinstance(pos, list):
                aux = []
                for aux1 in pos:
                    aux.append(self.__objects[name][aux1])
                return aux
            else:
                return self.__objects[name][pos]
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_objects in the class {1}. The valid keys for this class are as \
            follows: \n {2}".format(name, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())
    
    def set_objects(self, name=None, list_obj=None):
        ''' This function set a list of a specific object indicated in "name". 
        - This function rewrite the list of a specific object '''
        auxp = "No valid name has been passed to the function set_element in \
            the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        auxp = "No valid list of objects has been passed to the function \
            set_element in \
            the class {0}".format(self.__class__.__name__)
        assert list_obj is not None, " ".join(auxp.split())
        if name in self.__objects:
            if isinstance(self.__objects[name], list):
                auxp = "You are trying to set the list of\
                    objects {0} with different type of \
                    data element in the class {1}".format(name, \
                    self.__class__.__name__)
                assert isinstance(list_obj, list), " ".join(auxp.split())
                auxp = "The objects inside the list are not instances of the \
                    the class {0}".format(\
                    self.__objects[name][0].__class__.__name__)
                assert isinstance(list_obj[0], \
                    self.__objects[name][0].__class__.__name__), \
                    " ".join(auxp.split())
                self.__objects[name] = list_obj
        else:
            auxp = "No valid key {0} has been passed to the function \
            set_objects in the class {1}. The valid keys for this class are as \
            follows \n {2}".format(name, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())

    def set_electricity_network_data(self, ob=None):
        ''' set the data of the electricity network object '''
        assert isinstance(ob, ElectricityNetwork), "Incorrect object \
            passed to set the Electricity Network data"
        for xkey in self.__objects.keys():
            self.set_objects(name=xkey, list_obj=ob.get_objects(name=xkey))
        for xkey in self.__data.keys():
            self.set_element(name=xkey,val=ob.get_element(name=xkey))
    
    def update_all_positions(self):
        ''' Update the position of all nodes, transmission lines, etc. '''
        self.__update_pos_nodes()
        self.__update_pos_generators()
        self.__update_pos_transmission_lines()
        self.__update_pos_two_winding_trafos()
        self.__update_pos_three_winding_trafos()
    
    def __update_bus_pos_conv(self):
        '''Update the position of the nodes in all conventional generators'''
        for xn in self.__objects['bus']:
            for xg in self.__objects['conv']:
                if xn.get_element(name='number') == get_element(name='number'):
                    xg.set_pos(xn.get_pos())

    def __update_bus_pos_hydro(self):
        '''Update the position of the nodes in all hydro generators'''
        for xn in self.__objects['bus']:
            for xg in self.__objects['hydro']:
                if xn.get_number() == xg.get_bus():
                    xg.set_pos(xn.get_pos())
    
    def __update_bus_pos_RES(self):
        '''Update the position of the nodes in all RES generators'''
        for xn in self.__objects['bus']:
            for xg in self.__objects['RES']:
                if xn.get_number() == xg.get_bus():
                    xg.set_pos(xn.get_pos())
    
    def __update_generator_pos_buses(self):
        '''Update the position of the generators in all buses'''
        for xn in self.__data['bus']:
            aux = []
            aux1 = []
            for xgt in self.__data['GenTypes']:
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
        for xn in self.__objects['bus']:
            for xthwt in self.__objects['threewindingtrafo']:
                if xn.get_number() == xthwt.get_number_bus1():
                    xthwt.set_pos_bus1(xn.get_pos())
                elif xn.get_number() == xthwt.get_number_bus2():
                    xthwt.set_pos_bus2(xn.get_pos())
                elif xn.get_number() == xthwt.get_number_bus3():
                    xthwt.set_pos_bus3(xn.get_pos())
    
    def __update_pos_ends_transmission_lines(self):
        '''Update the position of the nodes in both ends of the transmission 
        line'''
        for xn in self.__objects['bus']:
            for xl in self.__objects['transmissionline']:
                if xn.get_number() == xl.get_bus_from():
                    xl.set_pos_from(xn.get_pos())
                elif xn.get_number() == xl.get_bus_to():
                    xl.set_pos_to(xn.get_pos())
    
    def __update_pos_ends_two_winding_trafos(self):
        '''Update the position of the nodes in both ends of the two winding
        transformer'''
        for xn in self.__objects['bus']:
            for xtwt in self.__objects['twowindingtrafo']:
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
        for xn in self.__objects['bus']:
            xn.set_pos(aux)
            aux += 1
        for xn in self.__objects['bus']:
            for xobj in self.self.__objects.keys():

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

class GenClass(_CommonMethods):
    ''' Core generation class '''
    def __init__(self):
        # Basic __data
        aux = [ 'maximum_active_power_generation', 
                'minimum_active_power_generation', 
                'maximum_reactive_power_generation',
                'minimum_reactive_power_generation', 'ramp', 'baseload',
                'model', 'shutdown_cost', 'startup_cost', 'position',
                'uncertainty', 'gen_number']
        self.__data = {}
        for x in aux:
            self.__data[x] = None
        
        aux =  ['cost_function_parameters', 'piecewise_linearization_parameters'
                , 'bus_number', 'bus_position']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self.__data.update(__data2)
        del __data2

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

class ThreeWindingTrafo(_CommonMethods):
    ''' Three winding transformer object '''
    def __init__(self):
        # Data that have to be single numbers or names
        aux = ['transformer_magnetizing_admittance', 'Position']
        self.__data = {}
        for x in aux:
            self.__data[x] = None
        
        # Data that have to be a list of numbers or names
        aux =  ['bus_number', 'bus_position', 'resistance_delta', 'tap_delta',
                'base_power_delta', 'reactance_delta', 'voltage_kv',
                'voltage_angle_fix', 'maximum_voltage_angle',
                'minimum_voltage_angle', 'long_term_thermal_limit', 
                'contingency_n-1', 'status']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self.__data.update(__data2)
        del __data2
