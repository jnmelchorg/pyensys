"""
A common template of parameters for power systems and temporal trees is provided
in this file. Most of the initial functions have been taken from the file pyeneD
created by Dr Eduardo Alejandro Martínez Ceseña.

@author: Dr. Jose Nicolas Melchor Gutierrez
         Dr Eduardo Alejandro Martínez Ceseña
"""

import copy

'''                               DEVICE CLASSES                            '''

class _CommonMethods():
    def __init__(self):
        ''' This Class contains the methods that are common to some classes '''
        # Assign random value to self._data in this function. The value will 
        # be overwritten in the child classes
        self._data = {}

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
        if name in self._data:
            if isinstance(self._data[name], list):
                if pos == ':':
                    return self._data[name]
                elif isinstance(pos, list):
                    aux = []
                    for aux1 in pos:
                        aux.append(self._data[name][aux1])
                    return aux
                else:
                    return self._data[name][pos]
            else:
                return self._data[name]
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_element in the class {1}. The valid keys for this class are as \
            follows: \n {2}".format(name, self.__class__.__name__, \
            self._data.keys()) 
            assert " ".join(auxp.split())
    
    def get_no_elements(self, name=None):
        ''' This function returns the number of elements in a list '''
        auxp = "No valid name has been passed to the function get_no_elements \
            in the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self._data:
            if isinstance(self._data[name], list):
                return len(self._data[name])
            else:
                auxp = "the key {0} in the class {1} is not a list".format(\
                    name, self.__class__.__name__)
                assert name is not None, " ".join(auxp.split())
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_no_elements in the class {1}. The valid keys for this class are \
            as follows: \n {2}".format(name, self.__class__.__name__, \
            self._data.keys()) 
            assert " ".join(auxp.split())
    
    def set_element(self, name=None, val=None):
        ''' This function set the value or a list of values ("val") to the 
        requested parameter "name". 
        - This function rewrites the existing values or value stored in the 
        parameter "name" '''
        auxp = "No valid name has been passed to the function get_element in \
            the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self._data:
            if isinstance(self._data[name], list):
                auxp = "You are trying to set the \
                parameter {0} that have to be a list with different type of \
                data element in the class {1}".format(name, \
                self.__class__.__name__)
                assert isinstance(val, list), " ".join(auxp.split())
                self._data[name] = val
            else:
                auxp = "You are trying to set the \
                parameter {0} that have to be a single number or other single type \
                of data with a list in the class {1}".format(name, \
                self.__class__.__name__)
                assert not isinstance(val, list), " ".join(auxp.split())
                self._data[name] = val
        else:
            auxp = "No valid key {0} has been passed to the function \
            set_element in the class {1}. The valid keys for this class are as \
            follows \n {2}".format(name, self.__class__.__name__, \
            self._data.keys()) 
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
        self._data = {}
        for x in aux:
            self._data[x] = None

        # Data that have to be a list of numbers or names
        aux =  ['bus_number', 'bus_position']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self._data.update(__data2)
        del __data2

class Transformers(Branch):
    ''' Two winding transformer class '''
    def __init__(self):
        super().__init__()
        # Basic _data
        aux = ['max_phase_shift_angle', 'min_phase_shift_angle', 'tap']
        __data2 = {}
        for x in aux:
            __data2[x] = None
        
        self._data.update(__data2)
        del __data2

class TransmissionLine(Branch):
    ''' Transmission Line class '''
    def __init__(self):
        super().__init__()

class Bus(_CommonMethods):
    ''' Electricity bus '''
    def __init__(self):
        # Data that have to be single numbers or names
        aux = ['voltage_kv', 'shunt_susceptance', 'area', 'type', 'position_x', 
               'position_y', 'shunt_conductance', 
               'active_power_demand_peak', 'reactive_power_demand_peak', 
               'position', 'name', 'number','initial_voltage_magnitude', 
               'initial_voltage_angle', 'maximum_voltage_magnitude', 
               'minimum_voltage_magnitude', 'zone', 'non_technical_losses_fix',
               'load_type']
        self._data = {}
        for x in aux:
            self._data[x] = None

        # Data that have to be a list of numbers or names
        aux =  ['contingency_n-1', 
                'transmissionline_position', 'transmissionline_number',
                'conv_position', 'conv_number',
                'hydro_position', 'hydro_number',
                'RES_position', 'RES_number',
                'transformers_position', 'transformers_number']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self._data.update(__data2)
        del __data2

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
        'notrafos' -> Number of transformers
        '''
        self._data = {
            'baseMVA': None,
            'Slack': None,
            'Security': [],  # list of N-1 cases to consider
            'Threewindinginitial' : None, # initial bus for three winding trafos
            'voltagethreewindingtrafos' : 1000000 # voltage for artificial nodes 
                                                  # in three winding trafos
            }
        
        self.__objects = {
            'bus' : [], # list of Bus objects
            'conv' : [], # list of conventional generator objects
            'hydro' : [], # list of hydro generator objects
            'RES' : [], # list of RES generator objects
            'transformers' : [], # list of two winding trafo objects
            'transmissionline' : [] # list of transmission line objects
            }
        
        # This list should contain the names of all series elements
        self.__series_elements = [
            'transmissionline', 'transformers'
        ]

        # This list should contain the names of all generator types
        self.__generation_types = [
            'conv', 'RES', 'hydro'
        ]

        self.__parameters_elements = [
            'baseMVA', 'Slack', 'Security', 'Threewindinginitial',
            'voltagethreewindingtrafos'
        ]

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

        # Initialise transformer object
        if 'notrafos' in kwargs.keys():
            self.__objects['transformers'] = [Transformers() for _ in
                range(kwargs.pop('notrafos'))]
        else:
            self.__objects['transformers'] = [Transformers()]
    
    def copy_electricity_network_data(self):
        ''' This returns a copy of the electricity network object'''
        return self.__dict__

    def delete_objects(self, name=None, pos=None):
        ''' This function delete an object or a list of objects for the 
        requested object "name". 
        - The function delete either a list with the positions indicated 
        in "pos" or the object of the position "pos"
        - "pos" needs to indicate a position between zero and the size of the 
        '''
        auxp = "No valid name has been passed to the function delete_objects \
            in the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        auxp = "No valid position has been passed to the function \
            delete_objects in the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self.__objects:
            copy_class = copy.deepcopy(self.__objects[name][0])
            copy_class.__init__()
            if isinstance(pos, list):
                self.__objects[name] = [i for j, i in \
                    enumerate(self.__objects[name]) if j not in pos]
            else:
                self.__objects[name].pop(pos)
            if self.__objects[name] == []:
                self.__objects[name] = [copy_class]
            self.__update_positions_objects(name)
            if name == 'bus':
                 # Updating the positions related to the bus object
                for xobj in self.__objects.keys():
                    if xobj != 'bus':
                        self.__update_relative_position_objects(\
                            name_object1='bus', \
                            name_element1='number', name_object2=xobj, \
                            name_element2='bus_number', 
                            name_position_element='bus_position')
            else:
                # Updating the positions of other objects in the bus object
                self.__update_relative_position_objects(name_object1=name, \
                    name_element1='number', name_object2='bus', \
                    name_element2=name+'_number', \
                    name_position_element=name+'_position')
        else:
            auxp = "No valid key {0} has been passed to the function \
            delete_objects in the class {1}. The valid keys for this class are \
            as follows: \n {2}".format(name, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())

    def get_generation_types_names(self):
        ''' This function returns a list of all generation types considered in 
        the electricity network class'''
        return self.__generation_types

    def get_no_objects(self, name=None):
        ''' This function returns the number of objects in a list '''
        auxp = "No valid name has been passed to the function get_no_objects \
            in the class {0}".format(self.__class__.__name__)
        assert name is not None, " ".join(auxp.split())
        if name in self.__objects:
            if self.__objects[name][0].get_element(name='position') != None:
                return len(self.__objects[name])
            else:
                return 0
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_no_objects in the class {1}. The valid keys for this class are \
            as follows: \n {2}".format(name, self.__class__.__name__, \
            self._data.keys()) 
            assert " ".join(auxp.split())

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
    
    def get_object_elements(self, name_object=None, name_element=None, \
        pos_element=':', pos_object=None):
        ''' This function returns the value or a list of values for the 
        requested parameter "name_element" of the object "name_obj" in position
        "pos_object". 
        - If the parameter is a list then the function returns either a list 
        with the positions indicated in "pos_element" or the value of the 
        position "pos_element"
        - "pos_element" needs to indicate a position between zero and the size 
        of the list of parameters
        - "pos_object" needs to indicate a position between zero and the size 
        of the list of objects
        - If the parameters is a value then "pos" can be ignored
        - The function returns by default the whole list of parameters '''

        auxp = "No valid object name has been passed to the function \
            get_objects_elements in the class \
            {0}".format(self.__class__.__name__)
        assert name_object is not None, " ".join(auxp.split())
        auxp = "No valid object position has been passed to the function \
            get_objects_elements in \
            the class {0}".format(self.__class__.__name__)
        assert pos_object is not None, " ".join(auxp.split())
        if name_object in self.__objects:
            return self.__objects[name_object][pos_object].get_element(\
                name=name_element, pos=pos_element)
        else:
            auxp = "No valid key {0} has been passed to the function \
            get_object_elements in the class {1}. The valid keys for this class are as \
            follows: \n {2}".format(name_object, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())
    
    def get_parameters_list(self):
        ''' This function returns a list parameters considered in 
        the electricity network class'''
        return self.__parameters_elements

    def get_series_elements_names(self):
        ''' This function returns a list of all series elements considered in 
        the electricity network class'''
        return self.__series_elements

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
                    self.__objects[name][0].__class__), \
                    " ".join(auxp.split())
                self.__objects[name] = list_obj
        else:
            auxp = "No valid key {0} has been passed to the function \
            set_objects in the class {1}. The valid keys for this class are as \
            follows \n {2}".format(name, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())

    def set_object_elements(self, name_object=None, name_element=None, \
        val=None, pos_object=None):
        ''' This function set the value or a list of values for the 
        requested parameter "name_element" of the object "name_obj" in position
        "pos_object". 
        - This function rewrite the existing values or value stored in the 
        parameter
        - "pos_object" needs to indicate a position between zero and the size 
        of the list of objects '''

        auxp = "No valid object name has been passed to the function \
            get_objects_elements in \
            the class {0}".format(self.__class__.__name__)
        assert name_object is not None, " ".join(auxp.split())
        auxp = "No valid object position has been passed to the function \
            get_objects_elements in \
            the class {0}".format(self.__class__.__name__)
        assert pos_object is not None, " ".join(auxp.split())
        if name_object in self.__objects:
            self.__objects[name_object][pos_object].set_element(\
                name=name_element, val=val)
        else:
            auxp = "No valid key {0} has been passed to the function \
            set_object_elements in the class {1}. The valid keys for this class are as \
            follows: \n {2}".format(name_object, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())

    def set_electricity_network_data(self, ob=None):
        ''' set the data of the electricity network object '''
        assert isinstance(ob, ElectricityNetwork), "Incorrect object \
            passed to set the Electricity Network data"
        for xkey in self.__objects.keys():
            self.set_objects(name=xkey, list_obj=ob.get_objects(name=xkey))
        for xkey in self.get_parameters_list():
            self.set_element(name=xkey,val=ob.get_element(name=xkey))
    
    def update_all_positions(self):
        ''' Update the position of all nodes, transmission lines, etc. '''
        for xobj in self.__objects.keys():
            self.__update_positions_objects(name_object=xobj)
        # Updating the positions related to the bus object
        for xobj in self.__objects.keys():
            if xobj != 'bus':
                self.__update_relative_position_objects(name_object1='bus', \
                    name_element1='number', name_object2=xobj, \
                    name_element2='bus_number', 
                    name_position_element='bus_position')
        # Updating the positions of other objects in the bus object
        for xobj in self.get_series_elements_names():
            self.__update_relative_position_objects(name_object1=xobj, \
                name_element1='number', name_object2='bus', \
                name_element2=xobj+'_number', \
                name_position_element=xobj+'_position')
        for xobj in self.get_generation_types_names():
            self.__update_relative_position_objects(name_object1=xobj, \
                name_element1='bus_number', name_object2='bus', \
                name_element2=xobj+'_number', \
                name_position_element=xobj+'_position')

    def __update_relative_position_objects(self, name_object1=None, \
        name_element1=None, name_object2=None, name_element2=None, 
        name_position_element=None):
        ''' This function update the relative position of the object 
        "name_object1" in the object "name_object2". 
        - "name_element1" and "name_element2" are elements in the objects that 
        are compared to update the positions 
        - "name_element1" is always a single parameter
        - "name_element2" can be a list or a single value
        - "name_position_element" is the name of the elements in the object 
        "name_object2" that will be updated '''
        auxp = "No valid object name 'name_object1' has been passed to the \
            function __update_relative_position_objects in \
            the class {0}".format(self.__class__.__name__)
        assert name_object1 is not None, " ".join(auxp.split())
        auxp = "No valid object name 'name_object2' has been passed to the \
            function __update_relative_position_objects in \
            the class {0}".format(self.__class__.__name__)
        assert name_object2 is not None, " ".join(auxp.split())
        aux = self.__objects[name_object2][0].get_element(name='position')
        aux1 = self.__objects[name_object1][0].get_element(name='position')
        # If any of the objects does not have values then there is nothing to
        # update
        if aux == None or aux1 == None:
            if name_object2 not in self.get_generation_types_names():
                for xobj2 in self.__objects[name_object2]:
                    xobj2.set_element(name=name_position_element, val=[])
            return

        if name_object1 in self.__objects and name_object2 in self.__objects:
            for xobj2 in self.__objects[name_object2]:
                aux1 = xobj2.get_element(name=name_element2)
                if isinstance(aux1, list):
                    aux3 = []
                    for xobj1 in self.__objects[name_object1]:
                        for aux2 in range(len(aux1)):
                            if aux1[aux2] == xobj1.get_element(\
                                name=name_element1):
                                aux3.append(xobj1.get_element(name='position'))
                    xobj2.set_element(name=name_position_element, val=aux3)
                else:
                    for xobj1 in self.__objects[name_object1]:
                        if aux1 == xobj1.get_element(name=name_element1):
                            xobj2.set_element(name=name_position_element, \
                                val=xobj1.get_element(name='position'))
                            break
        else:
            auxp = "No valid key {0} has been passed to the function \
            set_objects in the class {1}. The valid keys for this class are as \
            follows \n {2}".format(name_object1, self.__class__.__name__, \
            self.__objects.keys()) 
            assert name_object1 in self.__objects, " ".join(auxp.split())

            auxp = "No valid key {0} has been passed to the function \
            set_objects in the class {1}. The valid keys for this class are as \
            follows \n {2}".format(name_object2, self.__class__.__name__, \
            self.__objects.keys()) 
            assert name_object2 in self.__objects, " ".join(auxp.split())
    
    def __update_positions_objects(self, name_object=None):
        ''' This function update the position of the list of the object 
        "name_obj". 
        - This function rewrite the existing values of the positions '''

        auxp = "No valid object name has been passed to the function \
            get_objects_elements in \
            the class {0}".format(self.__class__.__name__)
        assert name_object is not None, " ".join(auxp.split())
        if name_object in self.__objects:
            if self.__objects[name_object][0].get_element(name='position') != \
                None:
                aux=0
                for xn in self.__objects[name_object]:
                    xn.set_element(name='position', val=aux)
                    aux += 1
        else:
            auxp = "No valid key {0} has been passed to the function \
            set_object_elements in the class {1}. The valid keys for this class are as \
            follows: \n {2}".format(name_object, self.__class__.__name__, \
            self.__objects.keys()) 
            assert " ".join(auxp.split())

class GenClass(_CommonMethods):
    ''' Core generation class '''
    def __init__(self):
        # Basic _data
        aux = [ 'maximum_active_power_generation', 
                'minimum_active_power_generation', 
                'maximum_reactive_power_generation',
                'minimum_reactive_power_generation', 'ramp', 'baseload',
                'model', 'shutdown_cost', 'startup_cost', 'position',
                'uncertainty', 'bus_number', 'bus_position', 'status']
        self._data = {}
        for x in aux:
            self._data[x] = None
        
        aux =  ['cost_function_parameters', 
        'piecewise_linearization_parameters']
        __data2 = {}
        for x in aux:
            __data2[x] = []
        
        self._data.update(__data2)
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
