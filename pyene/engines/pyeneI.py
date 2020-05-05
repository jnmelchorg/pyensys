# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:23:14 2018

Pyene Interfaces provides methods for exchanging information and models with
external tools, such as pypsa and pypower

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import numpy as np
import logging
import math
try:
    import pypsa
except ImportError:
    print('pypsa has not been installed - functionalities unavailable')

from pyene.engines.pyeneD import ElectricityNetwork, Generators


class EInterfaceClass:
    def pyene2pypsa(self, NM, xscen):
        # TODO: Remove xshift and prepare/validate tests
        xshift = 1
        '''Convert pyene files to pypsa format'''
        # Create pypsa network
        try:
            nu = pypsa.Network()
        except ImportError:
            return (0, False)

        nu.set_snapshots(range(NM.settings['NoTime']))
        baseMVA = NM.ENetwork.get_Base()

        # Names
        auxtxtN = 'Bus'
        auxtxtG = 'Gen'
        auxtxtLd = 'Load'

        '''                             NETWORK
        name - network name
        snapshots - list of snapshots or time steps
        snapshot_weightings - weights to control snapshots length
        now - current snapshot
        srid - spatial reference
        '''

        '''                               BUS
        Missing attributes:
            type - placeholder (not yet implemented in pypsa)
            carrier - 'AC' or 'DC'

        Implemented attributes:
            Name auxtxtN+str(xn)
            v_nom - Nominal voltage
            v_mag_pu_set - per unit voltage set point
            v_mag_pu_min - per unit minimum voltage
            v_mag_pu_max - per unit maximum voltage
            auxtxtN = 'Bus'  # Generic name for the nodes
            x - coordinates
            y - coordinates
        '''
        PVBus = np.zeros(NM.ENetwork.get_NoBus(), dtype=float)
        for ob in NM.Gen.Conv:
            if NM.ENetwork.Bus[ob.get_BusPos()].get_Type() == 2:
                PVBus[ob.get_BusPos()] = ob.get_VG()

        for ob in NM.ENetwork.Bus:
            if ob.get_kV() == 0:
                aux1 = 1
            else:
                aux1 = ob.get_kV()
            if ob.get_Type() == 2 or ob.get_Type() == 3:
                aux2 = ob.get_VM()
            else:
                aux2 = PVBus[ob.get_Pos()]

            if ob.get_X() is not None:
                nu.add('Bus', auxtxtN+str(ob.get_Number()),
                       v_nom=aux1, v_mag_pu_set=aux2,
                       v_mag_pu_min=ob.get_Vmin(),
                       v_mag_pu_max=ob.get_Vmax(),
                       x=ob.get_X(),
                       y=ob.get_Y())
            else:
                nu.add('Bus', auxtxtN+str(ob.get_Number()), v_nom=aux1,
                       v_mag_pu_set=aux2, v_mag_pu_min=ob.get_Vmin(),
                       v_mag_pu_max=ob.get_Vmax())

        '''                            GENERATOR
        Missing attributes:
            type - placeholder for generator type
            p_nom_extendable - boolean switch to increase capacity
            p_nom_min - minimum value for extendable capacity
            p_nom_max - maximum value for extendable capacity
            sign - power sign
            carrier - required for global constraints
            capital_cost - cost of extending p_nom by 1 MW
            efficiency - required for global constraints
            committable - boolean (only if p_nom is not extendable)
            start_up_cost - only if commitable is true
            shut_down_cost - only if commitable is true
            min_up_time - only if commitable is true
            min_down_time - only if commitable is true
            initial_status  - only if commitable is true
            ramp_limit_up - only if commitable is true
            ramp_limit_down - only if commitable is true
            ramp_limit_start_up - only if commitable is true
            ramp_limit_shut_down - only if commitable is true

        Implemented attributes:
            name - generator name
            bus - bus name
            control - 'PQ', 'PV' or 'Slack'
            p_min_pu - use default 0 for modelling renewables
            p_max_pu - multpier for intermittent maximum generation
            marginal_cost - linear model
        '''
        # Conventional fuel based generation
        xg = 0
        for ob in NM.Gen.Conv:
            xg += 1
            if NM.ENetwork.Bus[ob.get_BusPos()].get_Type() == 1:
                aux1 = 'PQ'
            elif NM.ENetwork.Bus[ob.get_BusPos()].get_Type() == 2:
                aux1 = 'PV'
            else:
                aux1 = 'Slack'
            nu.add('Generator', auxtxtG+str(xg),
                   bus=auxtxtN+str(ob.get_Bus()),
                   control=aux1,
                   p_nom_max=ob.get_Max()*baseMVA,
                   p_set=ob.get_P(),
                   q_set=ob.get_Q(),
                   marginal_cost=ob.cost['LCost'][0][0]
                   )

        yres = np.zeros(NM.settings['NoTime'], dtype=float)
        for ob in NM.Gen.RES:
            xg += 1
            for xt in range(NM.settings['NoTime']):
                yres[xt] = (NM.scenarios['RES']
                            [NM.resScenario[ob.get_Pos()][xscen]+xt])
            nu.add('Generator', auxtxtG+str(xg+1),
                   bus=auxtxtN+str(ob.get_Bus()),
                   control='PQ',
                   p_nom_max=yres,
                   p_nom=ob.get_Max()*baseMVA,
                   p_set=0,
                   q_set=0,
                   marginal_cost=ob.cost['LCost'][0][0]
                   )

        '''                             CARRIER
        name - Energy carrier
        co2_emissions - tonnes/MWh
        carrier_attribute
        '''

        '''                         GLOBAL CONSTRAINT
        name - constraint
        type - only 'primary_energy' is supported
        carrier_attribute - attributes such as co2_emissions
        sense - either '<=', '==' or '>='
        constant - constant for right hand-side of constraint
        '''

        #                           STORAGE UNIT
        #                              STORE
        '''                              LOAD
        Missing attributes:
            type - placeholder for type
            sign - change to -1 for generator

        Implemented attributes:
            name - auxtxtLd+str(xL)
            bus - Name of bus
        '''
        xL = 0
        ydemP = np.zeros(NM.settings['NoTime'], dtype=float)
        ydemQ = np.zeros(NM.settings['NoTime'], dtype=float)
        for xn in range(NM.ENetwork.get_NoBus()):
            if NM.demandE['PD'][xn] != 0:
                xL += 1
                for xt in range(NM.settings['NoTime']):
                    aux = (NM.scenarios['Demand']
                           [NM.busScenario[xn][xscen]+xt])
                    ydemP[xt] = NM.demandE['PD'][xn]*aux
                    ydemQ[xt] = NM.demandE['QD'][xn]*aux
                nu.add('Load', auxtxtLd+str(xL),
                       bus=auxtxtN+str(xn+1),
                       p_set=ydemP,
                       q_set=ydemQ
                       )

        #                         SHUNT IMPEDANCE

        '''                              LINE
        Missing attributes:
            type - assign string to re-calculate line parameters
            g - shunt conductivity
            s_nom_extendable - boolean switch to allow s_nom to be extended
            s_nom_min - minimum capacity for extendable s_nom
            s_nom_max - maximum capacity for extendable s_nom
            s_max_pu - multiplier (time series) for considering DLR
            capital_cost - cost for extending s_nom by 1
            length - line length
            terrain_factor - terrain factor for increasing capital costs
            num_parallel - number of lines in parallel
            v_ang_min - placeholder for minimum voltage angle drop/increase
            v_and_max - placeholder for maximum voltage angle drop/increase
            b - shunt susceptance

        Implemented attributes:
            Name - auxtxtL+str(xb)
            bus0 - One of the buses connected to the line
            bus1 - The other bus connected to the line
            x - series reactance
            r - series resistance
            snom - MVA limit
        '''
        auxtxtL = 'Line'
        for ob in NM.ENetwork.Branch:
            if ob.get_Tap() == 0:
                auxpu = nu.buses['v_nom']['Bus{}'.format(ob.get_BusF())]**2 / \
                    NM.ENetwork.get_Base()
                nu.add('Line', auxtxtL+str(ob.get_Number()),
                       bus0=auxtxtN+str(ob.get_BusF()),
                       bus1=auxtxtN+str(ob.get_BusT()),
                       x=ob.get_X()*auxpu,
                       r=ob.get_R()*auxpu,
                       s_nom=ob.get_Rate()*NM.ENetwork.get_Base()
                       )

        #                           LINE TYPES
        '''                           TRANSFORMER
        Missing attributes:
            type - assign string to re-calculate line parameters
            g - shunt conductivity
            s_nom_extendable - boolean switch to allow s_nom to be extended
            s_nom_min - minimum capacity for extendable s_nom
            s_nom_max - maximum capacity for extendable s_nom
            s_max_pu - multiplier (time series) for considering DLR
            capital_cost - cost for extending s_nom by 1
            num_parallel - number of lines in parallel
            tap_position - if a type is defined, determine tap position
            phase_shift - voltage phase angle
            v_ang_min - placeholder for minimum voltage angle drop/increase
            v_and_max - placeholder for maximum voltage angle drop/increase

        Implemented attributes:
            Name - auxtxtT+str(xb)
            bus0 - One of the buses connected to the line
            bus1 - The other bus connected to the line
            model - Matpower and pypower use pi admittance models
            tap_ratio - transformer ratio
            tap_side - taps at from bus in Matlab
        '''
        auxtxtT = 'Trs'
        for ob in NM.ENetwork.Branch:
            if ob.get_Tap() != 0:
                nu.add('Transformer', auxtxtT+str(ob.get_Number()),
                       bus0=auxtxtN+str(ob.get_BusF()),
                       bus1=auxtxtN+str(ob.get_BusT()),
                       model='pi',
                       x=ob.get_X(),
                       r=ob.get_R(),
                       b=ob.get_B(),
                       s_nom=ob.get_Rate()*NM.ENetwork.get_Base(),
                       tap_ratio=ob.get_Tap(),
                       tap_side=0
                       )

        #                        TRANSFORMER TYPES
        #                              LINK

        return (nu, True)

    def pypsa2pyene(self):
        '''Convert pyene files to pypsa format'''
        print('To be finalised pypsa2pyene')

    def pyene2pypower(self):
        '''Convert pyene files to pypsa format'''
        print('To be finalised pyene2pypower')

    def pypower2pyene(self):
        '''Convert pyene files to pypsa format'''
        print('To be finalised pypower2pyene')

class PSSErawInterface:
    ''' This class provides functionalities to read raw files from psse'''
    def __init__(self, pathpsseraw = None):
        ''' The init function open the psse raw file if the path \
            is provided'''
        if pathpsseraw != None:
            self.file = open(pathpsseraw, 'r')
        else:
            self.file = None
    
    def detectpsseversion(self):
        ''' This function determines the psse version used to store \
            the raw data '''
        self.psseversion = None
        aux1 = self.filecontent[0].split(',')
        aux1 = int(aux1[2])
        if  aux1 == 31 or aux1 == 33 or aux1 == 34:
            self.psseversion = aux1
        else:
            logging.warning('PSSE version not supported')

    def parsesystemdata(self):
        ''' This function parse the information that corresponds to \
            the whole power system '''
        aux1 = self.filecontent[0].split(',')
        aux2 = aux1[5].split('/')
        print(aux1[1])
        self.BaseUnitPower = float(aux1[1])
        self.BaseFrequency = float(aux2[0])
        self.UnitsTrafosRatings = int(aux1[3]) #  value <= 0 for MVA
                            # value > 0 for current expressed as MVA
        self.UnitsnonTrafosRatings = int(aux1[4]) #  value <= 0 for MVA
                            # value > 0 for current expressed as MVA
    
    def parsenodedata(self):
        ''' This function parse the information that corresponds to \
            the nodes of the power system '''
        aux2 = self.filecontent[3].split(',')
        self.BusNumber = np.array(int(aux2[0]))
        self.BusNomV = np.array(float(aux2[2]))
        aux1 = 4
        while self.filecontent[aux1][0:3] != '0 /':
            aux2 = self.filecontent[aux1].split(',')
            self.BusNumber = np.append(self.BusNumber, int(aux2[0]))
            self.BusNomV = np.append(self.BusNomV, float(aux2[2]))
            aux1 += 1
        self.NumberNodes = len(self.BusNumber)

    
    def parsetrafodata(self):
        ''' This function parse the information that corresponds to \
            the transformers of the power system '''
        aux1 = 14553
        aux3 = 0
        aux4 = 0
        self.ThreeWindingR = np.empty(1)
        self.ThreeWindingX = np.empty(1)
        self.ThreeWindingFrom = np.empty(1)
        self.ThreeWindingTo = np.empty(1)
        self.ConvTrafoR = np.empty(1)
        self.ConvTrafoX = np.empty(1)
        self.ConvTrafoFrom = np.empty(1)
        self.ConvTrafoTo = np.empty(1)        
        while self.filecontent[aux1][0:3] != '0 /':
            aux2 = self.filecontent[aux1].split(',')
            if int(aux2[2]) == 0:
                aux4 += 1
                Bus1 = int(aux2[0])
                Bus2 = int(aux2[1])
                CW = int(aux2[4])
                CZ = int(aux2[5])
                CM = int(aux2[6])
                aux1 += 1
                aux2 = self.filecontent[aux1].split(',')
                R12 = float(aux2[0])
                X12 = float(aux2[1])
                SBase12 = float(aux2[2])
                aux1 += 1
                aux2 = self.filecontent[aux1].split(',')
                NOMV1 = float(aux2[1])
                aux1 += 1
                aux2 = self.filecontent[aux1].split(',')
                NOMV2 = float(aux2[1])
                aux1 += 1
                # Verifying and adapting impedances to base power
                if CZ == 2:
                    if SBase12 != self.BaseUnitPower:
                        R12 = R12 * (self.BaseUnitPower/SBase12)
                        X12 = X12 * (self.BaseUnitPower/SBase12)
                elif CZ == 3:
                    R12 = R12/(self.BaseUnitPower * 1e6)
                    X12 = math.sqrt(X12**2 - R12**2) 
                if aux4 == 1:
                    self.ConvTrafoR[0] = R12
                    self.ConvTrafoX[0] = X12
                    self.ConvTrafoFrom[0] = Bus1
                    self.ConvTrafoTo[0] = Bus2
                else:
                    self.ConvTrafoR = np.append(self.ConvTrafoR, R12)
                    self.ConvTrafoX = np.append(self.ConvTrafoX, X12)
                    self.ConvTrafoFrom = np.append(\
                        self.ConvTrafoFrom, Bus1)
                    self.ConvTrafoTo = np.append(\
                        self.ConvTrafoTo, Bus2)
            else:
                aux3 += 1
                Bus1 = int(aux2[0])
                Bus2 = int(aux2[1])
                Bus3 = int(aux2[2])
                CW = int(aux2[4])
                CZ = int(aux2[5])
                CM = int(aux2[6])
                aux1 += 1
                aux2 = self.filecontent[aux1].split(',')
                R12 = float(aux2[0])
                X12 = float(aux2[1])
                SBase12 = float(aux2[2])
                R23 = float(aux2[3])
                X23 = float(aux2[4])
                SBase23 = float(aux2[5])
                R31 = float(aux2[6])
                X31 = float(aux2[7])
                SBase31 = float(aux2[8])
                aux1 += 1
                aux2 = self.filecontent[aux1].split(',')
                NOMV1 = float(aux2[1])
                aux1 += 1
                aux2 = self.filecontent[aux1].split(',')
                NOMV2 = float(aux2[1])
                aux1 += 1
                aux2 = self.filecontent[aux1].split(',')
                NOMV3 = float(aux2[1])
                aux1 += 1
                # Verifying and adapting impedances to base power
                if CZ == 2:
                    if SBase12 != self.BaseUnitPower:
                        R12 = R12 * (self.BaseUnitPower/SBase12)
                        X12 = X12 * (self.BaseUnitPower/SBase12)
                    if SBase23 != self.BaseUnitPower:
                        R23 = R23 * (self.BaseUnitPower/SBase23)
                        X23 = X23 * (self.BaseUnitPower/SBase23)
                    if SBase31 != self.BaseUnitPower:
                        R31 = R31 * (self.BaseUnitPower/SBase31)
                        X31 = X31 * (self.BaseUnitPower/SBase31)
                elif CZ == 3:
                    R12 = R12/(self.BaseUnitPower * 1e6)
                    X12 = math.sqrt(X12**2 - R12**2)
                    R23 = R23/(self.BaseUnitPower * 1e6)
                    X23 = math.sqrt(X23**2 - R23**2)
                    R31 = R31/(self.BaseUnitPower * 1e6)
                    X31 = math.sqrt(X31**2 - R31**2)

                # Verifying inconsistent reactances and resistances
                minX = min(X12, X23, X31)
                if (X12/minX) > 100:
                    X12 = 100 * minX
                if (X23/minX) > 100:
                    X23 = 100 * minX
                if (X31/minX) > 100:
                    X31 = 100 * minX

                # From delta to star
                R1 = (1/2)*(R12 + R31 - R23)
                R2 = (1/2)*(R12 + R23 - R31)
                R3 = (1/2)*(R31 + R23 - R12)
                X1 = (1/2)*(X12 + X31 - X23)
                X2 = (1/2)*(X12 + X23 - X31)
                X3 = (1/2)*(X31 + X23 - X12)
                if aux3 == 1:
                    self.ThreeWindingR[0] = R1
                    self.ThreeWindingR = np.append(self.ThreeWindingR, R2)
                    self.ThreeWindingR = np.append(self.ThreeWindingR, R3)
                    self.ThreeWindingX[0] = X1
                    self.ThreeWindingX = np.append(self.ThreeWindingX, X2)
                    self.ThreeWindingX = np.append(self.ThreeWindingX, X3)
                    self.ThreeWindingFrom[0] = Bus1
                    self.ThreeWindingFrom = np.append(\
                        self.ThreeWindingFrom, Bus2)
                    self.ThreeWindingFrom = np.append(\
                        self.ThreeWindingFrom, Bus3)
                    self.ThreeWindingTo[0] = 1000000 + aux3
                    self.ThreeWindingTo = np.append(\
                        self.ThreeWindingTo, 1000000 + aux3)
                    self.ThreeWindingTo = np.append(\
                        self.ThreeWindingTo, 1000000 + aux3)
                else:
                    self.ThreeWindingR = np.append(self.ThreeWindingR, R1)
                    self.ThreeWindingR = np.append(self.ThreeWindingR, R2)
                    self.ThreeWindingR = np.append(self.ThreeWindingR, R3)
                    self.ThreeWindingX = np.append(self.ThreeWindingX, X1)
                    self.ThreeWindingX = np.append(self.ThreeWindingX, X2)
                    self.ThreeWindingX = np.append(self.ThreeWindingX, X3)
                    self.ThreeWindingFrom = np.append(\
                        self.ThreeWindingFrom, Bus1)
                    self.ThreeWindingFrom = np.append(\
                        self.ThreeWindingFrom, Bus2)
                    self.ThreeWindingFrom = np.append(\
                        self.ThreeWindingFrom, Bus3)
                    self.ThreeWindingTo = np.append(\
                        self.ThreeWindingTo, 1000000 + aux3)
                    self.ThreeWindingTo = np.append(\
                        self.ThreeWindingTo, 1000000 + aux3)
                    self.ThreeWindingTo = np.append(\
                        self.ThreeWindingTo, 1000000 + aux3)  
        self.NumberThreeWindingTrafos = len(self.ThreeWindingTo)
        self.NumberConvTrafos = len(self.ConvTrafoR)

    def pssedataparser(self):
        ''' This function parse the psse data per device '''
        self.parsesystemdata()

    
    def readpsserawfile(self):
        if self.file == None:
            logging.warning('No PSSE raw file to read')
            return
        self.filecontent = self.file.readlines()
        self.detectpsseversion()
        self.parsesystemdata()
        self.parsenodedata()
        self.parsetrafodata()
        self.file.close()
        with open('pyene/externaldata/datatrafos3.dat', 'w') as f:
            for aux1 in range(self.NumberConvTrafos):
                f.write("%d\t%d\t%.9f\t%.9f\n" %(self.ConvTrafoFrom[aux1], \
                    self.ConvTrafoTo[aux1], self.ConvTrafoR[aux1], \
                    self.ConvTrafoX[aux1]))
            for aux1 in range(self.NumberThreeWindingTrafos):
                f.write("%d\t%d\t%.9f\t%.9f\n" %(self.ThreeWindingFrom[aux1], \
                    self.ThreeWindingTo[aux1], self.ThreeWindingR[aux1], \
                    self.ThreeWindingX[aux1]))
        



        print(self.filecontent[0].split(','))
        print(type(self.filecontent))
        print(type(self.filecontent[0]))

    def openpsserawfile(self, pathpsseraw = None):
        ''' This definition allows opening a psse raw file without passing the \
            path when the object is created'''
        if pathpsseraw != None:
            self.file = open(pathpsseraw, 'r')
        else:
            logging.warning('No path has been pass to the PSSE raw reader\
                therefore none file has been opened')

