# Python Energy & Network Engine
"""
Created on Thu Mar 29 14:04:58 2018

@author: mchihem2
"""
# convert in or long into float before performing divissions
from __future__ import division
from pyomo.core import ConcreteModel, Constraint, Objective, Suffix, Var, \
                       NonNegativeReals, minimize
from pyomo.opt import SolverFactory
import numpy as np
from .pyeneN import ENetworkClass as dn  # Network component
from .pyeneE import EnergyClass as de  # Energy balance/aggregation component
from tables import Int16Col, Float32Col, StringCol
from tables import IsDescription
import json
import os
import math
try:
    import pypsa
except ImportError:
    print('pypsa has not been installed - functionalities unavailable')


class pyeneClass():
    # Initialisation
    def __init__(self):
        # Chose to load data from file
        self.fRea = True
        self.Penalty = 1000000

    # Get mathematical model
    def SingleLP(self, ENM):
        # Define pyomo model
        mod = ConcreteModel()

        #                                 Sets                                #
        mod = ENM.getSets(mod)

        #                              Parameters                             #
        mod = ENM.getPar(mod)

        #                           Model Variables                           #
        mod = ENM.getVars(mod)

        #                             Constraints                             #
        mod = ENM.addCon(mod)

        return mod

    # Network only optimisation
    def NSim(self, conf):
        # Get network object
        NM = dn()

        # Initialise
        NM.initialise(conf)

        # Build LP model
        NModel = self.SingleLP(NM)

        #                          Objective function                         #
        NModel.OF = Objective(rule=NM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print results
        results = opt.solve(NModel)

        return (NM, NModel, results)

    # Energy only optimisation
    def ESim(self, conf):
        # Get energy object
        EM = de()

        # Chose to load data from file
        EM.fRea = True

        # Initialise
        EM.initialise(conf)

        # Build LP model
        EModel = self.SingleLP(EM)

        #                          Objective function                         #
        EModel.OF = Objective(rule=EM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print
        results = opt.solve(EModel)

        return (EM, EModel, results)

    #                           Objective function                            #
    def OF_rule(self, m):
        return sum((sum(sum(m.vGCost[self.hGC[xh]+xg, xt] for xg in m.sGen) +
                        sum(m.vFea[self.hFea[xh]+xf, xt] for xf in m.sFea) *
                        self.Penalty for xt in m.sTim) -
                    sum(self.NM.pumps['Value'][xdl] *
                        self.NM.networkE.graph['baseMVA'] *
                        sum(m.vDL[self.hDL[xh]+xdl+1, xt] *
                            self.NM.scenarios['Weights'][xt]
                            for xt in m.sTim) for xdl in m.sDL)) *
                   self.OFaux[xh] for xh in self.h)

    # Water consumption depends on water use by the electricity system
    def EMNM_rule(self, m, xL, xv):
        return (m.WOutFull[m.LLENM[xL][0], xv] ==
                self.NM.networkE.graph['baseMVA'] *
                sum(m.vGen[m.LLENM[xL][1]+xv, xt] *
                    self.NM.scenarios['Weights'][xt] for xt in m.sTim))

    #                               Constraints                               #
    def addCon(self, m):
        # Link water consumption from both models
        m.EMNM = Constraint(m.sLLEN, m.sVec, rule=self.EMNM_rule)
        # Allow collection of ual values
        m.dual = Suffix(direction=Suffix.IMPORT)

        return m

    #                                   Sets                                  #
    def getSets(self, m):
        m.sLLEN = range(self.NoLL)

        return m

    #                                Parameters                               #
    def getPar(self, m):
        m.LLENM = self.LLENM

        return m

    #                             Model Variables                             #
    def getVars(self, m):
        # Converting some parameters to variables
        del m.WOutFull
        m.WOutFull = Var(m.sNodz, m.sVec, domain=NonNegativeReals,
                         initialize=0.0)
        return m

    # Initialise energy and networks simulator
    def initialise(self, conf):
        # Creat objects
        self.conf = conf
        self.EM = de()
        self.NM = dn()

        # Adding hydro to the energy balance tree
        self.EM.settings = {
                'Fix': True,  # Force a specific number of vectors
                'Vectors': conf.NoHydro  # Number of vectors
                }

        # Initialise energy balance model
        self.EM.initialise(conf)

        # Get number of required network model instances
        NoNM = (1+self.EM.tree['Time'][self.EM.size['Periods']][1] -
                self.EM.tree['Time'][self.EM.size['Periods']][0])

        # Add time steps
        aux = self.EM.size['Scenarios']
        self.NM.scenarios['Number'] = aux
        self.NM.scenarios['NoDem'] = conf.NoDemProfiles
        self.NM.scenarios['NoRES'] = conf.NoRESProfiles
        self.NM.scenarios['Demand'] = np.ones(conf.Time*conf.NoDemProfiles,
                                              dtype=float)
        self.NM.settings['NoTime'] = conf.Time

        # Initialise Hydropower
        if conf.NoHydro > 0:
            self.NM.hydropower['Number'] = conf.NoHydro
            self.NM.hydropower['Bus'] = conf.Hydro
            self.NM.hydropower['Max'] = conf.HydroMax
            self.NM.hydropower['Cost'] = conf.HydroCost

        # Initialise Pumps
        if conf.NoPump > 0:
            self.NM.pumps['Number'] = conf.NoPump
            self.NM.pumps['Bus'] = conf.Pump
            self.NM.pumps['Max'] = conf.PumpMax
            self.NM.pumps['Value'] = conf.PumpVal

        # Initialise RES
        if conf.NoRES > 0:
            self.NM.RES['Number'] = conf.NoRES
            self.NM.RES['Bus'] = conf.RES
            self.NM.RES['Max'] = conf.RESMax
            self.NM.RES['Cost'] = conf.Cost
            self.NM.scenarios['RES'] = np.zeros(conf.NoRESProfiles*conf.Time,
                                                dtype=float)

        # Initialise network model
        self.NM.initialise(conf)

        # Add connections between energy balance and networ models
        self.NM.connections['set'] = range(NoNM)
        self.NM.connections['Flow'] = np.zeros(NoNM, dtype=int)
        self.NM.connections['Voltage'] = np.zeros(NoNM, dtype=int)
        self.NM.connections['Loss'] = np.zeros(NoNM, dtype=int)
        self.NM.connections['Generation'] = np.zeros(NoNM, dtype=int)
        self.NM.connections['Cost'] = np.zeros(NoNM, dtype=int)
        self.NM.connections['Pump'] = np.zeros(NoNM, dtype=int)
        self.NM.connections['Feasibility'] = np.zeros(NoNM, dtype=int)
        # Location of each instance
        aux = self.NM.networkE.number_of_edges()
        for xc in self.NM.connections['set']:
            self.NM.connections['Flow'][xc] = xc*(self.NM.NoBranch+1)
            self.NM.connections['Voltage'][xc] = xc*(self.NM.NoBuses+1)
            self.NM.connections['Loss'][xc] = xc*(aux+1)
            self.NM.connections['Generation'][xc] = xc*(self.NM.generationE
                                                        ['Number']+1)
            self.NM.connections['Cost'][xc] = xc*self.NM.generationE['Number']
            self.NM.connections['Pump'][xc] = xc*(self.NM.pumps['Number']+1)
            self.NM.connections['Feasibility'][xc] = xc*self.NM.NoFea

        # Build LL to link the models through hydro consumption
        self.NoLL = (1+self.EM.tree['Time'][self.EM.size['Periods']][1] -
                     self.EM.tree['Time'][self.EM.size['Periods']][0])
        self.LLENM = np.zeros((self.NoLL, 2), dtype=int)
        NoGen0 = (self.NM.generationE['Number']-self.NM.hydropower['Number'] -
                  self.NM.RES['Number']+1)
        for xc in range(self.NoLL):
            self.LLENM[xc][:] = ([self.EM.tree['Time'][self.EM.size['Periods']]
                                  [0]+xc, self.NM.connections['Generation']
                                  [xc]+NoGen0])

        # Taking sets for modelling local objective function
        self.hGC = self.NM.connections['Cost']
        self.hDL = self.NM.connections['Pump']
        self.hFea = self.NM.connections['Feasibility']
        self.h = self.NM.connections['set']

    def set_Hydro(self, index, value):
        ''' Set kWh of hydro that are available for a single site '''
        if self.NM.hydropower['Number'] == 1:
            self.EM.Weight['In'][1] = value
        else:
            self.EM.Weight['In'][1][index-1] = value

    def set_HydroPrice(self, index, value):
        ''' Set kWh of hydro that are available for a single site '''
        raise NotImplementedError('Water prices not yet enabled')
        # Adjust self.NM.GenLCst

    def set_GenCoFlag(self, index, value):
        ''' Adjust maximum output of generators '''
        if isinstance(value, bool):
            if value:
                # Maximum capacity
                self.NM.GenMax[index-1] = (self.generationE['Data']['PMAX']
                                           [index-1]/self.NM.networkE.graph
                                           ['baseMVA'])
            else:
                # Switch off
                self.NM.GenMax[index-1] = 0
        else:
            # Adjust capacity
            self.NM.GenMax[index-1] = value/self.NM.networkE.graph['baseMVA']

    def set_PumpPrice(self, index, value):
        ''' Set value for water pumped '''
        raise NotImplementedError('Pump prices not yet enabled')
        # Adjust self.NM.GenLCst

    def set_RES(self, index, value):
        '''
        Set PV/Wind profile  - more than one device can be connected to
        each profile
        '''

        aux1 = (index-1)*self.NM.settings['NoTime']
        aux2 = index*self.NM.settings['NoTime']

        self.NM.scenarios['RES'][aux1:aux2] = value
        xi = 0
        for xs in range(aux1, aux2):
            self.NM.scenarios['RES'][xs] = (value[xi] /
                                            self.NM.networkE.graph['baseMVA'])
            xi += 1

    def set_Demand(self, index, value):
        ''' Set a demand profile '''
        aux1 = (index-1)*self.NM.settings['NoTime']
        aux2 = index*self.NM.settings['NoTime']

        self.NM.scenarios['Demand'][aux1:aux2] = value

    def set_LineCapacity(self, index, value, *argv):
        ''' Adjust maximum capacity of a line '''
        aux1 = value
        if 'BR_R' in argv:
            aux2 = 0
        elif 'BR_X' in argv:
            aux2 = 1
        elif 'BR_B' in argv:
            aux2 = 2
        else:
            aux1 = value/self.NM.networkE.graph['baseMVA']
            aux2 = 3
        self.NM.branchData[self.NM.LLESec1[index-1][0]][aux2] = aux1

    def get_timeAndScenario(self, mod, *varg, **kwarg):
        # Specify times
        if 'times' in kwarg:
            auxtime = kwarg.pop('times')
        else:
            auxtime = mod.sTim

        # Remove weights
        if 'snapshot' in varg:
            auxweight = np.ones(len(mod.sTim), dtype=int)
            auxOF = np.ones(len(self.h), dtype=int)
        else:
            auxweight = self.NM.scenarios['Weights']
            auxOF = self.OFaux

        # Specify scenario
        if 'scens' in kwarg:
            auxscens = kwarg.pop('scens')
        else:
            auxscens = self.h

        return (auxtime, auxweight, auxscens, auxOF)

    def get_Hydro(self, mod, index):
        ''' Get surplus kWh from specific site '''
        HydroValue = mod.WOutFull[1, index-1].value

        return HydroValue

    def get_AllHydro(self, mod):
        ''' Get surplus kWh from all hydropower plants '''
        value = 0
        for xi in range(self.EM.settings['Vectors']):
            value += mod.WOutFull[1, xi].value

        return value

    def get_HydroMarginal(self, mod, index):
        ''' Get marginal costs for specific hydropower plant '''
        cobject = getattr(mod, 'SoCBalance')
        aux = mod.dual.get(cobject[1, index-1])
        if aux is None:
            HydroValue = 0
        else:
            HydroValue = -1*int(mod.dual.get(cobject[1, index-1]))

        return HydroValue

    def getMeanHydroMarginal(self, mod):
        value = 0
        for xi in range(self.EM.settings['Vectors']):
            value += self.get_HydroMarginal(mod, xi+1)
        return value/self.EM.settings['Vectors']

    def get_HydroFlag(self, mod, index):
        ''' Get surplus kWh from specific site '''
        cobject = getattr(mod, 'SoCBalance')
        aux = mod.dual.get(cobject[1, index-1])
        if aux is None:
            HydroValue = False
        else:
            aux2 = -1*int(mod.dual.get(cobject[1, index-1]))
            aux3 = self.Penalty/self.NM.networkE.graph['baseMVA']
            if aux2 > aux3:
                HydroValue = True
            else:
                HydroValue = False

        return HydroValue

    def get_NetDemand(self, mod, auxFlags, *varg, **kwarg):
        ''' Get MWh consumed by pumps, loads, losses '''
        value = 0
        if auxFlags[0]:  # Demand
            value += self.get_AllDemand(mod, *varg, **kwarg)

        if auxFlags[1]:  # Pumps
            value += self.get_AllPumps(mod, *varg, **kwarg)

        if auxFlags[2]:  # Loss
            value += self.get_AllLoss(mod, *varg, **kwarg)

        if auxFlags[3]:  # Curtailment
            value += self.get_AllDemandCurtailment(mod, *varg, **kwarg)

        if auxFlags[4]:  # Spill
            value += self.get_AllRES(mod, *varg, **kwarg)

        return value

    def get_OFparts(self, m, auxFlags, *varg, **kwarg):
        ''' Get components of the objective function '''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(m, *varg, **kwarg)

        value = 0
        if auxFlags[0]:  # Conventional generation
            value += sum(sum(sum(m.vGCost[self.hGC[xh]+xg, xt].value for xg
                                 in range(self.NM.settings['Generators']))
                             for xt in auxtime)*auxOF[xh] for xh in auxscens)
        if auxFlags[1]:  # RES generation
            value += sum(sum(sum(m.vGCost[self.hGC[xh]+xg, xt].value for xg
                                 in self.NM.RES['Link'])
                             for xt in auxtime)*auxOF[xh] for xh in auxscens)

        if auxFlags[2]:  # Hydro generation
            value += sum(sum(sum(m.vGCost[self.hGC[xh]+xg, xt].value for xg
                                 in self.NM.hydropower['Link'])
                             for xt in auxtime)*auxOF[xh] for xh in auxscens)

        if auxFlags[3]:  # Pumps
            value -= sum(sum(self.NM.pumps['Value'][xdl] *
                             self.NM.networkE.graph['baseMVA'] *
                             sum(m.vDL[self.hDL[xh]+xdl+1, xt].value *
                                 self.NM.scenarios['Weights'][xt]
                                 for xt in auxtime) for xdl in m.sDL) *
                         auxOF[xh] for xh in auxscens)

        if auxFlags[4]:  # Curtailment
            value += sum(sum(sum(m.vFea[self.hFea[xh]+xf, xt].value for xf
                                 in m.sFea)*self.Penalty for xt in auxtime) *
                         auxOF[xh] for xh in auxscens)

        return value

    def get_Pump(self, mod, index, *varg, **kwarg):
        ''' Get kWh consumed by a specific pump '''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(mod, *varg, **kwarg)
        value = 0
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += (mod.vDL[self.hDL[xh]+index, xt].value*auxweight[xt])
            value += acu*auxOF[xh]
        value *= self.NM.networkE.graph['baseMVA']

        return value

    def get_AllPumps(self, mod, *varg, **kwarg):
        ''' Get kWh consumed by all pumps '''
        value = 0
        for xp in range(self.NM.pumps['Number']):
            value += self.get_Pump(mod, xp+1, *varg, **kwarg)

        return value

    def get_DemandCurtailment(self, mod, bus, *varg, **kwarg):
        '''Get the kWh that had to be curtailed from a given bus'''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(mod, *varg, **kwarg)

        value = 0
        if self.NM.settings['Feasibility']:
            for xh in auxscens:
                acu = 0
                for xt in auxtime:
                    acu += (mod.vFea[self.hFea[xh]+bus, xt].value *
                            auxweight[xt])
                value += acu*auxOF[xh]
            value *= self.NM.networkE.graph['baseMVA']

        return value

    def get_AllDemandCurtailment(self, mod, *varg, **kwarg):
        '''Get the kWh that had to be curtailed from all buses'''
        # Specify buses
        if 'buses' in kwarg:
            auxbuses = kwarg.pop('buses')
        else:
            auxbuses = range(self.NM.networkE.number_of_nodes())

        value = 0
        if self.NM.settings['Feasibility']:
            for xn in auxbuses:
                value += self.get_DemandCurtailment(mod, xn+1, *varg, **kwarg)

        return value

    def get_Demand(self, mod, bus, *varg, **kwarg):
        '''Get the kWh that had to be curtailed from a given bus'''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(mod, *varg, **kwarg)

        value = 0
        xb = bus-1
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += (self.NM.busData[xb]*self.NM.scenarios['Demand']
                        [xt+self.NM.busScenario[xb][xh]])*auxweight[xt]
            value += acu*auxOF[xh]
        value *= self.NM.networkE.graph['baseMVA']

        return value

    def get_AllDemand(self, mod, *varg, **kwarg):
        '''Get the demand'''
        # Specify buses
        if 'buses' in kwarg:
            auxbuses = kwarg.pop('buses')
        else:
            auxbuses = range(self.NM.networkE.number_of_nodes())

        value = 0
        for xn in auxbuses:
            value += self.get_Demand(mod, xn+1, *varg, **kwarg)

        return value

    def get_Loss(self, mod, xb, *varg, **kwarg):
        '''Get the kWh that had to be curtailed from a given bus'''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(mod, *varg, **kwarg)

        value = 0
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += (mod.vLoss[self.NM.connections['Loss']
                                  [xh]+xb, xt].value)*auxweight[xt]
            value += acu*auxOF[xh]
        value *= self.NM.networkE.graph['baseMVA']

        return value

    def get_AllLoss(self, mod, *varg, **kwarg):
        '''Get the demand'''
        value = 0
        if self.NM.settings['Losses']:
            for xb in mod.sBranch:
                value += self.get_Loss(mod, xb+1, **kwarg)

        return value

    def get_Generation(self, mod, index, *varg, **kwarg):
        ''' Get kWh for a single generator '''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(mod, *varg, **kwarg)

        value = 0
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += (mod.vGen[self.NM.connections['Generation']
                                 [xh]+index, xt].value*auxweight[xt])
            value += acu*auxOF[xh]
        value *= self.NM.networkE.graph['baseMVA']

        return value

    def get_AllGeneration(self, mod, *varg, **kwarg):
        ''' Get kWh for all generators for the whole period '''
        if 'All' in varg:
            aux = range(1, self.NM.generationE['Number']+1)
        elif 'Conv' in varg:
            aux = range(1, self.NM.settings['Generators']+1)
        elif 'RES' in varg:
            aux = range(self.NM.settings['Generators'] +
                        self.NM.hydropower['Number']+1,
                        self.NM.settings['Generators'] +
                        self.NM.hydropower['Number']+1 +
                        self.NM.RES['Number'])
        elif 'Hydro' in varg:
            aux = range(self.NM.settings['Generators']+1,
                        self.NM.settings['Generators'] +
                        self.NM.hydropower['Number']+1)
        else:
            aux = range(1, self.NM.generationE['Number']+1)

        value = 0
        for xn in aux:
            value += self.get_Generation(mod, xn, *varg, **kwarg)

        return value

    def get_RES(self, mod, index, *varg, **kwarg):
        ''' Spilled kWh of RES for the whole period'''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(mod, *varg, **kwarg)

        xg = index-1
        value = 0
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += ((self.NM.RES['Max'][xg]*self.NM.scenarios['RES']
                         [self.NM.resScenario[xg][xh][1]+xt] -
                         mod.vGen[self.NM.resScenario[xg][xh][0], xt].value) *
                        auxweight[xt])
            value += acu*auxOF[xh]
        value *= self.NM.networkE.graph['baseMVA']

        return value

    def get_AllRES(self, mod, *varg, **kwarg):
        ''' Total RES spilled for the whole period '''
        value = 0
        for xr in range(self.NM.RES['Number']):
            value += self.get_RES(mod, xr+1, *varg, **kwarg)

        return value

    # Run integrated pyene model
    def run(self, mod):
        # Build pyomo model
        mod = self.build_Mod(self.EM, self.NM, mod)

        # Run pyomo model
        (mod, results) = self.Run_Mod(mod, self.EM, self.NM)

        return mod

    # Build pyomo model
    def build_Mod(self, EM, NM, mod):
        # Declare pyomo model

        #                                 Sets                                #
        mod = EM.getSets(mod)
        mod = NM.getSets(mod)
        mod = self.getSets(mod)

        #                           Model Variables                           #
        mod = EM.getVars(mod)
        mod = NM.getVars(mod)

        #                              Parameters                             #
        mod = EM.getPar(mod)
        mod = NM.getPar(mod)
        mod = self.getPar(mod)

        return mod

    # Run pyomo model
    def Run_Mod(self, mod, EM, NM):
        # Finalise model
        mod = self._AddPyeneCons(EM, NM, mod)

        #                          Objective function                         #
        self.OFaux = self._Calculate_OFaux(EM, NM)

        mod.OF = Objective(rule=self.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')
        # Print
        results = opt.solve(mod)

        return (mod, results)

    def _AddPyeneCons(self, EM, NM, mod):
        #                      Model additional variables                     #
        mod = self.getVars(mod)

        #                             Constraints                             #
        mod = EM.addCon(mod)
        mod = NM.addCon(mod)
        mod = self.addCon(mod)

        return mod

    def _Calculate_OFaux(self, EM, NM):
        WghtAgg = 0+EM.Weight['Node']
        OFaux = np.ones(len(NM.connections['set']), dtype=float)
        xp = 0
        for xn in range(EM.LL['NosBal']+1):
            aux = EM.tree['After'][xn][0]
            if aux != 0:
                for xb in range(aux, EM.tree['After'][xn][1]+1):
                    WghtAgg[xb] *= WghtAgg[xn]
            else:
                OFaux[xp] = WghtAgg[xn]
                xp += 1

        return OFaux

    # Run Energy and network combined model
    def Print_ENSim(self, mod, EM, NM):
        # Print results
        EM.print(mod)
        NM.print(mod)
        print('Water outputs:')
        for xn in mod.sNodz:
            for xv in mod.sVec:
                aux = mod.WOutFull[xn, xv].value
                print("%8.4f " % aux, end='')
            print('')
        print('Water inputs:')
        if type(mod.WInFull) is np.ndarray:
            print(mod.WInFull)
        else:
            for xn in mod.sNodz:
                for xv in mod.sVec:
                    aux = mod.WInFull[xn, xv].value
                    print("%8.4f " % aux, end='')
                print('')

    # Print initial assumptions
    def _CheckInE(self, EM):
        if EM.fRea:
            print('Loading resolution tree from file')
        else:
            print('Predefined resolution tree')
            print(EM.Input_Data)

    # Print initial assumptions
    def _CheckInN(self, NM):
        if NM.settings['Losses']:
            print('Losses are considered')
        else:
            print('Losses are neglected')
        if NM.settings['Feasibility']:
            print('Feasibility constrants are included')
        else:
            print('Feasibility constraints are neglected')
        print('Demand multiplyiers ', NM.settings['Demand'])
        print('Secuity: ', NM.settings['Security'])

    # Read time series for demand and RES
    def ReadTimeS(self, FileName):
        MODEL_JSON = os.path.join(os.path.dirname(__file__), '..', 'json',
                                  FileName)
        Profile_Data = json.load(open(MODEL_JSON))

        # Get number of profiles
        RESTypes = Profile_Data['metadata']['title']
        NoRESP = len(RESTypes)

        # Search for demand profile
        if "Demand" in RESTypes:
            DemandProfiles = Profile_Data['Demand']['Values']
            BusDem = Profile_Data['Demand']['Bus']
            LinkDem = Profile_Data['Demand']['Links']
            NoDemPeriod = len(Profile_Data['Demand']['Period'])
            RESTypes.remove('Demand')
            NoRESP -= 1
        else:
            DemandProfiles = [1]
            NoDemPeriod = 1
            BusDem = 'All'
            LinkDem = 0

        # Check if there is generation data
        if NoRESP == 0:
            # Remove demand data
            LLRESType = 0
            LLRESPeriod = 0
            RESProfs = 0
        else:
            # Location of each type of RES generation technology
            LLRESType = np.zeros((NoRESP, 2), dtype=int)
            NoRES = -1
            xt = -1
            sRes = np.zeros((NoRESP, 3), dtype=int)
            NoLink = 0
            for xr in RESTypes:
                xt += 1
                # Get number of periods and links
                sRes[xt][0] = len(Profile_Data[xr]['Period'])
                sRes[xt][1] = max(Profile_Data[xr]['Links'])
                sRes[xt][2] = len(Profile_Data[xr]['Bus'])
                LLRESType[xt][:] = [NoRES+1, NoRES+sRes[xt][0]]
                NoRES += sRes[xt][0]
                NoLink += sRes[xt][1]

            # Location of data for each period
            LLRESPeriod = np.zeros((NoRES+1, 2), dtype=int)
            RESBus = np.zeros(NoRES, dtype=int)
            RESLink = np.zeros(NoRES, dtype=int)
            xL = -1
            acu = -1
            for xt in range(NoRESP):
                for xp in range(sRes[xt][0]):
                    xL += 1
                    LLRESPeriod[xL][:] = [acu+1, acu+sRes[xt][1]]
                    acu += sRes[xt][1]

            # RES genertaion profiles
            if sRes[0][0]*sRes[0][1] == 1:
                Nohr = len(Profile_Data[RESTypes[0]]['Values'])
            else:
                Nohr = len(Profile_Data[RESTypes[0]]['Values'][0])
            RESProfs = np.zeros((acu+1, Nohr), dtype=float)
            acu = [0, 0, 0]
            xt = -1
            for xr in RESTypes:
                xt += 1
                aux = Profile_Data[xr]['Values']
                RESProfs[acu[0]:acu[0]+sRes[xt][0]*sRes[xt][1]][:] = aux
                acu[0] += sRes[xt][0]*sRes[xt][1]
                for xL in range(sRes[xt][2]):
                    RESBus[acu[1]] = Profile_Data[xr]['Bus'][xL]
                    RESLink[acu[1]] = Profile_Data[xr]['Links'][xL]+acu[2]
                    acu[1] += 1
                acu[2] += sRes[xt][1]

            return (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES,
                    NoRESP, LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink,
                    NoLink, Nohr)

    def pyene2pypsa(self, xscen):
        '''Convert pyene files to pypsa format'''
        # Create pypsa network
        try:
            nu = pypsa.Network()
        except ImportError:
            return (0, False)

        nu.set_snapshots(range(self.NM.settings['NoTime']))
        baseMVA = self.NM.networkE.graph['baseMVA']

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
            x - coordinates
            y - coordinates
            carrier - 'AC' or 'DC'

        Implemented attributes:
            Name auxtxtN+str(xn)
            v_nom - Nominal voltage
            v_mag_pu_set - per unit voltage set point
            v_mag_pu_min - per unit minimum voltage
            v_mag_pu_max - per unit maximum voltage
            auxtxtN = 'Bus'  # Generic name for the nodes
        '''
        PVBus = np.zeros(self.NM.networkE.number_of_nodes(), dtype=float)
        aux = (self.NM.generationE['Number']-self.NM.hydropower['Number'] -
               self.NM.RES['Number'])
        for xn in self.NM.generationE['Data']['GEN_BUS'][0:aux]:
            if self.NM.networkE.node[xn]['BUS_TYPE'] == 2:
                PVBus[xn-1] = self.NM.generationE['Data']['VG'][xn]
        for xn in self.NM.networkE.node:
            if self.NM.networkE.node[xn]['BASE_KV'] == 0:
                aux1 = 1
            else:
                aux1 = self.NM.networkE.node[xn]['BASE_KV']
            if self.NM.networkE.node[xn]['BUS_TYPE'] == 2 or \
               self.NM.networkE.node[xn]['BUS_TYPE'] == 3:
                aux2 = self.NM.networkE.node[xn]['VM']
            else:
                aux2 = PVBus[xn-1]
            nu.add('Bus', auxtxtN+str(xn),
                   v_nom=aux1,
                   v_mag_pu_set=aux2,
                   v_mag_pu_min=self.NM.networkE.node[xn]['VMIN'],
                   v_mag_pu_max=self.NM.networkE.node[xn]['VMAX'])

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
        # Fuel based generation
        aux = (self.NM.generationE['Number']-self.NM.hydropower['Number'] -
               self.NM.RES['Number'])
        xg = -1
        for xn in self.NM.generationE['Data']['GEN_BUS'][0:aux]:
            xg += 1
            if self.NM.networkE.node[xn]['BUS_TYPE'] == 1:
                aux1 = 'PQ'
            elif self.NM.networkE.node[xn]['BUS_TYPE'] == 2:
                aux1 = 'PV'
            else:
                aux1 = 'Slack'
            aux2 = (self.NM.generationE['Data']['PMAX'][xg] +
                    self.NM.generationE['Data']['PMIN'][xg])/2*baseMVA
            if self.NM.generationE['Costs']['MODEL'][xg] == 1:
                xi = 2
                while self.NM.generationE['Costs']['COST'][xg][xi] <= aux2:
                    xi += 2
                aux3 = ((self.NM.generationE['Costs']['COST'][xg][xi+1] -
                         self.NM.generationE['Costs']['COST'][xg][xi-1]) /
                        (self.NM.generationE['Costs']['COST'][xg][xi] -
                         self.NM.generationE['Costs']['COST'][xg][xi-2]))
            else:
                aux3 = 0
                for xi in range(self.NM.generationE['Costs']['NCOST'][xg]-1):
                    aux3 += ((self.NM.generationE['Costs']['NCOST'][xg]-xi-1) *
                             self.NM.generationE['Costs']['COST'][xg][xi] *
                             aux2**(self.NM.generationE['Costs']['NCOST']
                                    [xg]-xi-2))
            nu.add('Generator', auxtxtG+str(xg+1),
                   bus=auxtxtN+str(xn),
                   control=aux1,
                   p_nom_max=self.NM.generationE['Data']['PMAX'][xg]*baseMVA,
                   p_set=self.NM.generationE['Data']['PG'][xg],
                   q_set=self.NM.generationE['Data']['QG'][xg],
                   marginal_cost=aux3
                   )

        # Renewable generation
        aux = self.NM.generationE['Number']-self.NM.RES['Number']
        aux1 = 'PQ'
        xg = aux-1
        xr = -1
        yres = np.zeros(self.NM.settings['NoTime'], dtype=float)
        for xn in (self.NM.generationE['Data']['GEN_BUS']
                   [aux:self.NM.generationE['Number']]):
            xg += 1
            xr += 1
            for xt in range(self.NM.settings['NoTime']):
                yres[xt] = (self.NM.scenarios['RES']
                            [self.NM.resScenario[xr][xscen][1]+xt])
            nu.add('Generator', auxtxtG+str(xg+1),
                   bus=auxtxtN+str(xn),
                   control=aux1,
                   p_nom_max=yres,
                   p_nom=self.NM.RES['Max'][xr],
                   p_set=self.NM.generationE['Data']['PG'][xg],
                   q_set=self.NM.generationE['Data']['QG'][xg],
                   marginal_cost=self.NM.generationE['Costs']['COST'][xg][0]
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
        ydemP = np.zeros(self.NM.settings['NoTime'], dtype=float)
        ydemQ = np.zeros(self.NM.settings['NoTime'], dtype=float)
        for xn in self.NM.networkE.node:
            if self.NM.demandE['PD'][xn-1] != 0:
                xL += 1
                for xt in range(self.NM.settings['NoTime']):
                    aux = (self.NM.scenarios['Demand']
                           [self.NM.busScenario[xn-1][xscen]+xt])
                    ydemP[xt] = self.NM.demandE['PD'][xn-1]*aux
                    ydemQ[xt] = self.NM.demandE['QD'][xn-1]*aux
                nu.add('Load', auxtxtLd+str(xL),
                       bus=auxtxtN+str(xn),
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
        xb = 0
        for (xf, xt) in self.NM.networkE.edges:
            if self.NM.networkE[xf][xt]['TAP'] == 0:
                xb += 1
                auxpu = (nu.buses['v_nom']['Bus{}'.format(xf)]**2 /
                         self.NM.networkE.graph['baseMVA'])
                nu.add('Line', auxtxtL+str(xb),
                       bus0=auxtxtN+str(xf),
                       bus1=auxtxtN+str(xt),
                       x=self.NM.networkE[xf][xt]['BR_X']*auxpu,
                       r=self.NM.networkE[xf][xt]['BR_R']*auxpu,
                       s_nom=self.NM.networkE[xf][xt]['RATE_A']
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
        xb = 0
        for (xf, xt) in self.NM.networkE.edges:
            if self.NM.networkE[xf][xt]['TAP'] != 0:
                xb += 1
                nu.add('Transformer', auxtxtT+str(xb),
                       bus0=auxtxtN+str(xf),
                       bus1=auxtxtN+str(xt),
                       model='pi',
                       x=self.NM.networkE[xf][xt]['BR_X'],
                       r=self.NM.networkE[xf][xt]['BR_R'],
                       b=self.NM.networkE[xf][xt]['BR_B'],
                       s_nom=self.NM.networkE[xf][xt]['RATE_A'],
                       tap_ratio=self.NM.networkE[xf][xt]['TAP'],
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


class RESprofiles():
    ''' Produce time series for different RES '''
    def __init__(self):
        '''Inital assumptions'''

        # Solar pannels
        self.solar = {}
        self.solar['Latitude'] = 52  # Latitude
        self.solar['Albedo'] = 0.2  # Reflectance coefficient
        # Collector type: 1:Fixed 2: Single axis tracking 3: two axis tracking
        self.solar['Collector'] = 1
        self.solar['face'] = 0  # Collector face angle
        self.solar['tilt'] = self.solar['Latitude']  # Collector tilt angle
        self.solar['day'] = [1]  # Day to simulate
        self.solar['hour'] = [1]

        # Wind generators
        self.wind = {}
        self.wind['CutIN'] = 3  # Cut in wind speed (m/s)
        self.wind['Rated'] = 10  # Rated wind speed (m/s)
        self.wind['cutOUT'] = 24  # cut out wind speed (m/s)
        self.wind['Park'] = 0.95  # Wind park efficiency
        self.wind['Model'] = 1  # 1: Linear, 2: Parabolic, 3: Cubic

    def buildPV(self, direct, diffuse):
        ''' Produce PV generation multiplier '''
        datalength = len(direct)
        PVMultiplier = np.zeros(datalength, dtype=float)
        # Changing to radians
        latitude = math.radians(self.solar['Latitude'])
        face = math.radians(self.solar['face'])
        tilt = math.radians(self.solar['tilt'])

        # Produce multipliers
        xt = 0
        for xd in self.solar['day']:
            for xh in self.solar['hour']:
                PVMultiplier[xt] = \
                    self.getPVGeneration(direct[xt], diffuse[xt], xd, xh,
                                         latitude, face, tilt)
                xt += 1

        return PVMultiplier

    def buildWind(self, val):
        ''' Produce wind generation multiplier '''
        datalength = len(val)
        windMultiplier = np.zeros(datalength, dtype=float)

        # Produce multipliers
        for xw in range(datalength):
            # The turbines do not generate for winds under the cut-in speed
            if val[xw] > self.wind['CutIN']:
                # Produce at rated capacity
                if val[xw] > self.wind['Rated']:
                    if val[xw] <= self.wind['cutOUT']:
                        windMultiplier[xw] = self.wind['Park']
                else:
                    # Use function
                    windMultiplier[xw] = self.wind['Park'] * \
                        (val[xw]**self.wind['Model'] -
                         self.wind['CutIN']**self.wind['Model']) / \
                        (self.wind['Rated']**self.wind['Model'] -
                         self.wind['CutIN']**self.wind['Model'])

        return windMultiplier

    def getPVAngles(self, xd, xh, latitude, face, tilt):
        ''' Calculate solar angles '''
        # Solar declination
        Delta = 0.409279*math.sin(2*math.pi/365*(xd-81))

        # Hour angle
        HourAngle = math.pi*(1-xh/12)

        # Altitude angle
        Beta = math.asin(math.cos(latitude)*math.cos(Delta) *
                         math.cos(HourAngle)+math.sin(latitude) *
                         math.sin(Delta))
        # Azimuth
        azimuth = math.asin(math.cos(Delta)*math.sin(HourAngle)/math.cos(Beta))
        if math.cos(HourAngle) < math.tan(Delta)/math.tan(latitude):
            azimuth = math.pi-azimuth

        # Cosine of incident angle between the sun and the collectro's face
        Theta = math.cos(Beta)*math.cos(azimuth-face)*math.sin(tilt) + \
            math.sin(Beta)*math.cos(tilt)
        if Theta < 0:
            Theta = 0
        if Beta < 0:
            Beta = 0

        return (Beta, Theta, Delta)

    def getPVGeneration(self, direct, diffuse, xd, xh, latitude, face, tilt):
        ''' Get power generation kW/m2 '''
        # Get solar angles
        [Beta, Theta, Delta] = self.getPVAngles(xd, xh, latitude, face, tilt)

        # Choose collector model
        if self.solar['Collector'] == 1:
            # Fixed collector
            aux1 = (1-math.cos(tilt))/2
            aux2 = Theta
        elif self.solar['Collector'] == 2:
            # One axis tracking
            aux1 = (1-math.cos(math.pi/2-Beta+Delta))/2
            aux2 = math.cos(Delta)
        elif self.solar['Collector'] == 3:
            # Two axis tracking
            aux1 = (1-math.cos(math.pi/2-Beta))/2
            aux2 = 1

        PV = aux1*(diffuse+self.solar['Albedo']*(diffuse+direct))+aux2*direct
        return PV/1000

    def setPVday(self):
        ''' Set a 24h day '''
        self.solar['hour'] = np.zeros(24, dtype=int)
        for xt in range(24):
            self.solar['hour'][xt] = xt+1


class pyeneHDF5Settings():
    '''pytables auxiliary'''
    class PyeneHDF5Settings(IsDescription):
        '''Simulation settings'''
        treefile = StringCol(itemsize=30, dflt=" ", pos=0)  # File name (tree)
        networkfile = StringCol(itemsize=30, dflt=" ", pos=1)  # FIle (network)
        time = Int16Col(dflt=1, pos=2)  # generatop
        scenarios = Int16Col(dflt=1, pos=3)  # scenarios
        NoHydro = Int16Col(dflt=1, pos=4)  # hydropower plants
        NoPump = Int16Col(dflt=1, pos=5)  # pumps
        NoRES = Int16Col(dflt=1, pos=6)  # RES generators

    class PyeneHDF5Devices(IsDescription):
        '''Characteristics of devices '''
        location = Int16Col(dflt=1, pos=0)  # Location
        max = Float32Col(dflt=1, pos=1)  # Capacity
        cost = Float32Col(dflt=4, pos=2)  # Cost/value
        link = Int16Col(dflt=1, pos=3)  # Location

    class PyeneHDF5Results(IsDescription):
        time = Int16Col(dflt=1, pos=0)  # time period
        generation = Float32Col(dflt=1, pos=1)  # generatopm
        hydropower = Float32Col(dflt=1, pos=2)  # generatopm
        RES = Float32Col(dflt=1, pos=3)  # generatopm
        demand = Float32Col(dflt=1, pos=4)  # total demand
        pump = Float32Col(dflt=1, pos=5)  # use of pumps
        loss = Float32Col(dflt=1, pos=6)  # short integer
        curtailment = Float32Col(dflt=1, pos=7)  # short integer
        spill = Float32Col(dflt=1, pos=8)  # short integer

    def SaveSettings(self, fileh, EN, conf, root):
        HDF5group = fileh.create_group(root, 'Core')
        HDF5table = fileh.create_table(HDF5group, "table",
                                       self.PyeneHDF5Settings)
        HDF5row = HDF5table.row
        HDF5row['treefile'] = conf.TreeFile
        HDF5row['networkfile'] = conf.NetworkFile
        HDF5row['time'] = conf.Time
        HDF5row['scenarios'] = EN.NM.scenarios['Number']
        HDF5row['NoHydro'] = conf.NoHydro
        HDF5row['NoPump'] = conf.NoPump
        HDF5row['NoRES'] = conf.NoRES
        HDF5row.append()
        HDF5table.flush()

        HDF5table = fileh.create_table(HDF5group, "Hydpopower_plants",
                                       self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for xh in range(conf.NoHydro):
            HDF5row['location'] = EN.NM.hydropower['Bus'][xh]
            HDF5row['max'] = EN.NM.hydropower['Max'][xh]
            HDF5row['cost'] = EN.NM.hydropower['Cost'][xh]
            HDF5row['link'] = EN.NM.hydropower['Link'][xh]
            HDF5row.append()
        HDF5table.flush()

        HDF5table = fileh.create_table(HDF5group, "Pumps",
                                       self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for xh in range(conf.NoPump):
            HDF5row['location'] = EN.NM.pumps['Bus'][xh]
            HDF5row['max'] = EN.NM.pumps['Max'][xh]
            HDF5row['cost'] = EN.NM.pumps['Value'][xh]
            HDF5row.append()
        HDF5table.flush()

        HDF5table = fileh.create_table(HDF5group, "RES_generators",
                                       self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for xh in range(conf.NoRES):
            HDF5row['location'] = EN.NM.RES['Bus'][xh]
            HDF5row['max'] = EN.NM.RES['Max'][xh]
            HDF5row['cost'] = EN.NM.RES['Cost'][xh]
            HDF5row['link'] = EN.NM.RES['Link'][xh]
            HDF5row.append()
        HDF5table.flush()

    def saveResults(self, fileh, EN, mod, root, SimNo):
        HDF5group = fileh.create_group(root, 'Simulation_{:05d}'.format(SimNo))
        HDF5aux = np.zeros((EN.NM.scenarios['NoDem'],
                            EN.NM.settings['NoTime']), dtype=float)
        xp = 0
        for xs in range(EN.NM.scenarios['NoDem']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['Demand'][xp]
                xp += 1
        fileh.create_array(HDF5group, "Demand_profiles", HDF5aux)

        HDF5aux = np.zeros((EN.NM.scenarios['NoRES'],
                            EN.NM.settings['NoTime']), dtype=float)
        xp = 0
        for xs in range(EN.NM.scenarios['NoRES']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['RES'][xp] * \
                    EN.NM.networkE.graph['baseMVA']
                xp += 1
        fileh.create_array(HDF5group, "RES_profiles", HDF5aux)

        # Hydropower allowance
        aux = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        if type(mod.WInFull) is np.ndarray:
            if EN.EM.settings['Vectors'] == 1:
                aux[0] = mod.WInFull[1]
            else:
                for xv in mod.sVec:
                    aux[xv] = mod.WInFull[1][xv]
        else:
            for xv in mod.sVec:
                aux[xv] = mod.WInFull[1, xv].value

        fileh.create_array(HDF5group, "Hydro_Allowance", aux)

        hp_marginal = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        for xi in range(EN.EM.settings['Vectors']):
            hp_marginal[xi] = EN.get_HydroMarginal(mod, xi+1)
        fileh.create_array(HDF5group, "Hydro_Marginal", hp_marginal)

        for xs in range(EN.NM.scenarios['Number']):
            HDF5table = fileh.create_table(HDF5group,
                                           "Scenario_{:02d}".format(xs),
                                           self.PyeneHDF5Results)
            HDF5row = HDF5table.row
            for xt in range(EN.NM.settings['NoTime']):
                HDF5row['time'] = xt
                HDF5row['generation'] = \
                    EN.get_AllGeneration(mod, 'Conv', 'snapshot', times=[xt],
                                         scens=[xs])
                HDF5row['hydropower'] = \
                    EN.get_AllGeneration(mod, 'Hydro', 'snapshot', times=[xt],
                                         scens=[xs])
                HDF5row['RES'] = EN.get_AllGeneration(mod, 'RES', 'snapshot',
                                                      times=[xt], scens=[xs])
                HDF5row['spill'] = EN.get_AllRES(mod, 'snapshot', times=[xt],
                                                 scens=[xs])
                HDF5row['demand'] = EN.get_AllDemand(mod, 'snapshot',
                                                     times=[xt], scens=[xs])
                HDF5row['pump'] = EN.get_AllPumps(mod, 'snapshot', times=[xt],
                                                  scens=[xs])
                HDF5row['loss'] = EN.get_AllLoss(mod, 'snapshot', times=[xt],
                                                 scens=[xs])
                HDF5row['curtailment'] = \
                    EN.get_AllDemandCurtailment(mod, 'snapshot', times=[xt],
                                                scens=[xs])
                HDF5row.append()
            HDF5table.flush()
