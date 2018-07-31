# Python Energy & Network Engine
"""
Created on Thu Mar 29 14:04:58 2018

The python energy and networks engine (pyene) provides a portfolio of tools for
assessing energy systems, as well as interfaces to exchange information with
the FutureDAMS framework and other engines and software.

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
# convert in or long into float before performing divissions
from __future__ import division
from pyomo.core import ConcreteModel, Constraint, Objective, Suffix, Var, \
                       NonNegativeReals, minimize
from pyomo.opt import SolverFactory
import numpy as np
from .pyeneN import ENetworkClass as dn  # Network component
from .pyeneE import EnergyClass as de  # Energy component
import json
import os


class pyeneConfig():
    ''' Overall default configuration '''
    def __init__(self):
        from .pyeneE import pyeneEConfig
        from .pyeneN import pyeneNConfig
        from .pyeneR import pyeneRConfig

        self.EN = ENEConfig()  # pyene
        self.EM = pyeneEConfig()  # pyeneE - Energy
        self.NM = pyeneNConfig()  # pyeneN - Networks
        self.RM = pyeneRConfig()  # pyeneR - Renewables


class ENEConfig():
    ''' Default consifuration for the integrated model '''
    def __init__(self):
        # Chose to load data from file
        self.Penalty = 1000000


class pyeneClass():
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = ENEConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def _AddPyeneCons(self, EM, NM, mod):
        ''' Model additional variables '''
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

    def _CheckInE(self, EM):
        ''' Print initial assumptions '''
        if EM.settings['File']:
            print('Loading resolution tree from file')
        else:
            print('Predefined resolution tree')
            print(EM.Input_Data)

    def _CheckInN(self, NM):
        ''' Print initial assumptions '''
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

    def addCon(self, m):
        ''' Adding pyomo constraints'''
        # Link water consumption from both models
        m.cEMNM = Constraint(m.sLLEN, m.sVec, rule=self.cEMNM_rule)
        # Allow collection of ual values
        m.dual = Suffix(direction=Suffix.IMPORT)

        return m

    def build_Mod(self, EM, NM, mod):
        ''' Build pyomo model '''
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

    def cEMNM_rule(self, m, xL, xv):
        ''' Water use depends on the electricity system '''
        return (m.WOutFull[m.LLENM[xL][0], xv] ==
                self.NM.networkE.graph['baseMVA'] *
                sum(m.vGen[m.LLENM[xL][1]+xv, xt] *
                    self.NM.scenarios['Weights'][xt] for xt in m.sTim))

    def ESim(self, conf):
        ''' Energy only optimisation '''
        # Get energy object
        EM = de(conf.EM)

        # Initialise
        EM.initialise()

        # Build LP model
        EModel = self.SingleLP(EM)

        #                          Objective function                         #
        EModel.OF = Objective(rule=EM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print
        results = opt.solve(EModel)

        return (EM, EModel, results)

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

    def get_AllHydro(self, mod):
        ''' Get surplus kWh from all hydropower plants '''
        value = 0
        for xi in range(self.EM.settings['Vectors']):
            value += mod.WOutFull[1, xi].value

        return value

    def get_AllLoss(self, mod, *varg, **kwarg):
        '''Get the total losses'''
        value = 0
        if self.NM.settings['Losses']:
            for xb in mod.sBra:
                value += self.get_Loss(mod, xb+1, *varg, **kwarg)

        return value

    def get_AllPumps(self, mod, *varg, **kwarg):
        ''' Get kWh consumed by all pumps '''
        value = 0
        for xp in range(self.NM.pumps['Number']):
            value += self.get_Pump(mod, xp+1, *varg, **kwarg)

        return value

    def get_AllRES(self, mod, *varg, **kwarg):
        ''' Total RES spilled for the whole period '''
        value = 0
        for xr in range(self.NM.RES['Number']):
            value += self.get_RES(mod, xr+1, *varg, **kwarg)

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

    def get_Hydro(self, mod, index):
        ''' Get surplus kWh from specific site '''
        HydroValue = mod.WOutFull[1, index-1].value

        return HydroValue

    def get_HydroFlag(self, mod, index):
        ''' Get surplus kWh from specific site '''
        cobject = getattr(mod, 'cSoCBalance')
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

    def get_HydroMarginal(self, mod, index):
        ''' Get marginal costs for specific hydropower plant '''
        cobject = getattr(mod, 'cSoCBalance')
        aux = mod.dual.get(cobject[1, index-1])
        if aux is None:
            HydroValue = 0
        else:
            HydroValue = -1*int(mod.dual.get(cobject[1, index-1]))

        return HydroValue

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

    def get_MeanHydroMarginal(self, mod):
        value = 0
        for xi in range(self.EM.settings['Vectors']):
            value += self.get_HydroMarginal(mod, xi+1)
        return value/self.EM.settings['Vectors']

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

    def getClassInterfaces(self):
        ''' Return calss to interface with pypsa and pypower'''
        from .pyeneI import EInterfaceClass
        return EInterfaceClass()

    def getClassOutputs(self):
        ''' Return calss to produce outputs in H5 files'''
        from .pyeneO import pyeneHDF5Settings
        return pyeneHDF5Settings()

    def getClassRenewables(self, obj=None):
        ''' Return calss to produce RES time series'''
        from .pyeneR import RESprofiles
        return RESprofiles(obj)

    def getPar(self, m):
        ''' Add pyomo parameters'''
        m.LLENM = self.LLENM

        return m

    def getSets(self, m):
        '''Add pyomo sets'''
        m.sLLEN = range(self.NoLL)

        return m

    def getVars(self, m):
        ''' Add pyomo variables '''
        # Converting some parameters to variables
        del m.WOutFull
        m.WOutFull = Var(m.sNodz, m.sVec, domain=NonNegativeReals,
                         initialize=0.0)
        return m

    def initialise(self, conf):
        ''' Initialise energy and networks simulator '''
        # Creat objects
        self.EM = de(conf.EM)

        self.NM = dn(conf.NM)

        # Adding hydro to the energy balance tree
        self.EM.settings['Fix'] = True,  # Force a specific number of vectors
        self.EM.settings['Vectors'] = self.NM.hydropower['Number']  # Number

        # Initialise energy balance model
        self.EM.initialise()

        # Get number of required network model instances
        NoNM = (1+self.EM.tree['Time'][self.EM.size['Periods']][1] -
                self.EM.tree['Time'][self.EM.size['Periods']][0])

        # Add time steps
        aux = self.EM.size['Scenarios']
        self.NM.scenarios['Number'] = aux

        self.NM.scenarios['Demand'] = \
            np.ones(self.NM.settings['NoTime']*self.NM.scenarios['NoDem'],
                    dtype=float)

        # Initialise RES
        if self.NM.RES['Number'] > 0:
            self.NM.scenarios['RES'] = \
                np.zeros(self.NM.settings['NoTime']*self.NM.scenarios['NoRES'],
                         dtype=float)

        # Initialise network model
        self.NM.initialise()

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

    def NSim(self, conf):
        ''' Network only optimisation '''
        # Get network object
        NM = dn(conf.NM)

        # Initialise
        NM.initialise()

        # Build LP model
        NModel = self.SingleLP(NM)

        #                          Objective function                         #
        NModel.OF = Objective(rule=NM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print results
        results = opt.solve(NModel)

        return (NM, NModel, results)

    def OF_rule(self, m):
        ''' Objective function for energy and networkd model'''
        return sum((sum(sum(m.vGCost[self.hGC[xh]+xg, xt] for xg in m.sGen) +
                        sum(m.vFea[self.hFea[xh]+xf, xt] for xf in m.sFea) *
                        self.Penalty for xt in m.sTim) -
                    sum(self.NM.pumps['Value'][xdl] *
                        self.NM.networkE.graph['baseMVA'] *
                        sum(m.vDL[self.hDL[xh]+xdl+1, xt] *
                            self.NM.scenarios['Weights'][xt]
                            for xt in m.sTim) for xdl in m.sDL)) *
                   self.OFaux[xh] for xh in self.h)

    def Print_ENSim(self, mod, EM, NM):
        ''' Print results '''
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

    def ReadTimeS(self, FileName):
        ''' Read time series for demand and RES '''
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

    def run(self, mod):
        ''' Run integrated pyene model '''
        # Build pyomo model
        mod = self.build_Mod(self.EM, self.NM, mod)

        # Run pyomo model
        (mod, results) = self.Run_Mod(mod, self.EM, self.NM)

        return mod

    def Run_Mod(self, mod, EM, NM):
        ''' Run pyomo model '''
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

    def set_Demand(self, index, value):
        ''' Set a demand profile '''
        aux1 = (index-1)*self.NM.settings['NoTime']
        aux2 = index*self.NM.settings['NoTime']

        self.NM.scenarios['Demand'][aux1:aux2] = value

    def set_GenCoFlag(self, index, value):
        ''' Adjust maximum output of generators '''
        if isinstance(value, bool):
            if value:
                # Maximum capacity
                self.NM.GenMax[index-1] = (self.NM.generationE['Data']['PMAX']
                                           [index-1]/self.NM.networkE.graph
                                           ['baseMVA'])
            else:
                # Switch off
                self.NM.GenMax[index-1] = 0
        else:
            # Adjust capacity
            # TODO: Costs should be recalculated for higher capacities
            value /= self.NM.networkE.graph['baseMVA']
            if value > self.NM.generationE['Data']['PMAX'][index-1]:
                import warnings
                warnings.warn('Increasing generation capacity is not'
                              ' supported yet')

            self.NM.GenMax[index-1] = value

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

    def SingleLP(self, ENM):
        ''' Get mathematical model '''
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
