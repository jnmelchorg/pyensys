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
from pyensys.engines.pyeneN import ENetworkClass as dn  # Network cengine
from pyensys.engines.pyeneE import EnergyClass as de  # Energy engine
from pyensys.engines.pyeneH import HydrologyClass as hn  # Hydrology engine
from pyensys.engines.pyeneR import RESprofiles as rn  # RES engine#
from pyensys.engines.pyeneO import pyeneSave
from pyensys.engines.pyene_Models import EnergyandNetwork, Networkmodel
import json
import os


class pyeneConfig():
    ''' Overall default configuration '''
    def __init__(self):
        from .pyeneE import pyeneEConfig
        from .pyeneN import pyeneNConfig
        from .pyeneR import pyeneRConfig
        from .pyeneH import pyeneHConfig
        from .pyeneO import pyeneOConfig

        self.EN = ENEConfig()  # pyene
        self.EM = pyeneEConfig()  # pyeneE - Energy
        self.NM = pyeneNConfig()  # pyeneN - Networks
        self.RM = pyeneRConfig()  # pyeneR - Renewables
        self.HM = pyeneHConfig()  # pyeneH - Hidrology
        self.OM = pyeneOConfig()  # pyeneO - Outputs


class ENEConfig():
    ''' Default configuration for the integrated model '''
    def __init__(self):
        # Chose to load data from file
        self.Penalty = 1000000
        # Option to overwrite parameters
        self.Overwrite = {
                'DemandX': [],  # Position of demand profiles to overwrite
                'DemandL': [],  # Link position-->profile
                'Demand': [],  # New demand time-series
                'RESX': [],  # Position of RES profiles to overwrite
                'RESL': [],  # Link position-->profile
                'RES': [],  # New RES time-series
                'HydroX': [],  # Position of Hydropower values to overwrite
                'HydroL': [],  # Link position-->profile
                'Hydro': []  # New hydropower time-series
                }
        self.solverselection = {
            'pyomo': False,
            'glpk': True
        }


class pyeneClass():
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = ENEConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # Additional parameters
        self.s = {}
        self.p = {}

    def _AddPyeneCons(self, m):
        ''' Model additional variables '''
        m = self.addVars(m)

        #                             Constraints                             #
        m = self.EM.addCon(m)
        m = self.NM.addCon(m)
        m = self.HM.addCon(m)
        m = self.addCon(m)

        # IEEE paper control variables.
        m.pEeee = self.EM.Print['IEEE']

        return m

    def _Calculate_OFaux(self):
        WghtAgg = 0+self.EM.p['WghtFull']
        OFaux = np.ones(len(self.NM.get_ConS()), dtype=float)
        xp = 0
        for xn in range(self.EM.LL['NosBal']+1):
            aux = self.EM.tree['After'][xn][0]
            if aux != 0:
                for xb in range(aux, self.EM.tree['After'][xn][1]+1):
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
        # Link water consumption throughout different models
        if self.NM.hydropower['Number'] > 0:
            if self.HM.settings['Flag']:
                # Collect inputs for pyeneH from pyeneE and pyeneN
                m.cAHMIn1 = Constraint(self.s['LL'], self.EM.s['Vec'],
                                       rule=self.cAHMIn1_rule)
                if self.EM.settings['Vectors'] < self.p['NoHMin']:
                    m.cAHMIn2 = Constraint(self.s['LL'],
                                           range(self.EM.settings['Vectors'],
                                           self.p['NoHMin']), self.NM.s['Tim'],
                                           rule=self.cAHMIn2_rule)
                # Connect pyeneH and pyeneN
                if self.p['NoHydDown'] > 0:
                    m.cAHMOut1 = \
                        Constraint(self.s['LL'], range(self.p['NoHydDown']),
                                   self.NM.s['Tim'], rule=self.cAHMOut1_rule)

                m.cAHMOut2 = Constraint(self.s['LL'], range(self.p['NoHMout']),
                                        self.NM.s['Tim'],
                                        rule=self.cAHMOut2_rule)

            else:
                # Link pyeneE and pyeneN
                m.cAEMNM = Constraint(self.s['LL'], self.EM.s['Vec'],
                                      rule=self.cAEMNM_rule)

        m.dual = Suffix(direction=Suffix.IMPORT)

        return m

    def addPar(self, m):
        ''' Add pyomo parameters'''
        if self.HM.settings['Flag']:
            # The head may change per iteration
            # Hydro
            self.p['EffHydro'] = np.zeros(self.NM.hydropower['Number'],
                                          dtype=float)
            for xn in range(self.NM.hydropower['Number']):
                self.p['EffHydro'][xn] = self.HM.hydropower['Head'][xn] * \
                    self.HM.hydropower['Efficiency'][xn]*9.81/1000
            # Pumps
            self.p['EffPump'] = np.zeros(self.NM.pumps['Number'], dtype=float)
            for xn in range(self.NM.pumps['Number']):
                self.p['EffPump'][xn] = self.HM.pumps['Head'][xn] * \
                    self.HM.pumps['Efficiency'][xn]*9.81/1000

        return m

    def addSets(self, m):
        '''Add pyomo sets'''

        return m

    def addVars(self, m):
        ''' Add pyomo variables '''
        # Converting some parameters to variables
        del m.vEOut
        m.vEOut = Var(self.EM.s['Nodz'], self.EM.s['Vec'],
                      domain=NonNegativeReals, initialize=0.0)
        return m

    def build_Mod(self, m):
        ''' Build pyomo model '''

        # Sets
        m = self.EM.addSets(m)
        m = self.NM.addSets(m)
        m = self.HM.addSets(m)
        m = self.addSets(m)

        # Model Variables
        m = self.EM.addVars(m)
        m = self.NM.addVars(m)
        m = self.HM.addVars(m)

        # Parameters
        m = self.EM.addPar(m)
        m = self.NM.addPar(m)
        m = self.HM.addPar(m)
        m = self.addPar(m)

        return m

    def cAEMNM_rule(self, m, xL, xv):
        ''' Connecting  pyeneE and pyeneN (MW --> MW)'''
        # TODO: Send first half of the constraint to pyeneH
        return m.vEOut[self.p['pyeneE'][xL], xv] == \
            self.NM.In_From_EM(m, xL, xv)

    def cAHMIn1_rule(self, m, xL, xv):
        ''' Flows from pyeneE and pyeneHin (MW --> m^3/s)'''
        xp = self.p['pyeneHin'][xv][1]+xL*(self.NM.pumps['Number']+1)
        return m.vEOut[self.p['pyeneE'][xL], xv] == \
            sum((m.vHin[self.p['NoHMin']*xL+xv, xt] -
                 sum(m.vNPump[xp, xt] *
                     self.NM.ENetwork.get_Base() /
                     self.p['EffPump'][self.p['pyeneHin'][xv][1]]
                     for x in range(self.p['pyeneHin'][xv][0]))) *
                self.NM.scenarios['Weights'][xt]
                for xt in self.NM.s['Tim'])*self.p['EffHydro'][xv]

    def cAHMIn2_rule(self, m, xL, xv, xt):
        ''' Connecting  pyeneE and pyeneHin (MW --> m^3/s)'''
        return m.vHin[self.p['NoHMin']*xL+xv, xt] == \
            sum(m.vNPump[self.p['pyeneHin'][xv][1]+xL *
                         (self.NM.pumps['Number']+1)+1, xt] *
                self.NM.ENetwork.get_Base() /
                self.p['EffPump'][self.p['pyeneHin'][xv][1]]
                for x in range(self.p['pyeneHin'][xv][0]))

    def cAHMOut1_rule(self, m, xh, xv, xt):
        ''' pyeneH outputs taken from the aggregated downstream flows '''
        # Position of the hydropower plant
        xb = self.p['LLHydDown'][xv]
        # Node to be addressed
        xn = self.HM.hydropower['Node'][xb]-1

        return self.NM.In_From_HM(m, xh, xt, xb, self.p['EffHydro'][xb]) <= \
            sum(m.vHup[self.HM.p['ConRiver'][xh] +
                       self.HM.p['LLN2B1'][self.HM.p['LLN2B2'][xn, 3]+xd], xt]
                for xd in range(self.HM.p['LLN2B2'][xn, 2]))

    def cAHMOut2_rule(self, m, xh, xn, xt):
        ''' pyeneH outputs taken from nodal outputs '''
        # Position of hydro in pyeneH
        xb = self.p['LLHydOut'][xn][1]
        # Position of the pump
        xp = self.p['LLHPumpOut'][xn][1]

        return m.vHout[xn+xh*self.HM.nodes['OutNumber'], xt] >= \
            sum(self.NM.In_From_HM(m, xh, xt, xb, self.p['EffHydro'][xb])
                for x in range(self.p['LLHydOut'][xn][0])) + \
            sum(m.vNPump[1+xp+xh*(1+self.NM.pumps['Number']), xt] *
                self.NM.ENetwork.get_Base()/self.p['EffPump'][xp]
                for x in range(self.p['LLHPumpOut'][xn][0]))

    def CheckProfile(self, value):
        ''' Verify that the size of the profile makes sense'''
        if len(value) != self.NM.settings['NoTime']:
            if self.NM.settings['NoTime'] == 1:
                value = [np.mean(value)]
            elif self.NM.settings['NoTime'] == 2:
                value = [(sum(value[0:15])+sum(value[20:24]))/19,
                         np.mean(value[15:20])]
            else:
                raise RuntimeError('Incompatible demand profiles')

        return value

    def close_output_files(self):
        self.save_solution.close_output_files()

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

    def get_AllDemand(self, m, *varg, **kwarg):
        '''Get the demand'''
        # Specify buses
        if 'buses' in kwarg:
            auxbuses = kwarg.pop('buses')
        else:
            auxbuses = range(self.NM.ENetwork.get_NoBus())

        value = 0
        values = [0, 0]
        for xn in auxbuses:
            aux = self.get_Demand(m, xn+1, *varg, **kwarg)
            values[self.NM.ENetwork.Bus[xn].get_LT()] += aux
            value += aux

        return value, values

    def get_AllDemandCurtailment(self, m, *varg, **kwarg):
        '''Get the kWh that had to be curtailed from all buses'''
        # Specify buses
        if 'buses' in kwarg:
            auxbuses = kwarg.pop('buses')
        else:
            auxbuses = range(self.NM.ENetwork.get_NoBus())

        values = [0, 0]
        value = 0
        if self.NM.settings['Feasibility']:
            for xn in auxbuses:
                aux = self.get_DemandCurtailment(m, xn, *varg, **kwarg)
                values[self.NM.ENetwork.Bus[xn].get_LT()] += aux
            value = values[0]+values[1]

        return value, values

    def get_AllGeneration(self, m, *varg, **kwarg):
        ''' Get kWh for all generators for the whole period '''
        if 'All' in varg:
            aux = self.NM.Gen.get_GenAll()
        elif 'Conv' in varg:
            aux = self.NM.Gen.get_GenAllC()
        elif 'RES' in varg:
            aux = self.NM.Gen.get_GenAllR()
        elif 'Hydro' in varg:
            aux = self.NM.Gen.get_GenAllH()
        else:
            aux = self.NM.Gen.get_GenAll()

        value = 0
        for xn in aux:
            value += self.get_Generation(m, xn, *varg, **kwarg)

        return value

    def get_AllHydro(self, m):
        ''' Get surplus kWh from all hydropower plants '''
        value = 0
        for xi in range(self.EM.settings['Vectors']):
            value += m.vEOut[1, xi].value

        return value

    def get_AllLoss(self, m, *varg, **kwarg):
        '''Get the total losses'''
        value = 0
        if self.NM.settings['Losses']:
            for xb in self.NM.s['Bra']:
                value += self.get_Loss(m, xb, *varg, **kwarg)

        return value

    def get_AllPumps(self, m, *varg, **kwarg):
        ''' Get kWh consumed by all pumps '''
        value = 0
        for xp in range(self.NM.pumps['Number']):
            value += self.get_Pump(m, xp+1, *varg, **kwarg)

        return value

    def get_AllRES(self, m, *varg, **kwarg):
        ''' Total RES spilled for the whole period '''
        value = 0
        for xr in range(self.NM.RES['Number']):
            value += self.get_RES(m, xr+1, *varg, **kwarg)

        return value

    def get_Demand(self, m, bus, *varg, **kwarg):
        '''Get the kWh that had to be curtailed from a given bus'''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(*varg, **kwarg)

        value = 0
        xb = bus-1
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += (self.NM.busData[xb]*self.NM.scenarios['Demand']
                        [xt+self.NM.busScenario[xb][xh]])*auxweight[xt]
            value += acu*auxOF[xh]
        value *= self.NM.ENetwork.get_Base()

        return value

    def get_DemandCurtailment(self, m, bus, *varg, **kwarg):
        '''Get the kWh that had to be curtailed from a given bus'''
        if self.NM.p['LLFea1'][bus] == 0:
            return 0

        value = 0
        if isinstance(m, ConcreteModel):
            (auxtime, auxweight, auxscens,
             auxOF) = self.get_timeAndScenario(*varg, **kwarg)

            if self.NM.settings['Feasibility']:
                for xh in auxscens:
                    acu = 0
                    for xt in auxtime:
                        acu += auxweight[xt] * \
                            m.vNFea[self.NM.get_ConFea(xh) + \
                                    self.NM.p['LLFea2'][bus], xt].value
                    value += acu*auxOF[xh]
                value *= self.NM.ENetwork.get_Base()
        elif isinstance(m, EnergyandNetwork) or isinstance(m, Networkmodel):
            if m.FlagProblem and m.FlagFeasibility:
                LoadCurtailment = m.GetLoadCurtailmentNodes()
            else:
                LoadCurtailment = m.GetLoadCurtailmentSystemED()
            for xs in m.LongTemporalConnections:
                for xt in range(m.ShortTemporalConnections):
                    for k in range(m.NumberContingencies + 1):
                        value += LoadCurtailment[xs, xt, k, bus]

        return value

    def get_Generation(self, m, index, *varg, **kwarg):
        ''' Get kWh for a single generator '''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(*varg, **kwarg)

        value = 0
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += m.vNGen[self.NM.get_ConG(xh)+index, xt].value * \
                    auxweight[xt]
            value += acu*auxOF[xh]
        value *= self.NM.ENetwork.get_Base()

        return value

    def get_Hydro(self, m, index):
        ''' Get surplus kWh from specific site '''
        HydroValue = m.vEOut[1, index-1].value

        return HydroValue

    def get_HydroFlag(self, m, index):
        ''' Get surplus kWh from specific site '''
        cobject = getattr(m, 'cSoCBalance')
        aux = m.dual.get(cobject[1, index-1])
        if aux is None:
            HydroValue = False
        else:
            aux2 = -1*int(m.dual.get(cobject[1, index-1]))
            aux3 = self.Penalty/self.NM.ENetwork.get_Base()
            if aux2 > aux3:
                HydroValue = True
            else:
                HydroValue = False

        return HydroValue

    def get_HydroMarginal(self, m, index):
        ''' Get marginal costs for specific hydropower plant '''
        cobject = getattr(m, 'cSoCBalance')
        aux = m.dual.get(cobject[1, index-1])
        if aux is None:
            HydroValue = 0
        else:
            HydroValue = -1*int(m.dual.get(cobject[1, index-1]))

        return HydroValue

    def get_Loss(self, m, xb, *varg, **kwarg):
        '''Get losses for a given branch'''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(*varg, **kwarg)

        value = 0
        if self.NM.settings['Flag']:
            # If the electricity network has been modelled
            if self.NM.settings['Losses']:
                value = 0
                for xh in auxscens:
                    acu = 0
                    for xt in auxtime:
                        acu += m.vNLoss[self.NM.get_ConL(xh)+xb, xt].value * \
                            auxweight[xt]
                    value += acu*auxOF[xh]
                value *= self.NM.ENetwork.get_Base()
            elif self.NM.settings['Loss'] is not None:
                # TODO: Enable option
                import sys
                sys.exit('Options with network and predefined losses not yet enabled')

        elif self.NM.settings['Loss'] is not None:
            # If losses have been estimated
            import sys
            sys.exit('Options without network and predefined losses not yet enabled')

        import sys
        sys.exit('Just stop')
        return value

    def get_MeanHydroMarginal(self, m):
        value = 0
        for xi in range(self.EM.settings['Vectors']):
            value += self.get_HydroMarginal(m, xi+1)
        return value/self.EM.settings['Vectors']

    def get_NetDemand(self, m, auxFlags, *varg, **kwarg):
        ''' Get MWh consumed by pumps, loads, losses '''
        value = 0
        if auxFlags[0]:  # Demand
            value += self.get_AllDemand(m, *varg, **kwarg)

        if auxFlags[1]:  # Pumps
            value += self.get_AllPumps(m, *varg, **kwarg)

        if auxFlags[2]:  # Loss
            value += self.get_AllLoss(m, *varg, **kwarg)

        if auxFlags[3]:  # Curtailment
            value += self.get_AllDemandCurtailment(m, *varg, **kwarg)[0]

        if auxFlags[4]:  # Spill
            value += self.get_AllRES(m, *varg, **kwarg)

        return value

    def get_OFpart(self, m, xg, *varg, **kwarg):
        ''' Get components of the objective function '''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(*varg, **kwarg)

        value = sum(sum(m.vNGCost[self.NM.get_ConC(xh)+xg, xt].value
                        for xt in auxtime)*auxOF[xh] for xh in auxscens)

        return value

    def get_OFparts(self, m, auxFlags, *varg, **kwarg):
        # TODO: Validate
        ''' Get components of the objective function '''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(*varg, **kwarg)

        value = 0
        if auxFlags[0]:  # Conventional generation
            for x in self.NM.Gen.Conv:
                value += sum(sum(m.vNGCost[self.NM.get_ConC(xh)+x.get_vNGen(),
                                           xt].value for xt in
                                 auxtime)*auxOF[xh] for xh in auxscens)
        if auxFlags[1]:  # RES generation
            for x in self.NM.Gen.RES:
                value += sum(sum(m.vNGCost[self.NM.get_ConC(xh)+x.get_vNGen(),
                                           xt].value for xt in
                                 auxtime)*auxOF[xh] for xh in auxscens)

        if auxFlags[2]:  # Hydro generation
            for x in self.NM.Gen.Hydro:
                value += sum(sum(m.vNGCost[self.NM.get_ConC(xh)+x.get_vNGen(),
                                           xt].value for xt in
                                 auxtime)*auxOF[xh] for xh in auxscens)

        if auxFlags[3]:  # Pumps
            value -= sum(sum(self.NM.pumps['Value'][xdl] *
                             self.NM.ENetwork.get_Base() *
                             sum(m.vNPump[self.NM.get_ConP(xh)+xdl+1,
                                          xt].value
                                 for xt in auxtime)
                             for xdl in self.NM.s['Pump']) *
                         auxOF[xh] for xh in auxscens)

        if auxFlags[4]:  # Curtailment
            value += sum(sum(sum(m.vNFea[self.NM.get_ConFea(xh)+xf, xt].value
                                 for xf in self.NM.s['Fea'])*self.Penalty
                             for xt in auxtime) *
                         auxOF[xh] for xh in auxscens)

        return value

    def get_Pump(self, m, index, *varg, **kwarg):
        ''' Get kWh consumed by a specific pump '''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(*varg, **kwarg)
        value = 0
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += auxweight[xt] * \
                    m.vNPump[self.NM.get_ConP(xh)+index, xt].value
            value += acu*auxOF[xh]
        value *= self.NM.ENetwork.get_Base()

        return value

    def get_RES(self, m, index, *varg, **kwarg):
        ''' Spilled kWh of RES for the whole period'''
        (auxtime, auxweight, auxscens,
         auxOF) = self.get_timeAndScenario(*varg, **kwarg)

        xg = index-1
        value = 0
        for xh in auxscens:
            acu = 0
            for xt in auxtime:
                acu += ((self.NM.RES['Max'][xg]*self.NM.scenarios['RES']
                         [self.NM.resScenario[xg][xh]+xt] -
                         m.vNGen[self.NM.connections['Generation'][xh] +
                                 self.NM.Gen.get_vNGenR(xg), xt].value) *
                        auxweight[xt])
            value += acu*auxOF[xh]
        value *= self.NM.ENetwork.get_Base()

        return value

    def get_timeAndScenario(self, *varg, **kwarg):
        # Specify times
        if 'times' in kwarg:
            auxtime = kwarg.pop('times')
        else:
            auxtime = self.NM.s['Tim']

        # Remove weights
        if 'snapshot' in varg:
            auxweight = np.ones(len(self.NM.s['Tim']), dtype=int)
            auxOF = np.ones(len(self.NM.get_ConS()), dtype=int)
        else:
            auxweight = self.NM.scenarios['Weights']
            auxOF = self.OFaux

        # Specify scenario
        if 'scens' in kwarg:
            auxscens = kwarg.pop('scens')
        else:
            auxscens = self.NM.get_ConS()

        return (auxtime, auxweight, auxscens, auxOF)

    def getClassInterfaces(self):
        ''' Return calss to interface with pypsa and pypower'''
        from .pyeneI import EInterfaceClass

        return EInterfaceClass()

    def getClassOutputs(self, obj=None):
        ''' Return calss to produce outputs in H5 files'''
        from .pyeneO import pyeneHDF5Settings

        return pyeneHDF5Settings(obj)

    def getClassRenewables(self, obj=None):
        ''' Return calss to produce RES time series'''
        from .pyeneR import RESprofiles

        return RESprofiles(obj)

    def HSim(self, conf):
        ''' Hydrology only optimisation '''
        # Get network object
        HM = hn(conf.HM)

        # Initialise
        HM.initialise()

        # Build LP model
        HModel = self.SingleLP(HM)

        #                          Objective function                         #
        HModel.OF = Objective(rule=HM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print results
        results = opt.solve(HModel)

        return (HM, HModel, results)

    def initialise(self, conf=None, path=None):
        ''' Initialise energy and networks simulator '''
        if path is None:
            # Creat objects
            self.EM = de(conf.EM)

            self.NM = dn(conf.NM)

            # Adding hydro to the energy balance tree
            self.EM.settings['Fix'] = True,  # Force a specific number of vectors
            # Get number of vectors
            if conf.HM.settings['Flag']:
                # one per input node of the water network
                self.EM.settings['Vectors'] = len(conf.HM.nodes['In'])
            else:
                # One per hydropower plant
                self.EM.settings['Vectors'] = self.NM.hydropower['Number']
            self.s['Hydro'] = range(self.NM.hydropower['Number'])

            # Initialise energy balance model
            self.EM.initialise()

            # Get number of required network model instances
            NoNM = (1+self.EM.tree['Time'][self.EM.size['Periods']][1] -
                    self.EM.tree['Time'][self.EM.size['Periods']][0])

            # Add time steps
            aux = self.EM.size['Scenarios']
            self.NM.scenarios['Number'] = aux
            self.NM.scenarios['NoDem'] = aux


            if self.NM.scenarios['NoDem'] > 0:
                aux = self.NM.scenarios['Demand']
                self.NM.scenarios['Demand'] = \
                    np.ones(self.NM.settings['NoTime'] *
                            self.NM.scenarios['NoDem'], dtype=float)
                for x in range(len(aux)):
                    self.NM.scenarios['Demand'][x] = aux[x]

            # Initialise RES
            if self.NM.RES['Number'] > 0:
                self.NM.scenarios['RES'] = \
                    np.zeros(self.NM.settings['NoTime']*self.NM.scenarios['NoRES'],
                             dtype=float)

            # Initialise network model
            self.NM.initialise(rn(conf.RM))

            # Add connections between energy balance and network models
            self.NM.set_ConS(range(NoNM))
            auxFlow = np.zeros(NoNM, dtype=int)
            auxVol = np.zeros(NoNM, dtype=int)
            auxLoss = np.zeros(NoNM, dtype=int)
            auxGen = np.zeros(NoNM, dtype=int)
            auxCost = np.zeros(NoNM, dtype=int)
            auxPump = np.zeros(NoNM, dtype=int)
            auxFea = np.zeros(NoNM, dtype=int)
            # Location of each instance
            aux = self.NM.get_ConB()
            for xc in self.NM.get_ConS():
                auxFlow[xc] = xc*(self.NM.NoBranch)
                auxVol[xc] = xc*(self.NM.NoBuses)
                auxLoss[xc] = xc*(aux)
                auxGen[xc] = xc*(self.NM.Gen.get_NoGen())
                auxCost[xc] = xc*self.NM.Gen.get_NoGen()
                auxPump[xc] = xc*(self.NM.pumps['Number']+1)
                auxFea[xc] = xc*self.NM.NoFea

            self.NM.set_ConF(auxFlow)
            self.NM.set_ConV(auxVol)
            self.NM.set_ConL(auxLoss)
            self.NM.set_ConG(auxGen)
            self.NM.set_ConC(auxCost)
            self.NM.set_ConP(auxPump)
            self.NM.set_ConFea(auxFea)
            del auxFlow, auxVol, auxLoss, auxGen, auxPump, auxFea

            # Build LL to link the models through hydro consumption
            # Adding connections to pyeneE
            self.p['Number'] = NoNM
            self.s['LL'] = range(self.p['Number'])
            self.p['pyeneE'] = np.zeros(self.p['Number'], dtype=int)
            for xc in self.s['LL']:
                self.p['pyeneE'][xc] = \
                    self.EM.tree['Time'][self.EM.size['Periods']][0]+xc

            # Adding connections to pyeneN
            self.p['pyeneN'] = np.zeros(self.p['Number'], dtype=int)
            aux = self.NM.Gen.get_NoGen()+1
            for xc in self.s['LL']:
                self.p['pyeneN'][xc] = self.NM.get_ConG(xc)+aux

            # Create hydraulic data
            if conf.HM.settings['Flag']:
                # set periods
                conf.HM.settings['NoTime'] = self.NM.settings['NoTime']
                # adjust length of time period
                conf.HM.settings['seconds'] *= self.NM.scenarios['Weights'][0]
                # Remove fixed water flows
                conf.HM.settings['In'] = []
                # Define scenarios
                conf.HM.connections['Number'] = NoNM
                # Connect beginning and end of each scenario
                conf.HM.connections['LinksF'] = np.zeros((NoNM, 2), dtype=int)
                conf.HM.connections['LinksT'] = np.ones((NoNM, 2), dtype=int)
                for xh in range(NoNM):
                    conf.HM.connections['LinksF'][xh][0] = xh
                    conf.HM.connections['LinksT'][xh][0] = xh

            # Create hydraulic model
            self.HM = hn(conf.HM)

            # Initialise model
            self.HM.initialise()

            # TODO: Replace LL
            # Create LL
            if self.HM.settings['Flag']:
                # Inputs to pyeneH
                self.p['NoHMin'] = len(self.HM.nodes['In'])
                self.p['pyeneHin'] = np.zeros((self.p['NoHMin'], 2), dtype=int)
                # Adding pump inputs
                for x in range(self.NM.pumps['Number']):
                    xp = self.HM.pumps['To'][x]
                    if xp != 0:
                        xn = self.EM.settings['Vectors']-1

                        while self.HM.nodes['In'][xn] != xp:
                            xn -= 1
                        self.p['pyeneHin'][xn][0:2] = [1, x]

                # Outputs from pyeneH
                # Hydropower units with rivers connected downstream - The info
                # is calculated based on the summation of the flows
                self.p['NoHydDown'] = 0
                self.p['LLHydDown'] = np.zeros(self.NM.hydropower['Number'],
                                               dtype=int)
                for x in range(self.NM.hydropower['Number']):
                    xn = self.HM.hydropower['Node'][x]
                    if self.HM.p['LLN2B2'][xn-1, 2] > 0:
                        self.p['LLHydDown'][self.p['NoHydDown']] = x
                        self.p['NoHydDown'] += 1
                self.p['LLHydDown'] = self.p['LLHydDown'][0:self.p['NoHydDown']]

                # Hydropower units without downstream rivers or other devices that
                # require comparison with water outpouts
                self.p['NoHMout'] = len(self.HM.nodes['Out'])
                self.p['LLHydOut'] = np.zeros((self.p['NoHMout'], 2), dtype=int)
                self.s['LLHydOut'] = np.zeros(self.p['NoHMout'], dtype=int)
                xh = 0
                for x in range(self.NM.hydropower['Number']):
                    xn1 = self.HM.hydropower['Node'][x]-1
                    xn2 = self.HM.connections['NodeSeqOut'][xn1]
                    if xn2 != 0:
                        self.p['LLHydOut'][xn2-1][:] = [1, x]
                        self.s['LLHydOut'][xh] = x
                        xh += 1
                self.s['LLHydOut'] = self.s['LLHydOut'][0:xh]

                # Avoid double counting
                for x in self.p['LLHydDown']:
                    xn = self.HM.hydropower['Node'][x]-1
                    xn = self.HM.connections['NodeSeqOut'][xn]
                    self.p['LLHydOut'][xn-1][:] = [0, 0]
                # Pumps
                self.p['LLHPumpOut'] = np.zeros((self.p['NoHMout'], 2), dtype=int)
                for x in range(self.NM.pumps['Number']):
                    xn1 = self.HM.pumps['From'][x]-1
                    xn2 = self.HM.connections['NodeSeqOut'][xn1]
                    if xn2 != 0:
                        self.p['LLHPumpOut'][xn2-1][:] = [1, x]
        else:
            _, file_extension = os.path.splitext(path)
            
            from .pyene_Models import models
            self.mod_cython_cpp = models()
            if file_extension == ".xlsx":
                from .pyeneI import excel2pyene as e2p
                read = e2p()
                read.read_excel(energy_file=path, model=self.mod_cython_cpp)
            self.mod_cython_cpp.initialise()
            outpput_file_name = "outputs"
            for options in self.mod_cython_cpp.model_options:
                if options.name == "output file name":
                    outpput_file_name = options.value

            self.save_solution = pyeneSave()
            self.save_solution.initialise(model=self.mod_cython_cpp, dir=os.path.dirname(path), name_file=outpput_file_name)

    def NSim(self, conf):
        ''' Network only optimisation '''
        # Get network object
        NM = dn(conf.NM)

        # Initialise
        NM.initialise(rn(conf.RM))

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
        ''' Objective function for energy and networks model'''
        return sum((sum((sum(m.vNGCost[self.NM.get_ConC(xh)+xg, xt]
                             for xg in self.NM.s['Gen']) +
                         sum(m.vNFea[self.NM.get_ConFea(xh)+xf, xt]
                             for xf in self.NM.s['Fea']) *
                         self.Penalty)*self.NM.scenarios['Weights'][xt]
                        for xt in self.NM.s['Tim']) -
                    sum(self.NM.pumps['Value'][xdl] *
                        self.NM.ENetwork.get_Base() *
                        sum(m.vNPump[self.NM.get_ConP(xh)+xdl+1, xt] *
                            self.NM.scenarios['Weights'][xt]
                            for xt in self.NM.s['Tim'])
                        for xdl in self.NM.s['Pump'])) *
                   self.OFaux[xh] for xh in self.NM.get_ConS()) + \
            m.vHpenalty

    def Print_ENSim(self, m):
        ''' Print results '''
        
        from pyensys.engines.pyeneP import PrintClass
        prnt = PrintClass(m, self)
        
        self.EM.print(m)
        for xh in range(self.p['Number']):
            self.NM.print(m, [xh], prnt)
            self.HM.print(m, [xh])
            print()

        print('Water outputs:')
        for xn in self.EM.s['Nodz']:
            for xv in self.EM.s['Vec']:
                aux = m.vEOut[xn, xv].value
                print("%8.4f " % aux, end='')
            print('')
        print('Water inputs:')
        if type(m.vEIn) is np.ndarray:
            print(m.vEIn)
        else:
            for xn in self.EM.s['Nodz']:
                for xv in self.EM.s['Vec']:
                    aux = m.vEIn[xn, xv].value
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

    def run(self, solver=None, subscripts={}):
        ''' Run integrated pyene model '''
        if solver == "pyomo":
            # Build pyomo model
            m = ConcreteModel()
            m = self.build_Mod(m)

            # Run pyomo model
            (m, results) = self.Run_Mod(m)

            return m
        elif solver == "CLP":
            from pyene.engines.pyene_Models import EnergyandNetwork

            ENMod = EnergyandNetwork(self.EN.EM, self.EN.NM, self.EN)
            ENMod.optimisationENM()

            return ENMod
        else:
            from .pyene_Models import models
            if isinstance(self.mod_cython_cpp, models):
                self.mod_cython_cpp.evaluate(subscripts=subscripts)
    
    def save_outputs(self, sim_no=None):
        self.save_solution.save_results(model=self.mod_cython_cpp, sim_no=sim_no)
    
    def update_parameter(self, information):
        return self.mod_cython_cpp.update_parameter(information=information)
    
    def get_moea_variables(self):
        return self.mod_cython_cpp.get_moea_variables()
    
    def get_moea_objectives(self):
        return self.mod_cython_cpp.get_moea_objectives()
    
    def close_output_files(self):
        self.save_solution.close_output_files()

    def Run_Mod(self, m):
        ''' Run pyomo model '''
        # Finalise model
        m = self._AddPyeneCons(m)

        #                          Objective function                         #
        self.OFaux = self._Calculate_OFaux()

        m.OF = Objective(rule=self.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')
        # Print
        results = opt.solve(m)

        return (m, results)

    def set_Demand(self, index, value):
        ''' Set a demand profile '''
        if index <= self.NM.scenarios['Number']:
            # Do the profiles match?
            value = self.CheckProfile(value)

            aux1 = (index-1)*self.NM.settings['NoTime']
            aux2 = index*self.NM.settings['NoTime']

            self.NM.scenarios['Demand'][aux1:aux2] = value

    def set_GenCoFlag(self, index, value):
        ''' Adjust maximum output of generators '''
        if isinstance(value, bool):
            if value:
                # Maximum capacity
                self.NM.Gen.set_Max(index-1,
                                    self.NM.Gen.get_Max(index-1) /
                                    self.NM.ENetwork.get_Base())
            else:
                # Switch off
                self.NM.Gen.set_Max(index-1, 0)
        else:
            # Adjust capacity
            # TODO: Costs should be recalculated for higher capacities
            value /= self.NM.ENetwork.get_Base()
            if value > self.NM.Gen.get_Max(index-1):
                import warnings
                warnings.warn('Increasing generation capacity is not'
                              ' supported yet')

            self.NM.Gen.set_Max(index-1, value)

    def set_Hydro(self, index, value):
        ''' Set MWh of hydro that are available for a single site '''
        if self.NM.hydropower['Number'] == 1:
            self.EM.Weight['In'][1] = value
        else:
            self.EM.Weight['In'][1][index-1] = value

    def set_HydroPrice(self, index, value):
        ''' Set kWh of hydro that are available for a single site '''
        raise NotImplementedError('Water prices not yet enabled')
        # Adjust self.NM.p['GenLCst']

    def _set_LineCapacityAux(self, value, index, *argv):
        ''' Auxiliary for selecting line parameters '''
        if 'BR_R' in argv:
            for xb in index:
                self.NM.ENetwork.Branch[xb].set_R(value)
        elif 'BR_X' in argv:
            for xb in index:
                self.NM.ENetwork.Branch[xb].set_X(value)
        elif 'BR_B' in argv:
            for xb in index:
                self.NM.ENetwork.Branch[xb].set_B(value)
        else:
            value /= self.NM.ENetwork.get_Base()
            for xb in index:
                self.NM.ENetwork.Branch[xb].set_Rate(value)

    def set_LineCapacity(self, index, value, *argv):
        ''' Adjust maximum capacity of a line - pass BR_R for R/X/B'''
        self._set_LineCapacityAux(value, index, *argv)

    def set_LineCapacityAll(self, value, *argv):
        ''' Adjust capacity of all lines - pass BR_R for R/X/B'''
        index = range(self.NM.ENetwork.get_NoBra())
        self._set_LineCapacityAux(value, index, *argv)

    def set_PumpPrice(self, index, value):
        ''' Set value for water pumped '''
        raise NotImplementedError('Pump prices not yet enabled')
        # Adjust self.NM.p['GenLCst']

    def set_RES(self, index, value):
        '''
        Set PV/Wind profile  
        
        index - beginning from 1
        value - time series
        
        if the index is in self.Overwrite['RESX'], the time series in 'RES'
        will overwrite value
        '''

        # Overwrite value?
        if index in self.Overwrite['RESX']:
            value = self.Overwrite['RES'][self.Overwrite['RESL']
                                          [self.Overwrite['RESX'].index(index)
                                           ]]

        if index <= self.NM.scenarios['NoRES']:
            value = self.CheckProfile(value)

            aux1 = (index-1)*self.NM.settings['NoTime']
            aux2 = index*self.NM.settings['NoTime']

            self.NM.scenarios['RES'][aux1:aux2] = value
            xi = 0
            for xs in range(aux1, aux2):
                self.NM.scenarios['RES'][xs] = \
                    (value[xi]/self.NM.ENetwork.get_Base())
                xi += 1

    def SingleLP(self, ENM):
        ''' Get mathematical model '''
        # Define pyomo model
        mod = ConcreteModel()

        #                                 Sets                                #
        mod = ENM.addSets(mod)

        #                              Parameters                             #
        mod = ENM.addPar(mod)

        #                           Model Variables                           #
        mod = ENM.addVars(mod)

        #                             Constraints                             #
        mod = ENM.addCon(mod)

        return mod


    # TODO: Generalise this functions
    def set_element_status(self, ID=None, status=True, typ=None):
        ''' This function sets the status of the element "typ" to on or off '''
        if typ == "branch":
            if ID:
                self.NM.ENetwork.Branch[ID].data['BR_STATUS'] = status
            else:
                raise ValueError("ID -->{}<--- is not valid to set the status of the branch".format(ID))
        if typ == "generator":
            if ID:
                self.NM.Gen.Conv[ID].data['GEN'] = status
            else:
                raise ValueError("ID -->{}<--- is not valid to set the status of the generator".format(ID))
