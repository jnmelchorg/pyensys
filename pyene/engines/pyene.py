# Python Energy & Network Engine
"""
Created on Thu Mar 29 14:04:58 2018

@author: mchihem2
"""
# convert in or long into float before performing divissions
from __future__ import division

# Make pyomo symbols known to python
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory

import numpy as np
from .pyeneN import ENetworkClass as dn  # Network component
from .pyeneE import EnergyClass as de  # Energy balance/aggregation component
import json
import os
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
        return sum(sum(sum(m.vGCost[self.hGC[xh]+xg, xt] for xg in m.sGen) +
                       sum(m.vFea[self.hFea[xh]+xf, xt] for xf in m.sFea) *
                       self.Penalty for xt in m.sTim) * self.OFaux[xh] -
                   sum(m.ValDL[xdl]*sum(m.vDL[self.hDL[xh]+xdl+1, xt]
                                        for xt in m.sTim) for xdl in m.sDL) *
                   self.OFaux[xh]
                   for xh in self.h)

    # Water consumption depends on water use by the electricity system
    def EMNM_rule(self, m, xL, xv):
        return (m.WOutFull[m.LLENM[xL][0], xv] ==
                self.NM.networkE.graph['baseMVA'] *
                sum(m.vGen[m.LLENM[xL][1]+xv, xt] for xt in m.sTim))

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
            self.EM.Weight['In'][index-1] = value
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
        ''' Set PW/Wind profile '''
        aux1 = (index-1)*self.NM.settings['NoTime']
        aux2 = index*self.NM.settings['NoTime']

        self.NM.scenarios['RES'][aux1:aux2] = value

    def set_Demand(self, index, value):
        ''' Set a demand profile '''
        aux1 = (index-1)*self.NM.settings['NoTime']
        aux2 = index*self.NM.settings['NoTime']

        self.NM.scenarios['Demand'][aux1:aux2] = value

    def get_Hydro(self, mod, index):
        ''' Get surplus kWh from specific site '''
        HydroValue = mod.WOutFull[1, index-1].value

        return HydroValue

    def get_HydroMarginal(self, mod, index):
        ''' Get marginal costs for specific hydropower plant '''
        cobject = getattr(mod, 'SoCBalance')
        aux = mod.dual.get(cobject[1, index-1])
        if aux is None:
            HydroValue = 0
        else:
            HydroValue = -1*int(mod.dual.get(cobject[1, index-1]))

        return HydroValue

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

    def get_Pump(self, mod, index):
        ''' Get kWh consumed by a specific pump '''
        PumpValue = 0
        for xh in mod.sCon:
            acu = 0
            for xt in mod.sTim:
                acu += mod.vDL[self.hDL[xh]+index, xt].value
            PumpValue += acu*self.OFaux[xh]
        PumpValue *= self.NM.networkE.graph['baseMVA']

        return PumpValue

    def get_DemandCurtailment(self, mod, bus):
        '''Get the kWh that had to be curtailed from a given bus'''
        DemandValue = 0
        if self.NM.settings['Feasibility']:
            for xh in mod.sCon:
                aux2 = self.hFea[xh]+bus
                acu = 0
                for xt in mod.sTim:
                    acu += mod.vFea[aux2, xt].value
                DemandValue += acu*self.OFaux[xh]
        DemandValue *= self.NM.networkE.graph['baseMVA']

        return DemandValue

    # COllect demand curtailed in all buses
    def get_AllDemandCurtailment(self, mod):
        '''Get the kWh that had to be curtailed from all buses'''
        DemandValue = 0
        if self.NM.settings['Feasibility']:
            for xh in mod.sCon:
                acu = 0
                for xn in range(self.hFea[xh], self.hFea[xh]+self.NM.NoFea):
                    for xt in mod.sTim:
                        acu += mod.vFea[xn, xt].value
                DemandValue += acu*self.OFaux[xh]
        DemandValue *= self.NM.networkE.graph['baseMVA']

        return DemandValue

    # Collect curtailment of RES
    def get_RES(self, mod, RESNode):
        xg = RESNode.index-1
        RESValue = 0
        aux = self.EM.tree['Time'][self.EM.size['Periods']][0]
        for xh in mod.sDL:
            acu = 0
            for xt in mod.sTim:
                acu += (self.NM.RES['Max'][xg] *
                        self.NM.scenarios['RES']
                        [self.NM.resScenario[xg][xh][1]+xt] -
                        mod.vGen[self.NM.resScenario[xg][xh][0], xt].value)
            RESValue += acu*self.EM.Weight['Node'][aux+xh]

        return RESValue

    # Run integrated pyene model
    def run(self):
        # Build pyomo model
        mod = self.build_Mod(self.EM, self.NM)

        # Run pyomo model
        (mod, results) = self.Run_Mod(mod, self.EM, self.NM)

        return mod

    # Build pyomo model
    def build_Mod(self, EM, NM):
        # Declare pyomo model
        mod = ConcreteModel()
        mod = self.SingleLP(EM)

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
        #                      Model additional variables                     #
        mod = self.getVars(mod)

        #                             Constraints                             #
        mod = EM.addCon(mod)
        mod = NM.addCon(mod)
        mod = self.addCon(mod)

        #                          Objective function                         #
        WghtAgg = 0+EM.Weight['Node']
        self.OFaux = np.ones(len(NM.connections['set']), dtype=float)
        xp = 0
        for xn in range(EM.LL['NosBal']+1):
            aux = EM.tree['After'][xn][0]
            if aux != 0:
                for xb in range(aux, EM.tree['After'][xn][1]+1):
                    WghtAgg[xb] *= WghtAgg[xn]
            else:
                self.OFaux[xp] = WghtAgg[xn]
                xp += 1

        mod.OF = Objective(rule=self.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')
        # Print
        results = opt.solve(mod)

        return (mod, results)

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
        print('Water inputs:\n', mod.WInFull)

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
            if (self.NM.networkE.node[xn]['BUS_TYPE'] == 2 or
                self.NM.networkE.node[xn]['BUS_TYPE'] == 3):
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
