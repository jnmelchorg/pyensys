"""
Created on Mon April 06 2020

This python file containts the classes and methods for stochastic analysis of
any power system using one of the following methods:

Monte Carlo simulation

@author: Dr. Jose Nicolas Melchor Gutierrez
"""

import math
import random
import time
import numpy as np
from pyene.engines.pyene import pyeneClass as pe
from .pyene_Models import Networkmodel as NMod # Energy model in glpk

# TODO: inherit pyene parameters
class MonteCarloSimulation():
    ''' This class contains all methods to perform the Monte Carlo simulation 
    of any power system '''

    def __init__(self, **kwargs):
        ''' This class method initialise '''
        if 'totaltimesteps' in kwargs:
            self.totaltimesteps = kwargs.pop('totaltimesteps')
        else:
            self.totaltimesteps = 0
        
        if 'iterationsMC' in kwargs:
            self.iterationsMC = kwargs.pop('iterationsMC')
        else:
            self.iterationsMC = 0

    def calculaterandomfailureandrecover(self, ob=None):
        ''' This class method calculates the time to failure and time to 
        recover of the different elements in the system '''
        assert isinstance(ob, pe), "The object that has been passed is not an \
            instance of pyene"
        TTF_gen = [None] * ob.NM.Gen.get_NoCon() # Time to failure generators
        TTR_gen = [None] * ob.NM.Gen.get_NoCon() # Time to recover generators
        TTF_lines = [None] * ob.NM.ENetwork.get_NoBra() # Time to failure lines
        TTR_lines = [None] * ob.NM.ENetwork.get_NoBra() # Time to recover lines
        
        for xg in range(ob.NM.Gen.get_NoCon()):
            TTF_gen[xg] = math.floor(-ob.NM.Gen.Conv[xg].data['MTTF'] * \
                math.log(random.random()))
            TTR_gen[xg] = math.floor(-ob.NM.Gen.Conv[xg].data['MTTR'] * \
                math.log(random.random()))
        
        for xl in range(ob.NM.ENetwork.get_NoBra()):
            TTF_lines[xl] = math.floor(-ob.NM.ENetwork.Branch[xl].data['MTTF'] \
                * math.log(random.random()))
            TTR_lines[xl] = math.floor(-ob.NM.ENetwork.Branch[xl].data['MTTR'] \
                * math.log(random.random()))
        
        return TTF_gen, TTR_gen, TTF_lines, TTR_lines

    def calculatematrixfailureandrecover(self, ob=None, **kwargs):
        ''' This class method creates the on/off matrix of different elements 
        per timestep for the Monte Carlo simulation '''
        Status_Generators = np.zeros( (ob.NM.Gen.get_NoCon(), \
            self.totaltimesteps) )
        Status_Lines = np.zeros( (ob.NM.ENetwork.get_NoBra(), \
            self.totaltimesteps) )

        TTF_gen = kwargs.pop('TTF_gen')
        TTR_gen = kwargs.pop('TTR_gen')
        TTF_lines = kwargs.pop('TTF_lines')
        TTR_lines = kwargs.pop('TTR_lines')

        for xl in range(ob.NM.ENetwork.get_NoBra()):
            count_l = 0
            while count_l < self.totaltimesteps:
                if TTF_lines[xl] == 0:
                    Status_Lines[xl,:] = 1
                    count_l = self.totaltimesteps
                else:
                    for _ in range(TTF_lines[xl]):
                        if (count_l<self.totaltimesteps):
                            Status_Lines[xl,count_l] = 1
                            count_l = count_l + 1

                if TTR_lines[xl] == 0:
                    Status_Lines[xl,:] = 1
                    count_l = self.totaltimesteps
                else:
                    for _ in range(TTR_lines[xl]):
                        if (count_l<self.totaltimesteps):
                            Status_Lines[xl,count_l] = 0
                            count_l = count_l + 1
        
        for xg in range(ob.NM.Gen.get_NoCon()):
            count_g = 0
            while count_g < self.totaltimesteps:
                if TTF_gen[xg] == 0:
                    Status_Generators[xg,:] = 1
                    count_g = self.totaltimesteps
                else:
                    for _ in range(TTF_gen[xg]):
                        if (count_g<self.totaltimesteps):
                            Status_Generators[xg,count_g] = 1
                            count_g = count_g + 1

                if TTR_gen[xg] == 0:
                    Status_Generators[xg,:] = 1
                    count_g = self.totaltimesteps
                else:
                    for _ in range(TTR_gen[xg]):
                        if (count_g<self.totaltimesteps):
                            Status_Generators[xg,count_g] = 0
                            count_g = count_g + 1
        
        sum_lines = Status_Lines.sum(axis=0)
        sum_generators = Status_Generators.sum(axis=0)
        Intact_Hour = np.zeros( self.totaltimesteps )
        for xt in range(self.totaltimesteps):
            if sum_lines[xt] == ob.NM.ENetwork.get_NoBra() and \
                sum_generators[xt] == ob.NM.Gen.get_NoCon():
                Intact_Hour[xt] = 1
                
        return Status_Lines, Status_Generators, Intact_Hour

    def runmontecarlo(self, ob=None, **kwargs):
        ''' This class method runs the monte carlo simulation '''
        assert isinstance(ob, pe), "The object that has been passed is not an \
            instance of pyene"

        demand_profiles = kwargs.pop('dem_profiles')
        # Running the power system without failure
        OF_timesteps_intact = np.zeros(self.totaltimesteps)
        LC_timesteps_intact = np.zeros(self.totaltimesteps)
        ConvC_timesteps_intact = np.zeros(self.totaltimesteps)
        Model = NMod(ob.NM)

        start = time.time()

        for xstep in range(self.totaltimesteps):
            for xn in range(Model.NumberNodesPS):
                if Model.NumberDemScenarios == 0:
                    Model.MultScenariosDemand[0,xn] = demand_profiles[xstep]
                else:
                    Model.MultScenariosDemand[0, 0, xn] = demand_profiles[xstep]
            Model.optimisationNM()
            if not Model.FlagProblem and Model.FlagFeasibility:
                LoadCurtailment = Model.GetLoadCurtailmentSystemED()
            elif Model.FlagProblem and Model.FlagFeasibility:
                LoadCurtailment = Model.GetLoadCurtailmentNodes()
            ThermalGenerationCurtailment = \
                Model.GetThermalGenerationCurtailmentNodes()
            for xh in Model.LongTemporalConnections:
                for xn in range(Model.NumberNodesPS):
                    for xco in range(Model.NumberContingencies + 1):
                        for xt in range(Model.ShortTemporalConnections):
                            if Model.FlagFeasibility:
                                LC_timesteps_intact[xstep] = \
                                    LC_timesteps_intact[xstep] + \
                                    LoadCurtailment[xh, xt, xco, xn]
                for xn in range(Model.NumberConvGen):
                    for xco in range(Model.NumberContingencies + 1):
                        for xt in range(Model.ShortTemporalConnections):
                            if Model.FlagFeasibility and \
                                Model.NumberConvGen > 0:
                                ConvC_timesteps_intact[xstep] = \
                                    ConvC_timesteps_intact[xstep] + \
                                    ThermalGenerationCurtailment\
                                    [xh, xt, xco, xn]
            OF_timesteps_intact[xstep] = Model.GetObjectiveFunctionNM()
            # print(OF_timesteps_intact[xstep])
            # print(LC_timesteps_intact[xstep])
            # print(ConvC_timesteps_intact[xstep])
            # print(xstep)
            # print()

        # Running the power system with failures
        OF_timesteps_MC = np.zeros(self.totaltimesteps * self.iterationsMC)
        LC_timesteps_MC = np.zeros(self.totaltimesteps * self.iterationsMC)
        ConvC_timesteps_MC = np.zeros(self.totaltimesteps * self.iterationsMC)

        # start = time.time()
        for xit in range(self.iterationsMC):
            # Calculate failure and recover times
            failure_recover_elements = {
                'TTF_gen' : None,
                'TTR_gen' : None,
                'TTF_lines' : None,
                'TTR_lines' : None
            }
            failure_recover_elements['TTF_gen'], \
            failure_recover_elements['TTR_gen'], \
            failure_recover_elements['TTF_lines'], \
            failure_recover_elements['TTR_lines'] = \
                self.calculaterandomfailureandrecover(ob)
            
            status_elements = {
                'Status_Lines' : None,
                'Status_Generators' : None,
                'Intact_Hour' : None
            }
    
            # Determine the status of elements
            status_elements['Status_Lines'], \
            status_elements['Status_Generators'], \
            status_elements['Intact_Hour'] = \
                self.calculatematrixfailureandrecover(ob=ob, \
                **failure_recover_elements)
            for xstep in range(self.totaltimesteps):
                for xn in range(Model.NumberNodesPS):
                    if Model.NumberDemScenarios == 0:
                        Model.MultScenariosDemand[0,xn] = demand_profiles[xstep]
                    else:
                        Model.MultScenariosDemand[0, 0, xn] = demand_profiles[xstep]
                
                if status_elements['Intact_Hour'][xstep] == 0:
                    for xconv in range(Model.NumberConvGen):
                        if status_elements['Status_Generators'][xconv, xstep] \
                            == 0:
                            Model.ActiveConv[xconv] = False
                        else:
                            Model.ActiveConv[xconv] = True
                    for xl in range(Model.NumberLinesPS):
                        if status_elements['Status_Lines'][xl, xstep] == 0:
                            Model.ActiveBranches[0, xl] = False
                        else:
                            Model.ActiveBranches[0, xl] = True
                    Model.optimisationNM()
                    if not Model.FlagProblem and Model.FlagFeasibility:
                        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
                    elif Model.FlagProblem and Model.FlagFeasibility:
                        LoadCurtailment = Model.GetLoadCurtailmentNodes()
                    ThermalGenerationCurtailment = \
                        Model.GetThermalGenerationCurtailmentNodes()
                    for xh in Model.LongTemporalConnections:
                        for xn in range(Model.NumberNodesPS):
                            for xco in range(Model.NumberContingencies + 1):
                                for xt in range(Model.ShortTemporalConnections):
                                    if Model.FlagFeasibility:
                                        LC_timesteps_MC[\
                                        xit*self.totaltimesteps + xstep] = \
                                            LC_timesteps_MC[\
                                            xit*self.totaltimesteps + xstep] + \
                                            LoadCurtailment[xh, xt, xco, xn]
                        for xn in range(Model.NumberConvGen):
                            for xco in range(Model.NumberContingencies + 1):
                                for xt in range(Model.ShortTemporalConnections):
                                    if Model.FlagFeasibility and \
                                        Model.NumberConvGen > 0:
                                        ConvC_timesteps_MC[\
                                        xit*self.totaltimesteps + xstep] = \
                                            ConvC_timesteps_MC[\
                                            xit*self.totaltimesteps + xstep] + \
                                            ThermalGenerationCurtailment\
                                            [xh, xt, xco, xn]
                    OF_timesteps_MC[xit*self.totaltimesteps + xstep] = \
                        Model.GetObjectiveFunctionNM()
                else:
                    LC_timesteps_MC[xit*self.totaltimesteps + xstep] =\
                        LC_timesteps_intact[xstep]
                    ConvC_timesteps_MC[xit*self.totaltimesteps + xstep] =\
                        ConvC_timesteps_intact[xstep]
                    OF_timesteps_MC[xit*self.totaltimesteps + xstep] =\
                        OF_timesteps_intact[xstep]
                    # print(OF_timesteps_MC[xit*self.totaltimesteps + xstep])
                    # print(LC_timesteps_MC[xit*self.totaltimesteps + xstep])
                    # print(ConvC_timesteps_MC[xit*self.totaltimesteps + \
                    #     xstep])
                    # print(xit*self.totaltimesteps + xstep)
                    # print()

        end = time.time()
        time1 = end - start
        print('Time Monte Carlo:')
        print('%.15f' %(time1))
        print('Mean Time Monte Carlo per iteration:')
        print('%.15f' %(time1/self.totaltimesteps))
        print('\n\n')
        
        np.set_printoptions(threshold=np.inf)
        print(LC_timesteps_MC)


        # from .pyeneO import PrintinScreen

        # PiS = PrintinScreen(ob)
        # PiS.PrintallResults(Model)


