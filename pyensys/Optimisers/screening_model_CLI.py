# -*- coding: utf-8 -*-

'''
@author: Wangwei Kong & Andrey Churkin

Screening model consider different years and scenarios with contingency 

    Main function:
        Required inputs: 
            Test case: country name, case name, .m file related to the case name
            Load info: multipliers for different year/ scenarios, yearly peak load
            Cost: branch investment cost (default to 100 $/MW), load curtailment penalty (default to 1e4 $/MW)
            Contingency: contingency status (True/False), contingency lists
            Investment catalogue
        
            
        Outputs:
            Invtervension list: branch investments values for all years and scenarios
            
            Yearly investments (printed not stored): branch investment for each year
            
        Note:
            The screening model is run for each year/scenario to find the branch investments.
            Each year (y) will take previous investments from year y-1, then form new investments for year y.
                            
    
'''

from __future__ import (division, print_function)
from pyomo.core import ConcreteModel, Constraint, minimize, NonNegativeReals, \
 Objective, Var, Reals
from pyomo.environ import SolverFactory
from pyomo.core import value as Val
import networkx as nx
import pyomo.environ as pyo
from dataclasses import dataclass
import json
import os
import math
import numpy as np
# from scenarios_multipliers import get_mult
# from input_output_function import  get_peak_data, read_input_data
# from process_data import mult_for_bus
# import cProfile
# import pstats

from os.path import join, dirname

import pandas as pd # to read new EV load data

@dataclass
class network_parameter:
    name                :   str     = None      # Name of the parameter
    position_tree       :   dict    = None      # Position in the energy tree - representative days
                                                # in case of parameters changing in time
    hour                :   int     = None      # Hour of the parameter in case of parameters 
                                                # changing in time
    ID                  :   str     = None      # ID of element
    type                :   str     = None      # Type of element, e.g. bus, branch
    sub_type            :   str     = None      # Sub type of element, e.g. thermal, hydro
    bus                 :   int     = None      # Number of the bus (node) that the element is 
                                                # related
    ends                :   list    = None      # list of ends for branches in format [from, to]
    value               :   float   = None      # Value of specific parameter
 
@dataclass
class nodes_info_network:
    type                :   str     = None      # Type of element, e.g. bus, branch, generator
    sub_type            :   str     = None      # Sub type of element, e.g. thermal, hydro
    ID                  :   str     = None      # ID of element
    node                :   int     = None      # Number of node in graph
    parameters          :   list    = None      # Parameters associated to the node in the graph
    variables           :   list    = None      # Variables associated to the node in the graph
    bus                 :   int     = None      # Number of the bus related to the graph's node
    ends                :   list    = None      # list of ends for branches in format [from, to]

# ####################################################################
# ####################################################################
def model_screening(mpc, gen_status, line_status, cicost, penalty_cost,
                    cont_list, prev_invest, peak_Pd, mult, Pd_additions, NoTime=1):
    ''''read paras and vars from jason file'''
    def readVarPara():
    
        '''Input parameters for generator, bus and branch'''
        
        '''
            Recorded parameters are:
                     auxGen = ['PMAX', 'PMIN', 'QMAX', 'QMIN', 'VG']
                     auxBus = ['BASE_KV', 'PD', 'QD', 'VMAX', 'VMIN']
                     auxBranch = ['BR_B', 'BR_R', 'BR_X', 'RATE_A', 'BR_STATUS']
    
            Recorded variables are:
                     auxGen=['Pout','Qout']
                     auxBus=['Pin','Qin''Pout','Qout']
                     auxBranch=['P','Q','ANG']    
        '''
        
        # Input generator parameters   
        nw_parameters=[]
        auxGen = ['PMAX', 'PMIN', 'QMAX', 'QMIN', 'VG','GEN_BUS']
        
        for NoGen in range(mpc['NoGen']):
            for gen_para_name in auxGen:
                gen_para_temp = network_parameter( 
                            name             = gen_para_name,      
                            position_tree    = None,      
                            hour             = None,      
                            ID               = 'Gen'+str(NoGen),    
                            type             = 'generator',      
                            sub_type         = None,      
                            bus              = mpc['gen']['GEN_BUS'][NoGen],      
                            ends             = None,      
                            value            = mpc['gen'][gen_para_name][NoGen]
                                              )
                # Add generator parameters  
                nw_parameters.append(gen_para_temp)
    
        del auxGen,gen_para_temp, gen_para_name
        
        # Input bus parameters   
        auxBus = ['BASE_KV', 'PD', 'QD', 'VMAX', 'VMIN']
        
        for NoBus in range(mpc['NoBus']):
            for bus_para_name in auxBus:
                bus_para_temp = network_parameter( 
                            name             = bus_para_name,      
                            position_tree    = None,      
                            hour             = None,      
                            ID               = 'Bus'+str(NoBus),  
                            type             = 'bus',      
                            sub_type         = None,      
                            bus              = mpc['bus']['BUS_I'][NoBus],      
                            ends             = None,      
                            value            = mpc['bus'][bus_para_name][NoBus]
                                              )        
                # Add generator parameters  
                nw_parameters.append(bus_para_temp)
        del auxBus, bus_para_temp, bus_para_name
        
        
        # Input branch parameters   
        auxBranch = ['BR_B', 'BR_R', 'BR_X', 'RATE_A', 'BR_STATUS']
        
        for NoBranch in range(mpc['NoBranch']):
            for branch_para_name in auxBranch:
                branch_para_temp = network_parameter( 
                            name             = branch_para_name,      
                            position_tree    = None,      
                            hour             = None,      
                            ID               = 'Branch'+str(NoBranch),
                            type             = 'branch',      
                            sub_type         = None,                            
                            ends             = [mpc['branch']['F_BUS'][NoBranch], mpc['branch']['T_BUS'][NoBranch]],     
                            value            = mpc['branch'][branch_para_name][NoBranch]
                                              )    
                # Add branch parameters  
                nw_parameters.append(branch_para_temp)
        del auxBranch, branch_para_temp, branch_para_name
        
     
        
        return nw_parameters
    
    
    # ####################################################################
    # ####################################################################
        
        
    '''Network model to record var and para in graph and get values from node'''
    class NetworkModel():
        def __init__(self):
            self.readVarPara = readVarPara()
            self.network_parameters = self.readVarPara # nw_parameters
            # self.network_variables = self.readVarPara[1]  # nw_variables
            self._create_graph()
            
            
        def _create_nodes_graph(self):
               nodes_graph = []
               exist = False
               counter = 0
               
               # Creating list of nodes  and adding parameters
               for parameter in self.network_parameters:
                     if nodes_graph:
                         for node_g in nodes_graph:
                             if node_g.ID == parameter.ID:
                                 exist = True
                                 node_g.parameters.append(parameter)
                                 break
                     if not exist:
                         node = nodes_info_network() 
                         node.node = counter
                         node.type = parameter.type
                         node.sub_type = parameter.sub_type
                         node.ID = parameter.ID
                         node.parameters = [parameter]
                         node.bus = parameter.bus
                         node.ends = parameter.ends
                         counter += 1
                         nodes_graph.append(node)
                     exist = False
         
        
               # Adding nodes to graph
               for node_g in nodes_graph:
                   self.network.add_node(node_g.node, obj=node_g)
        
        
        def _create_edges_graph(self):
            # Creating branches of graph
            branches_graph = []
            for node_g in self.network.nodes(data=True):
                if node_g[1]['obj'].type == "generator":
                    for aux in self.network.nodes(data=True):
                        if aux[1]['obj'].type == "bus" and node_g[1]['obj'].bus == aux[1]['obj'].bus:
                            branches_graph.append([aux[1]['obj'].node, node_g[1]['obj'].node])
                            break
                elif node_g[1]['obj'].type == "branch":
                    flag = [False, False]
                    for aux in self.network.nodes(data=True):
                        if aux[1]['obj'].type == "bus" and node_g[1]['obj'].ends[0] == aux[1]['obj'].bus:
                            branches_graph.append([aux[1]['obj'].node, node_g[1]['obj'].node])
                            flag[0] = True
                        elif aux[1]['obj'].type == "bus" and node_g[1]['obj'].ends[1] == aux[1]['obj'].bus:
                            branches_graph.append([node_g[1]['obj'].node, aux[1]['obj'].node])
                            flag[1] = True
                        if flag[0] and flag[1]:
                            break
            self.branches_graph = branches_graph
            for branches in  branches_graph:
                self.network.add_edge(branches[0], branches[1])
        
        
        def _create_graph(self):
            self.network = nx.MultiGraph()
            self._create_nodes_graph()
            self._create_edges_graph()
            
            
        
    
    
        def get_value_network(self, ID=None, name=None, position_tree=None, hour=None, typ=None):
                ''' This function retrieves the values of variables and parameters
        
                    Parameters
                    ----------
                    Mandatory:\\
                    ID              :   Unique ID of the network element\\
                    name            :   Name of variable or parameter to be retrieved\\
                    position_tree   :   Dictionary containing information of the location of the information
                                        in relation with the energy tree. If the value does not vary with
                                        the energy tree then this value should be left in None\\
                    hour            :   integer that indicates the specific hour of the requested data. If
                                        the data does not change in time then this input must be left in
                                        None\\
                    typ             :   This refers to the type of element to be retrieved. This value
                                        can be either "variable" or "parameter". Other values will not
                                        be accepted
                '''
                if not position_tree:
                    for node in self.network.nodes(data=True):
                        if node[1]['obj'].ID == ID and typ == "parameter":
                            for parameter in node[1]['obj'].parameters:
                                if parameter.name == name and (not hour or hour == parameter.hour):
                                    return parameter.value
                        if node[1]['obj'].ID == ID and typ == "variable":
                            for variable in node[1]['obj'].variables:
                                if variable.name == name and (not hour or hour == variable.hour):
                                    return variable.value
                else:
                    number_node = self._get_initial_tree_node(position_tree)
                    return self._calculate_value_tree(number_node, ID, position_tree, hour, typ, False)
        
                return None
     
    
    
    # ####################################################################
    
    
    '''optimization (pyomo) model'''
    
    #  Sets 
    def addSet(m): 
        m.Set={}
        ''' Add pyomo sets '''
        m.Set['Bra'] = range(mpc['NoBranch'])
        m.Set['Bus'] = range(mpc['NoBus'])
        m.Set['Gen'] = range(mpc['NoGen'])
        m.Set['Tim'] = range(NoTime) #range(24)
        #m.zset = range(3) # piece wise generator cost
        m.Set['Cont'] = range(len(cont_list))
    
        return m
    
    
    #   Parameters 
    def addPara(m):
        m.para={}
    
        for node in NetworkModel.network.nodes(data=True):
            for NoPar in range(len(node[1]['obj'].parameters)):  
                m.para [node[1]['obj'].ID + str('_') + node[1]['obj'].parameters[NoPar].name ] \
                    = node[1]['obj'].parameters[NoPar].value
       
        return m
    
        
    
     
    # Var          
    def addVar(m):
      
        # Gen
        m.Pgen = Var(m.Set['Gen'],m.Set['Cont'], m.Set['Tim'], domain=NonNegativeReals, initialize=10)
        m.Cgen = Var(m.Set['Gen'],m.Set['Cont'], m.Set['Tim'], domain=NonNegativeReals, initialize=10)
    
        # Branch
        m.Pbra = Var(m.Set['Bra'], m.Set['Cont'], m.Set['Tim'], domain=Reals, initialize=0) # Branch power flow
        m.ICbra = Var(m.Set['Bra'], m.Set['Tim'], domain=NonNegativeReals, initialize=0) # invest capacity
        
        # Bus angle
        m.Ang = Var(m.Set['Bus'], m.Set['Cont'], m.Set['Tim'], bounds=(-2*math.pi, 2*math.pi), initialize=0) # from 0
        
        # Load curtailment
        m.Plc = Var(m.Set['Bus'], m.Set['Cont'],  m.Set['Tim'], domain=NonNegativeReals, initialize=0)
        
        return m
         
    
    class rules:
    
       # Gen output constraint rules
        def genMax_rule(m, xg,xk, xt):
            if gen_status == True and mpc["gen"]["GEN"][xg] == 0 :
                return m.Pgen[xg,  xk, xt] == 0 
            else:
                gen_bus = m.para["Gen"+str(xg)+"_GEN_BUS"]
                bus_number = [i for i,x in enumerate(mpc["bus"]["BUS_I"]) if x==gen_bus]
                return m.Pgen[xg, xk, xt] <= mult[bus_number[0]] * m.para["Gen"+str(xg)+"_PMAX"]
            
        
        def genMin_rule(m,xg, xk, xt):
            if gen_status == True and mpc["gen"]["GEN"][xg] == 0 :
                return m.Pgen[xg,  xk, xt] == 0 
            else:   
                return m.Pgen[xg,xk, xt] >= m.para["Gen"+str(xg)+"_PMIN"]
        
        
        
        # Branch constraint
        # DC power flow
        def DCPF_rule(m, xbr,xk, xt):
            
            br_X = mpc['branch']['BR_X'][xbr]/ mpc['baseMVA']
            fbus_name = mpc['branch']['F_BUS'][xbr]
            fbus = mpc['bus']['BUS_I'].index(fbus_name)
            tbus_name = mpc['branch']['T_BUS'][xbr]
            tbus = mpc['bus']['BUS_I'].index(tbus_name)
            
            if line_status == True and mpc["branch"]["BR_STATUS"][xbr] == 0:
                temp_line_stat = 0
            else:
                temp_line_stat = 1
            
            if cont_list[xk][xbr] == 0 or temp_line_stat == 0:
                return Constraint.Skip
            else:             
                return  m.Pbra[xbr,xk, xt] == ( m.Ang[fbus,xk, xt] - m.Ang[tbus,xk, xt]) / br_X
        
        
        
        def slackBus_rule(m,xk, xt):
            for i in range(mpc['NoBus']):
                if mpc['bus']['BUS_TYPE'][i] == 3:
                    slc_bus = i
            
            return m.Ang[slc_bus,xk, xt] == 0
        
    
        # # Branch capacity 
        # # TODO: check line status relation with investment
        # def braCapacity_rule(m,xbr,xk,xt):
        #     if m.para["Branch"+str(xbr)+"_RATE_A"] != 0:
        #         return m.Pbra[xbr, xk, xt] <=  cont_list[xk][xbr] * \
        #                                         ( (m.ICbra[xbr, xt] + prev_invest[xbr] + m.para["Branch"+str(xbr)+"_RATE_A"] )  ) 
        #     else:
        #         return  m.Pbra[xbr, xk, xt]  <= cont_list[xk][xbr] * float('inf') #* mpc["branch"]["BR_STATUS"][xbr]
       
        
        # # both flow directions       
        # def braCapacityN_rule(m,xbr,xk, xt):
        #     if m.para["Branch"+str(xbr)+"_RATE_A"] != 0:
        #         return  - m.Pbra[xbr,xk,  xt] <= cont_list[xk][xbr] *\
        #                                         ( (m.ICbra[xbr, xt] + prev_invest[xbr] + m.para["Branch"+str(xbr)+"_RATE_A"] )  )
        #     else:
        #         return  - m.Pbra[xbr,xk,  xt]  <= cont_list[xk][xbr] * float('inf') #* mpc["branch"]["BR_STATUS"][xbr]
        
        
        # Branch capacity 
        def braCapacity_rule(m,xbr,xk,xt):
            if line_status == True and mpc["branch"]["BR_STATUS"][xbr] == 0:
                temp_line_stat = 0
            else:
                temp_line_stat = 1
            
            if cont_list[xk][xbr] == 0 or temp_line_stat == 0:
                return m.Pbra[xbr, xk, xt] == 0 #Constraint.Skip
            else:             
                if m.para["Branch"+str(xbr)+"_RATE_A"] != 0:                  
                    return m.Pbra[xbr, xk, xt] <=   m.ICbra[xbr, xt] + prev_invest[xbr] + m.para["Branch"+str(xbr)+"_RATE_A"] # incude investments in previous years
                    # return m.Pbra[xbr, xk, xt] <=   m.ICbra[xbr, xt] + 0 + m.para["Branch"+str(xbr)+"_RATE_A"] # without investments in previous years
                    
                else:   
                    return  m.Pbra[xbr, xk, xt]  <=  float('inf')
                
                    
        
        # both flow directions       
        def braCapacityN_rule(m,xbr,xk, xt):
            if line_status == True and mpc["branch"]["BR_STATUS"][xbr] == 0:
                temp_line_stat = 0
            else:
                temp_line_stat = 1
            
            
            if cont_list[xk][xbr] == 0 or temp_line_stat == 0:
                return  m.Pbra[xbr, xk, xt] == 0 #Constraint.Skip
            else:             
                if m.para["Branch"+str(xbr)+"_RATE_A"] != 0:
                    # return  - m.Pbra[xbr,xk,  xt] <=  m.ICbra[xbr, xt] + prev_invest[xbr] + m.para["Branch"+str(xbr)+"_RATE_A"] 
                    return  - m.Pbra[xbr,xk,  xt] <=  m.ICbra[xbr, xt] + 0 + m.para["Branch"+str(xbr)+"_RATE_A"] # without prev investments

                
                else:
                    return  - m.Pbra[xbr,xk,  xt]  <=  float('inf') 
            
            
        
        
        
        
        
        # Nodal power balance
        def nodeBalance_rule(m, xb,xk,xt):
    
            return sum( m.Pgen[genCbus[xb][i],xk,xt]  for i in range(len(genCbus[xb])) )  \
                    + sum( m.Pbra[braTbus[xb][i]-noDiff,xk,xt]  for i in range(len(braTbus[xb])) )  \
                    == sum( m.Pbra[braFbus[xb][i]-noDiff,xk,xt]  for i in range(len(braFbus[xb])) ) \
                      + mult[xb] *Pd[xb] - m.Plc[xb,xk,xt] + Pd_additions[xb]
    
        def loadcurtail_rule(m, xb,xk,xt):
            
            return  mult[xb] *abs(Pd[xb]) >= m.Plc[xb,xk,xt]
        
        # # Cost Constraints
        # Piece wise gen cost: Number of piece = 3
        def pwcost_rule(m,xg,xk,xp,xt):
                      
            return m.Cgen[xg,xk,xt] >= m.Pgen[xg,xk,xt] * lcost[xp][xg] + min_y[xp][xg] 
            
            

                
    
        
    
              
    def addConstraints(m):
        
       # Add Gen constraint rules
        m.genMax = Constraint( m.Set['Gen'], m.Set['Cont'], m.Set['Tim'], rule=rules.genMax_rule )
        m.genMin = Constraint( m.Set['Gen'], m.Set['Cont'], m.Set['Tim'], rule=rules.genMin_rule )
       
        # piecve wise gen cost
        m.pwcost = Constraint(m.Set['Gen'],m.Set['Cont'], range(NoPieces),  m.Set['Tim'],rule=rules.pwcost_rule)
        
        # Add branch flow DC OPF
        m.DCPF = Constraint( m.Set['Bra'], m.Set['Cont'], m.Set['Tim'], rule=rules.DCPF_rule ) 
        
        # Set slack bus angle to 0
        m.slackBus = Constraint( m.Set['Cont'],m.Set['Tim'], rule=rules.slackBus_rule ) 
        
        
        # Add branch capacity constraints
        m.braCapacity = Constraint( m.Set['Bra'],m.Set['Cont'], m.Set['Tim'], rule=rules.braCapacity_rule )
        m.braCapacityN = Constraint( m.Set['Bra'],m.Set['Cont'], m.Set['Tim'], rule=rules.braCapacityN_rule )
        
    
        # Add nodal balance constraint rules
        m.nodeBalance = Constraint( m.Set['Bus'],m.Set['Cont'],m.Set['Tim'], rule=rules.nodeBalance_rule ) 
        
        m.loadcurtail = Constraint( m.Set['Bus'],m.Set['Cont'],m.Set['Tim'], rule=rules.loadcurtail_rule)  
        
    
        return m
    
        
    # piece wise gen cost
    def genCost_rule(mpc):
        if mpc['gencost']['MODEL'] != []:
            # Define piece wise cost curve approximation 
            LGcost = np.zeros((3, mpc['NoGen']), dtype=float)
            xval = np.zeros((4,mpc['NoGen']), dtype=float)
            yval = np.zeros((4,mpc['NoGen']), dtype=float)
            lcost = np.zeros((3, mpc['NoGen']), dtype=float)
            
            for NoGen in range(mpc['NoGen']):
                if mpc['gencost']['MODEL'][NoGen] == 1:          # Piece-wise model
                    NoPieces = mpc['gencost']['NCOST'][NoGen]
                    xval = np.zeros(NoPieces, dtype=float)
                    yval = np.zeros(NoPieces, dtype=float)
                    xp = 0
                    for x in range(NoPieces):
                        xval[x] = mpc['gencost']['COST'][xp]
                        yval[x] = mpc['gencost']['COST'][xp+1]
                        xp += 2
                    # Convert to LP constraints 
                    for xv in range(NoPieces):
                        lcost[xv][NoGen] = (yval[xv+1][NoGen]-yval[xv][NoGen]) / (xval[xv+1][NoGen] - xval[xv][NoGen])
         
                
                else:                                                 # Polinomial model
                    # Select number of pieces for the approximation
                   
                    Delta = mpc['gen']['PMAX'][NoGen]
                    
                    if Delta > 0 :
                 
                        Delta /= 3
                   
                        NoPieces = 3#int(np.floor(mpc['gen']['PMAX'][NoGen]/Delta))
    
                        aux = mpc['gen']['PMIN'][NoGen]
                        for xp in range(NoPieces+1):
                            xval[xp][NoGen] = aux
                            xc = mpc['gencost']['NCOST'][NoGen]-1 
                            yval[xp][NoGen] = mpc['gencost']['COST'][NoGen][xc]
                            for x in range(1, mpc['gencost']['NCOST'][NoGen]):
                                xc -= 1
                                yval[xp][NoGen] += mpc['gencost']['COST'][NoGen][xc]*xval[xp][NoGen]**x
                            aux += Delta
        
                        # Convert to LP constraints 
                        for xv in range(NoPieces):
                            lcost[xv][NoGen] = (yval[xv+1][NoGen]-yval[xv][NoGen]) / (xval[xv+1][NoGen] - xval[xv][NoGen])
                    
                   
                        
                        
                # LGcost =  y0-lcost*x0
                for xp in range(NoPieces):
                    LGcost[xp][NoGen] = yval[xp+1][NoGen] - xval[xp+1][NoGen] *  lcost[xp][NoGen]
          
                
                
                        

                            
        else:
            xval = np.random.uniform(low=10, high=50, size=(4,mpc['NoGen']))
            xval[1] = xval[1]*5 
            xval[2] = xval[2]*10
            xval[3] = xval[3]*15
            yval = np.ones((4,mpc['NoGen']), dtype=float)
            lcost = np.random.uniform(low=1, high=10, size=(3,mpc['NoGen']))
            lcost[1] = lcost[1]*3
            lcost[2] = lcost[2]*5
            
                
                    
        return  (NoPieces, lcost, LGcost)


    
    # find all connections
    def nodeConnections_rule():
    
        noDiff = 0
        
        # genCbus[bus][gen_number]
        genCbus = []
        braFbus = []
        braTbus = []
        # only 1 timepoint for Pd
        Pd = []
        for xb in range(mpc["NoBus"]):
            bus_number = mpc["bus"]["BUS_I"][xb]
            # find generator connections
            gen_number = [i for i,x in enumerate(mpc["gen"]["GEN_BUS"]) if x==bus_number]
            genCbus.append(gen_number)
            
            # find branch from this bus
            braF_number = [i for i,x in enumerate(mpc["branch"]["F_BUS"]) if x==bus_number]
            braFbus.append(braF_number)
            # find branch to this bus
            braT_number = [i for i,x in enumerate(mpc["branch"]["T_BUS"]) if x==bus_number]
            braTbus.append(braT_number)
            
            # record demand value
            Pd.append( mpc['bus']['PD'][xb])
    
        if peak_Pd !=[] :
            Pd = peak_Pd
    
            
        
       
        return (noDiff, genCbus, braFbus, braTbus, Pd)
    
    
    
    # Objective function 
    def OFrule(m):
    
        # investment cost: £200/MVA #TODO: update the cost
        return (        # investment cost
                        sum(m.ICbra[xbr,xt] for xbr in m.Set['Bra'] for xt in m.Set['Tim'] ) *cicost +
                        # generation cost
                        sum( m.Cgen[xg,0,xt] for xg in m.Set['Gen'] for xt in m.Set['Tim'] ) +
                        # load curtailment cost
                        sum( m.Plc[xb,xk,xt]  for xb in m.Set['Bus'] for xk in m.Set['Cont'] for xt in m.Set['Tim']) *penalty_cost
                
                )
    
    
    
    
    
    
    ###############################################################################################
    ###############################################################################################   
    ###############################################################################################
    
    
   
    
    # build network model use graph
    NetworkModel = NetworkModel()
    
    
    
    # read mpc file and find info, define gen cost
    
    NoPieces, lcost, min_y = genCost_rule(mpc)
    
    # if not specified, demand value PD is read from mpc file
    noDiff, genCbus, braFbus, braTbus, Pd = nodeConnections_rule()
    
    
    ''' Build a pyomo model '''
    
    # Defining concrete optimisation model
    model = ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    # Adding sets
    model = addSet(model)

    # Adding parameters
    model = addPara(model)

    # Adding variables
    model = addVar(model)

    # Adding constraints
    model = addConstraints(model)

    # Adding objective function
    model.obj = Objective(rule=OFrule, sense=minimize)

    # solve pyomo model
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    print("results.solver.status: ",results.solver.status)


    maxICbra=[]
    interv=[]
    for xb in range(mpc['NoBranch']):
        tempICbra = []
        for xt in range(NoTime):
            tempICbra.append(Val(model.ICbra[xb, xt]))
        maxICbra.append(max(tempICbra))
        if maxICbra[xb] > 0:
            interv.append(maxICbra[xb])

    interv = [math.ceil(x) for x in interv]
    interv.sort()

    interv_vect = Val(model.ICbra[:, 0])

    return interv, maxICbra, interv_vect


def main_screening(mpc, gen_status, line_status, multiplier, cicost,
                   penalty_cost, peak_Pd, ci_catalogue, cont_list):
    ''' Time point '''
    # Number of time points
    NoTime = 1

    # initialise branch investments
    prev_invest = [0]*mpc["NoBranch"]
    interv_list = []
    interv_clust = []  # clusters of investment options

    interv_dict = {k: [] for k in range(mpc["NoBranch"])}
    year_name = [2020, 2030, 2040, 2050]

    for xy in range(len(multiplier)):
        for xsc in range(len(multiplier[xy])):
            mult = multiplier[xy][xsc]

            # use_data_update = True # activate to use new loads from "EV-PV-Storage_Data_for_Simulations.xlsx"
            use_data_update = False # use initial load data (do not include additional EV-PV loads)

            if use_data_update == True:
                if year_name[xy] != 2020:

                    EV_data_path = "C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests" # to be updated
                    EV_data_file_name = 'EV-PV-Storage_Data_for_Simulations.xlsx' # !we need to add this to CLI!
                    EV_data_file_path = os.path.join(EV_data_path, EV_data_file_name)


                    # EV_data_sheet_names = 'PT_Dx_01_' # !we need to add this to CLI!
                    # EV_data_sheet_names = 'HR_Dx_01_' # !we need to add this to CLI!
                    EV_data_sheet_names = 'UK_Dx_01_' # !we need to add this to CLI!

                    EV_load_data = pd.read_excel(EV_data_file_path, sheet_name = EV_data_sheet_names + str(year_name[xy]), skiprows = 1)
                    EV_load_data_MW_profile = EV_load_data["EV load (MW)"]
                    EV_load_data_MW_max = np.max(EV_load_data_MW_profile[0:24])
                    Pd_additions = EV_load_data["Node Ratio"]*EV_load_data_MW_max # how much new load per node
                else:
                    Pd_additions = [0] * mpc['NoBus'] # zero additional EV load
            else:
                Pd_additions = [0] * mpc['NoBus'] # zero additional EV load

            #  take preveious years investment
            temp_interv_list, temp_prev_invest, temp_interv_clust = \
                model_screening(mpc, gen_status, line_status,  cicost,
                                penalty_cost, cont_list, prev_invest, peak_Pd,
                                mult, Pd_additions, NoTime)
            # interv_list.append(temp_interv_list)
            interv_list.extend(temp_interv_list)

            # reduce catalogue in the interv. clustering:
            for xbr in range(mpc["NoBranch"]):
                if temp_interv_clust[xbr] > 0:
                    if mpc["branch"]["TAP"][xbr] == 0:  # line
                        temp_interv_clust[xbr] = \
                            min([i for i in ci_catalogue[0]
                                 if i >= temp_interv_clust[xbr]])
                    else:  # transformer
                        temp_interv_clust[xbr] = \
                            min([i for i in ci_catalogue[1]
                                 if i >= temp_interv_clust[xbr]])

            interv_clust.append(temp_interv_clust)

            # record intervention lists for each branch
            for xbr in range(mpc["NoBranch"]):
                interv_dict[xbr].append(temp_prev_invest[xbr])
                interv_dict[xbr].sort()

        prev_invest = [a+b for a, b in zip(temp_prev_invest, prev_invest)]

    # Save screening results separately:
    file_name = "screen_result_interv_dict"
    with open(join(dirname(__file__), "..", "tests\\outputs\\")+file_name+".json", 'w') as fp:
        json.dump(interv_dict, fp)

    return interv_dict, interv_clust
