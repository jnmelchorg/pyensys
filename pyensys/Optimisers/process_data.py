# -*- coding: utf-8 -*-
"""
@author: Wangwei Kong

Data process
"""
from __future__ import (division, print_function)
from pyomo.core import ConcreteModel, Constraint, minimize, NonNegativeReals, \
 Objective, Var, RangeSet, Binary, Set, Reals
from pyomo.core import value as Val


def record_bra_from_pyo_result(model,mpc,NoSce, pyo_var, year_peak):
    
    record_pyo_var = []
    # if yearly peak value only:
    if year_peak == True: 
        
      for xy in model.Set["Year"]:
            record_pyo_var.append([])  
            for xsc in range(NoSce**xy): 
                record_pyo_var[xy].append([])
                for xbr in range(mpc['NoBranch']):
                    record_pyo_var[xy][xsc].append(Val(pyo_var[xbr,xy,xsc,0,0,0]))
    else:
        
        for xy in model.Set["Year"]:
            record_pyo_var.append([])
            for xsc in range(NoSce**xy):
                record_pyo_var[xy].append([])               
                for xse in model.Set['Sea']:
                    record_pyo_var[xy][xsc].append([])
                    for xd in  model.Set['Day']:
                        record_pyo_var[xy][xsc][xse].append([])
                        for xt in  model.Set['Tim']:
                            record_pyo_var[xy][xsc][xse][xd].append([])
                            for xbr in range(mpc['NoBranch']):
                                record_pyo_var[xy][xsc][xse][xd][xt].append(Val(pyo_var[xbr,xy,xsc,xse,xd,xt]))
                
    return record_pyo_var

def record_bus_from_pyo_result(model,mpc,NoSce, pyo_var, year_peak):
    
    record_pyo_var = []
    # if yearly peak value only:
    if year_peak == True: 
        
      for xy in model.Set["Year"]:
            record_pyo_var.append([])  
            for xsc in range(NoSce**xy): 
                record_pyo_var[xy].append([])
                for xbr in range(mpc['NoBus']):
                    record_pyo_var[xy][xsc].append(Val(pyo_var[xbr,xy,xsc,0,0,0]))
    else:
        
        for xy in model.Set["Year"]:
            record_pyo_var.append([])
            for xsc in range(NoSce**xy):
                record_pyo_var[xy].append([])               
                for xse in model.Set['Sea']:
                    record_pyo_var[xy][xsc].append([])
                    for xd in  model.Set['Day']:
                        record_pyo_var[xy][xsc][xse].append([])
                        for xt in  model.Set['Tim']:
                            record_pyo_var[xy][xsc][xse][xd].append([])
                            for xbr in range(mpc['NoBus']):
                                record_pyo_var[xy][xsc][xse][xd][xt].append(Val(pyo_var[xbr,xy,xsc,xse,xd,xt]))
                
    return record_pyo_var

def record_invest_from_pyo_result(model,mpc,NoSce, ci_var,S_ci):
    
    record_pyo_var = []

        
    for xy in model.Set["Year"]:
          record_pyo_var.append([])  
          for xsc in range(NoSce**xy): 
              record_pyo_var[xy].append([])
              for xbr in range(mpc['NoBranch']):
                  if S_ci[str(xbr)] != []:
                      record_pyo_var[xy][xsc].append(Val(sum(S_ci[str(xbr)][xint] * ci_var[xbr,xint,xy,xsc] for xint in model.Set["Intev"][xbr])))
                  else:
                      record_pyo_var[xy][xsc].append(0)
    
                
    return record_pyo_var


def record_investCost_from_pyo_result(model,mpc,NoSce, ci_var):
    
    record_pyo_var = []

        
    for xy in model.Set["Year"]:
          record_pyo_var.append([])  
          for xsc in range(NoSce**xy):
              record_pyo_var[xy].append(Val(ci_var[xy,xsc] ))
    
                
    return record_pyo_var


def initial_value(mpc,NoYear,NoSce, input_val):
    
    record_var = []

        
    for xy in range(NoYear):
          record_var.append([])  
          for xsc in range(NoSce**xy): 
              record_var[xy].append([])
              for xbr in range(mpc['NoBranch']):
                  record_var[xy][xsc].append(Val(input_val))

                
    return record_var

def recordValues(mpc):
    # record original branch capacity        
    bra_cap = []
    for xbr in range(mpc['NoBranch']):
        bra_cap.append( mpc["branch"]["RATE_A"][xbr] )
    
    
    gen_cost = []
    for xgc in range(mpc["NoGen"]):
        gen_cost.append(mpc["gencost"]["COST"][xgc][0] )
        
    return (bra_cap, gen_cost)


def replaceGenCost(mpc, gen_cost, action):
    # action  = 0, remove gen cost
    # action  = else, recover gen cost
    
    if action == 0:
        # remove gen cost in mpc
        for xgc in range(mpc["NoGen"]):
            mpc["gencost"]["COST"][xgc][0] = 0.1#*(xgc + 1)
    else:
        # recover gen cost
        for xgc in range(mpc["NoGen"]):
            mpc["gencost"]["COST"][xgc][0] = gen_cost[xgc]
    
    return mpc


def mult_for_bus(busMult_input, multiplier, mpc):
    
    if busMult_input == []: 
        mult_bus = []
        for xy in range(len(multiplier)):
            mult_bus.append([])
            for xsc in range(len(multiplier[xy])):
                mult_bus[xy].append([])
                
                temp_mult =  [multiplier[xy][xsc]]*mpc["NoBus"]
                mult_bus[xy][xsc]= temp_mult
    
    else:
        mult_bus = busMult_input.copy()
        
    return mult_bus



def get_factors(d, NoYear):
    DF = [0]*NoYear
    for y in range(NoYear):
        DF[y] = 1/ ((1-d)**y)
    
    # capaital recovery factor
    CRF = [1] * NoYear
    xy = 1
    while xy < NoYear:
        N_year = xy *10
        CRF[xy] = (d * ((1+d)**N_year) ) / ( (1+d)**N_year -1) # d = 0.035 # discount rate <= 30 years: 3.5%
        xy += 1

    return DF, CRF