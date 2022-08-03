# -*- coding: utf-8 -*-
"""
@author: Wangwei Kong

Input and output functions

    Required inputs:
        contingency list
        country name
        test_case name
        time series data
        investment catalogue
        investment unit costs
        
        
    Outputs:
        mpc in json
        load data
        flex data
"""

# from pyexcel_ods import get_data
# from pyexcel_ods import save_data
import json
import os
import pandas as pd
from conversion_model_mat2json import any2json
from scenarios_multipliers import get_mult
import numpy as np


def json_directory():
    ''' Directory contain JSON files for pytest '''
    return os.path.join(os.path.dirname(__file__), 'tests', 'json')




def read_input_data(ods_file_name, country = "HR", test_case = "HR_2020_Location_1" ):
    cont_list = []
    file_name  = test_case 
    
    # todo: update json input to exclude dummy generators?
    '''load m file'''
    converter = any2json()
    converter.matpower2json(folder_path=json_directory(), \
                            name_matpower=file_name, name_json=file_name)
    print('m file converted to json')

    
    filepath = os.path.join(json_directory(), file_name+'.json')

    ''' Load json file''' 
    # load json file from file directory
    mpc = json.load(open(filepath))
    print('load json file')                             
    
    
    '''Load multipliers for different scnearios'''
    multiplier = get_mult(country) # default to HR
    
    # ''' Load xlsx file'''
    # base_time_series_data = get_data("Transmission_Network_PT_2020_24hGenerationLoadData.ods")
    # base_time_series_data  = pd.read_excel('tests/excel/'+ xlsx_file_name + ".xlsx", sheet_name=None)
    # print('load xlsx file')
   
    # ''' Load ods for contingencies file''' 
    # # ods_file_name = "case_template_CR_L3"
    # ods_file = 'SCOPF_R5/input_data/'+ ods_file_name + ".ods"
    
    # if os.path.exists(ods_file):
    #     cont_ods = pd.read_excel(ods_file,sheet_name = "contingencies")
    
    #     NoCon =  len(cont_ods)
    #     con_bra = []
        
    #     for xc in range(NoCon):
            
    #         fbus = cont_ods["From"][xc]
    #         tbus = cont_ods["To"][xc]
            
    #         # find all branch fron the bus
    #         con_bra_fbus = [index for (index, item) in enumerate(mpc["branch"]["F_BUS"]) if item == fbus]
    #         con_bra_tbus = [index for (index, item) in enumerate(mpc["branch"]["T_BUS"]) if item == tbus]
    #         con_bra.append( list(set(con_bra_fbus).intersection(set(con_bra_tbus)))[0] )
        
        
    #     # create contingecy list
    #     cont_list = [[1]*mpc["NoBranch"]]     
        
    #     for xc in range(NoCon):
            
    #         temp_list = [[1]*mpc["NoBranch"]]  
            
    #         temp_list[0][con_bra[xc]] = 0
            
    #         cont_list.extend(temp_list)
        
        
    # else:
    #     print("input data for contiengcy not found. Use N-1 for simulation")
    #     # generate N-1 contingencies
    #     if cont_list==[]:
    #         cont_list = [[1]*mpc["NoBranch"]] 

    #         temp_list = (cont_list[0]-np.diag(cont_list[0]) ).tolist()
            
    #         # # reduce the size of contingency list
    #         # temp_list = temp_list[:len(temp_list)//29]

    #         cont_list.extend(temp_list)
    
    
    
    ''' Load intervention infor''' 
    
    if os.path.exists('tests/json/intervention.json'):
        file = open('tests/json/intervention.json')
        intv = json.load(file)
        file.close()
        
        print("Reading intervention lists and costs data")
        ci_catalogue = []
        ci_catalogue.append( intv["line_list"] )
        ci_catalogue.append( intv["transformer_list"])
        
        ci_cost = []
        ci_cost.append( intv["line_cost"] )
        ci_cost.append( intv["transformer_cost"] )
        
        # TODO: update intervention inputs for different countries?
        # check input data
        if len(ci_catalogue[0]) != len(ci_cost[0]):
            print("Sizes of input line investment data don't match, default values are used")
            
            ci_catalogue[0] = [10,50,100,200,500,800]
            ci_cost[0] = [20 * i for i in ci_catalogue[0]]
        
        if len(ci_catalogue[1]) != len(ci_cost[1]):
            print("Sizes of input transformer investment data don't match, default values are used")
            
            ci_catalogue[1] = [560,880,1200,2400,5600]
            ci_cost[1] = [20 * i for i in ci_catalogue[1]]
        
        
    else:
        print("Using default intervention lists and costs")

        # ci_catalogue = [10,50,100,200,500,800,1000,2000,5000]
        # ci_cost = [5 * i for i in ci_catalogue]
        ci_catalogue = []
        ci_cost = []
        # lines
        # ci_catalogue.append([10,50,100,200,500,800]) # MVA
        ci_catalogue.append([0.003,0.006,0.009,0.015,0.03,0.045,0.075,0.1125,0.15,0.225,0.3,0.5,0.75,1.0,2.0,5.0,10.0,20.0,30.0,40.0,50.0,60.0,80.0,100.0,250.0,500.0]) # more realistic data for distribution networks
        ci_cost.append( [20 * i for i in ci_catalogue[0]])
        # transformers
        # ci_catalogue.append([560,880,1200,2400,5600])
        ci_catalogue.append([0.003,0.006,0.009,0.015,0.03,0.045,0.075,0.1125,0.15,0.225,0.3,0.5,0.75,1.0,2.0,5.0,10.0,20.0,30.0,40.0,50.0,60.0,80.0,100.0,250.0,500.0]) # more realistic data for distribution networks
        ci_cost.append( [20 * i for i in ci_catalogue[1]])
        
        

    return (mpc, multiplier, 0, cont_list,ci_catalogue,ci_cost)




def read_screenModel_output(country, mpc,test_case, ci_catalogue,intv_cost):
    # reading outputs from the screening model of the reduced intervention list
    file_name = "screen_result_" + country + "_" + test_case
    
    
    if os.path.exists('results/'+ file_name + '.json'):
        
               
        S_ci = json.load(open(os.path.join(os.path.dirname(__file__), 
                                          'results', file_name +'.json')))
    else:
        print("screen results not found. Using predefined intervetion lists, this will cause longer computing time. ")
        S_ci = ci_catalogue[0]
        # expand catalogue for each branch
        S_ci  = {str(k): ci_catalogue[0] for k in range(mpc["NoBranch"])}
        
        for xbr in range(mpc["NoBranch"]):
            if mpc["branch"]["TAP"][xbr] != 0:  # transformer
                S_ci[str(xbr)] = ci_catalogue[1]
        
        
        


    # if not specified, using a linear cost
    ci_cost = {k: [] for k in range(mpc["NoBranch"])}
    for xbr in range(mpc["NoBranch"]):
        if S_ci[str(xbr)] != []:
            # ci_cost[xbr] = [5*i for i in S_ci[str(xbr)]]  # £/MW
            
            # ci_cost[xbr] = [intv_cost[i]  for i in S_ci[str(xbr)]]
            for xci in range(len(S_ci[str(xbr)])):
                
                if mpc["branch"]["TAP"][xbr] == 0:  # line
                    temp = [i for i,x in enumerate(ci_catalogue[0]) if x==S_ci[str(xbr)][xci]]
                
                    ci_cost[xbr].append(intv_cost[0][temp[0]])
                    
                else: # transformer
                    temp = [i for i,x in enumerate(ci_catalogue[1]) if x==S_ci[str(xbr)][xci]]
                
                    ci_cost[xbr].append(intv_cost[1][temp[0]])
                    


    return S_ci, ci_cost


                     
def output_data2Json(NoPath, NoYear, path_sce, sum_CO, yearly_CO, ci, sum_ciCost, Cflex, Pflex, outputAll=False,country = "HR", test_case = "HR_2020_Location_1" , pt = "_pt1"):

    output_data = {}
    # sce_data = {}
    year_num = [2020, 2030, 2040, 2050]
    output_data = { "Country": country, 
                    "Case name": test_case}
    
    if sum_CO == 0: # part 1 output without operation cost
    
        # output all the pathways (scenarios)
        if outputAll == True:      
            for xp in range(NoPath):
                sce_data = {}
                sce_data["Total investment cost (EUR)"] = sum_ciCost +  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Branch investment cost (EUR)"] = sum_ciCost
                sce_data["Flexibility investment cost (EUR)"] =  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Total Operation Cost (EUR)"] =  0
                
                for xy in range(NoYear):
                    
                    Pflex[xy][path_sce[xp][xy]] =  [0 if abs(x)<=1e-4 else x for x in Pflex[xy][path_sce[xp][xy]]]
                    
                    sce_data[str(year_num[xy])] = {
                                            "Operation cost (EUR/year)": 0, 
                                            "Branch investment (MVA)":  ci[xy][path_sce[xp][xy]], 
                                            "Flexibility investment (MW)": Pflex[xy][path_sce[xp][xy]], 
                                         }
                
                output_data["Scenario " +str(xp+1)] = sce_data
                
        else:
            # only output two extreme scenarios
            temp_xp = 0
            for xp in [0,NoPath-1]:
                
                temp_xp += 1
                sce_data = {}
                sce_data["Total investment cost (EUR)"] = sum_ciCost +  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Branch investment cost (EUR)"] = sum_ciCost
                sce_data["Flexibility investment cost (EUR)"] =  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Net Present Operation Cost (EUR)"] =  0
                
                for xy in range(NoYear):
                    
                    Pflex[xy][path_sce[xp][xy]] =  [0 if abs(x)<=1e-4 else x for x in Pflex[xy][path_sce[xp][xy]]]
                        
                    sce_data[str(year_num[xy])] = {
                                            "Operation cost (EUR/year)": 0, 
                                            "Branch investment (MVA)":  ci[xy][path_sce[xp][xy]], 
                                            "Flexibility investment (MW)": Pflex[xy][path_sce[xp][xy]], 
                                         }
                    
                       
                    
             
                output_data["Scenario " +str(temp_xp)] = sce_data
    
            
    else: # part 2 output with operation cost
        
        # output all the pathways (scenarios)
        if outputAll == True:      
            for xp in range(NoPath):
                sce_data = {}
                sce_data["Total investment cost (EUR)"] = sum_ciCost +  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Branch investment cost (EUR)"] = sum_ciCost
                sce_data["Flexibility investment cost (EUR)"] =  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Total Operation Cost (EUR)"] =  sum_CO
                
                for xy in range(NoYear):
                    
                    Pflex[xy][path_sce[xp][xy]] =  [0 if abs(x)<=1e-4 else x for x in Pflex[xy][path_sce[xp][xy]]]
                    
                    sce_data[str(year_num[xy])] = {
                                            "Operation cost (EUR/year)": yearly_CO[xy][path_sce[xp][xy]], 
                                            "Branch investment (MVA)":  ci[xy][path_sce[xp][xy]], 
                                            "Flexibility investment (MW)": Pflex[xy][path_sce[xp][xy]], 
                                         }
                
                output_data["Scenario " +str(xp+1)] = sce_data
                
        else:
            # only output two extreme scenarios
            temp_xp = 0
            for xp in [0,NoPath-1]:
                
                temp_xp += 1
                sce_data = {}
                sce_data["Total investment cost (EUR)"] = sum_ciCost +  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Branch investment cost (EUR)"] = sum_ciCost
                sce_data["Flexibility investment cost (EUR)"] =  sum(Cflex[xy][path_sce[xp][xy]] for xy in range(NoYear) )
                sce_data["Net Present Operation Cost (EUR)"] =  sum_CO
                
                for xy in range(NoYear):
                    
                    Pflex[xy][path_sce[xp][xy]] =  [0 if abs(x)<=1e-4 else x for x in Pflex[xy][path_sce[xp][xy]]]
                     
                    sce_data[str(year_num[xy])] = {
                                            "Operation cost (EUR/year)": yearly_CO[xy][path_sce[xp][xy]], 
                                            "Branch investment (MVA)":  ci[xy][path_sce[xp][xy]], 
                                            "Flexibility investment (MW)": Pflex[xy][path_sce[xp][xy]], 
                                         }
                    
                       
                    
             
                output_data["Scenario " +str(temp_xp)] = sce_data
           
        
        
    file_name = "investment_result_" + country + "_" + test_case + pt
        
    ''' Output json file''' 
    with open('results/' + file_name +'.json', 'w') as fp:
        json.dump(output_data, fp)
    
    return print("Investment result file created")



# mpc, base_time_series_data,  multiplier = read_input_data("case5")


def get_time_series_data(mpc,  base_time_series_data, peak_hour = 19):
    # prepare base and peak data for optimisation
    
    

    load_bus = base_time_series_data["Load P (MW)"]["Bus \ Hour"].values.tolist()
    all_Pd = base_time_series_data["Load P (MW)"].values.tolist()
    all_Qd = base_time_series_data["Load Q (Mvar)"].values.tolist()
    
    base_Pd = [] #24h data
    base_Qd = []
    
    peak_Pd = []
    peak_Qd = []
    
    all_Pflex_up = base_time_series_data["Upward flexibility"].values.tolist()
    all_Pflex_dn = base_time_series_data["Downward flexibility"].values.tolist()

    base_Pflex_up = []
    base_Pflex_dn = []
    
    peak_Pflex_up = [] # Pflex_max in optimisation
    peak_Pflex_dn = [] # Pflex_max in optimisation
    
    # No input data for Q flex from current data set
    peak_Qflex_up = None
    peak_Qflex_dn = None
    
    for ib in range(mpc["NoBus"]):
        
        bus_i = mpc['bus']['BUS_I'][ib]
        # find if the bus has load
        load_bus_i = [i for i,x in enumerate(load_bus) if x == bus_i] 
        # record the load        
        if load_bus_i != []:
            # Load P and Q
            temp = all_Pd[load_bus_i[0]].copy()
            temp.pop(0)
            base_Pd.append(temp)
            
            temp = all_Qd[load_bus_i[0]].copy()
            temp.pop(0)
            base_Qd.append(temp)
            
            # Peak load P and Q
            peak_Pd.append(base_Pd[ib][peak_hour])
            peak_Qd.append(base_Pd[ib][peak_hour])
            
            # flex has the same connection of load
            # PFlex up and down ward
            temp = all_Pflex_up[load_bus_i[0]].copy()
            temp.pop(0)
            base_Pflex_up.append(temp)
            
            temp = all_Pflex_dn[load_bus_i[0]].copy()
            temp.pop(0)
            base_Pflex_dn.append(temp)
            
            # Peak Pflex up and down
            peak_Pflex_up.append(base_Pflex_up[ib][peak_hour])
            peak_Pflex_dn.append(base_Pflex_dn[ib][peak_hour])
                      
        # record 0 load
        else:
            temp = [0]*24
            base_Pd.append(temp)
            base_Qd.append(temp)
            base_Pflex_up.append(temp)
            base_Pflex_dn.append(temp)
            
            
            peak_Pd.append(0)
            peak_Qd.append(0)
            
            peak_Pflex_up.append(0)
            peak_Pflex_dn.append(0)
            
           
    print('read laod and flex data')         
    return (base_Pd , base_Qd ,peak_Pd ,peak_Qd ,base_Pflex_up, base_Pflex_dn , peak_Pflex_up , peak_Pflex_dn,peak_Qflex_up , peak_Qflex_dn,load_bus)





# peak load P for screening model
def get_peak_data(mpc,  base_time_series_data, peak_hour = 19):
       

    load_bus = base_time_series_data["Load P (MW)"]["Bus \ Hour"].values.tolist()
    all_Pd = base_time_series_data["Load P (MW)"].values.tolist()
    
    base_Pd = [] #24h data
    
    peak_Pd = []
    
   
    
    for ib in range(mpc["NoBus"]):
        
        bus_i = mpc['bus']['BUS_I'][ib]
        # find if the bus has load
        load_bus_i = [i for i,x in enumerate(load_bus) if x == bus_i] 
        # record the load        
        if load_bus_i != []:
            # Load P and Q
            temp = all_Pd[load_bus_i[0]].copy()
            temp.pop(0)
            base_Pd.append(temp)
            
           
            # Peak load P and Q
            peak_Pd.append(base_Pd[ib][peak_hour])
            
                      
        # record 0 load
        else:
            temp = [0]*24
            base_Pd.append(temp)
           
            peak_Pd.append(0)
            
        
        
    return peak_Pd



# ''' Main '''
# cont_list = []
# country = "PT" # Select country for case study: "PT", "UK" or "HR"
# test_case = 'Transmission_Network_PT_2020' 
# peak_hour = 19
# ci_catalogue = "Default"
# ci_cost = "Default"
# output_data = 0
# # read input data outputs mpc and load infor
# mpc, base_time_series_data,  multiplier, NoCon,ci_catalogue,ci_cost= read_input_data( cont_list, country,test_case,ci_catalogue,ci_cost)

# # get peak load for screening model
# peak_Pd = get_peak_data(mpc, base_time_series_data, peak_hour)

# # get all load, flex infor for investment model
# base_Pd , base_Qd ,peak_Pd ,peak_Qd ,base_Pflex_up, base_Pflex_dn , peak_Pflex_up , peak_Pflex_dn = get_time_series_data(mpc,  base_time_series_data)

# # save outputs
# output_data(output_data, country, test_case )


# read WP5 outputs of useful life
def read_asset_life():
    useful_life = pd.read_csv('tests/csv/mpc_useful_life.csv')
    
    
    return useful_life
