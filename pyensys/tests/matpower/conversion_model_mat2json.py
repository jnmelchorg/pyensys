# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:26:33 2021

@author: p96677wk
"""

import numpy as np
import logging
import math
import os
import json

# include 24 hourly demand P & Q    
class any2json:
    
    
    def matpower2json(self, folder_path=None, name_matpower=None, name_json=None):
        ''' This class method converts matpower files to json files with the 
        format required by pyene
        
        - The path must have the following characteristics:
            * folder_path = path\\to\\folder for windows
            * folder_path = path/to/folder for linux
        - The extension .m and .json must not be included in name_matpower and
        name_json
        '''
        assert folder_path != None, 'No directory path has been pass to load\
            the matpower file'
        assert name_matpower != None, 'No file name has been pass to load\
            the matpower file'
        
        filepath = os.path.join(folder_path, name_matpower+'.m')

        NoTime = 1 # number of time periods!

        jsonformat = {
            "metadata": {
                "title": name_json,
                "description": 'power system of '+name_json,
                "minimum_version": "0.4"
            },
            "version": 2,
            "baseMVA": None,
            "NoBus": None,
            "NoBranch": None,
            "NoGen": None,
            "Slack": None,
            "bus": {
                "BUS_I": [],
                "BUS_TYPE": [],
                "PD": [],
                "QD": [],
                "GS": [],
                "BS": [],
                "BUS_AREA": [],
                "VM": [],
                "VA": [],
                "BASE_KV": [],
                "ZONE": [],
                "VMAX": [],
                "VMIN": []
            },
            "branch": {
                "F_BUS": [],
                "T_BUS": [],
                "BR_R": [],
                "BR_X": [],
                "BR_B": [],
                "RATE_A": [],
                "RATE_B": [],
                "RATE_C": [],
                "TAP": [],
                "SHIFT": [],
                "BR_STATUS": [],
                "ANGMIN": [],
                "ANGMAX": []
            },
            "gen": {
                "GEN_BUS": [],
                "PG": [],
                "QG": [],
                "QMAX": [],
                "QMIN": [],
                "VG": [],
                "MBASE": [],
                "GEN": [],
                "PMAX": [],
                "PMIN": [],
                "PC1": [],
                "PC2": [],
                "QC1MIN": [],
                "QC1MAX": [],
                "QC2MIN": [],
                "QC2MAX": [],
                "RAMP_AGC": [],
                "RAMP_10": [],
                "RAMP_30": [],
                "RAMP_Q": [],
                "APF": []
            },
            "gencost": {
                "MODEL": [],
                "STARTUP": [],
                "SHUTDOWN": [],
                "NCOST": [],
                "COST": []
            },
            
            
            "demandP": {},
            "demandQ": {}
         
        }
        
        
        
        for t in range(NoTime):
            jsonformat['demandP'][t] = []
            jsonformat['demandQ'][t] = []
                       

        with open(filepath) as fp:
            line = fp.readline()
            flags_bus=False
            flags_gen=False
            flags_branch=False
            flags_gencost=False
            flags_demandP=False
            flags_demandQ=False
            
            while line:
                if line.split() != [] and line.split()[0] == 'mpc.baseMVA':
                    aux = ""
                    for x in line.split()[2]:
                        if x != ";":
                            aux = aux + x
                    jsonformat['baseMVA'] = float(aux)
                elif line.split() != [] and line.split()[0] == 'mpc.bus':
                    flags_bus = True
                    line = fp.readline()
                    continue
                elif line.split() != [] and line.split()[0] == 'mpc.gen':
                    flags_gen = True
                    line = fp.readline()
                    continue
                elif line.split() != [] and line.split()[0] == 'mpc.branch':
                    flags_branch = True
                    line = fp.readline()
                    continue
                elif line.split() != [] and line.split()[0] == 'mpc.gencost':
                    flags_gencost = True
                    line = fp.readline()
                elif line.split() != [] and line.split()[0] == 'mpc.demandP':
                    flags_demandP = True
                    line = fp.readline()
                elif line.split() != [] and line.split()[0] == 'mpc.demandQ':
                    flags_demandQ = True
                    line = fp.readline()
              
                    continue
                
                if flags_bus and line.split() != [] and line.split()[0] != '];':
                    aux1 = line.split()
                    for val, pos in zip(aux1, jsonformat['bus'].keys()):
                        if pos == 'BUS_TYPE' and int(aux1[1]) == 3:
                            jsonformat['Slack'] = int(aux1[0])
                            jsonformat['bus']['BUS_TYPE'].append(int(aux1[1]))
                        elif pos == 'VMIN':
                            aux2 = ""
                            for x in val:
                                if x != ";":
                                    aux2 = aux2 + x
                            jsonformat['bus']['VMIN'].append(float(aux2))
                        elif pos != 'BUS_I' and pos != 'BUS_TYPE' and \
                            pos != 'BUS_AREA' and pos != 'ZONE':
                            jsonformat['bus'][pos].append(float(val))
                        else:
                            jsonformat['bus'][pos].append(int(val))
                elif flags_bus and line.split() != [] and line.split()[0] == '];':
                    flags_bus = False
                
                if flags_gen and line.split() != [] and line.split()[0] != '];':
                    aux1 = line.split()
                    for val, pos in zip(aux1, jsonformat['gen'].keys()):
                        if pos == 'APF':
                            aux2 = ""
                            for x in val:
                                if x != ";":
                                    aux2 = aux2 + x
                            jsonformat['gen']['APF'].append(float(aux2))
                        elif pos != 'GEN_BUS' and pos != 'GEN':
                            jsonformat['gen'][pos].append(float(val))
                        else:
                            jsonformat['gen'][pos].append(int(val))
                elif flags_gen and line.split() != [] and line.split()[0] == '];':
                    flags_gen = False
                
                if flags_branch and line.split() != [] and line.split()[0] != '];':
                    aux1 = line.split()
                    for val, pos in zip(aux1, jsonformat['branch'].keys()):
                        if pos == 'ANGMAX':
                            aux2 = ""
                            for x in val:
                                if x != ";":
                                    aux2 = aux2 + x
                            jsonformat['branch']['ANGMAX'].append(float(aux2))
                        elif pos != 'F_BUS' and pos != 'T_BUS' and \
                            pos != 'BR_STATUS':
                            jsonformat['branch'][pos].append(float(val))
                        else:
                            jsonformat['branch'][pos].append(int(val))
                elif flags_branch and line.split() != [] and line.split()[0] == '];':
                    flags_branch = False
                
                if flags_gencost and line.split() != [] and line.split()[0] != '];':
                    aux1 = line.split()
                    cnt = 0
                    for pos in jsonformat['gencost'].keys():
                        if pos == 'COST':
                            auxlist = []
                            for x in range(int(aux1[3])):
                                if x < int(aux1[3]) - 1:
                                    auxlist.append(float(aux1[cnt + x]))
                                else:
                                    aux2 = ""
                                    for x1 in aux1[cnt + x]:
                                        if x1 != ";":
                                            aux2 = aux2 + x1
                                    auxlist.append(int(aux2))
                            jsonformat['gencost']['COST'].append(auxlist)
                        elif pos != 'MODEL' and pos != 'NCOST':
                            jsonformat['gencost'][pos].append(float(aux1[cnt]))
                        else:
                            jsonformat['gencost'][pos].append(int(aux1[cnt]))
                        cnt += 1
                elif flags_gencost and line.split() != [] and line.split()[0] == '];':
                    flags_gencost = False     
                    
                 ###############################################   
                 #########       Demand in 24 hours  ###########
                 ###############################################
                 
                if flags_demandP and line.split() != [] and line.split()[0] != '];':
                   aux1 = line.split()
                   auxlist = []
                   for t in range(NoTime):    
                       auxlist.append(float(aux1[t]))
                       jsonformat['demandP'][t].append(auxlist[t])
                     
                elif flags_demandP and line.split() != [] and line.split()[0] == '];':
                    flags_demandP = False
                
                    
                
                if flags_demandQ and line.split() != [] and line.split()[0] != '];':
                   aux1 = line.split()
                   auxlist = []
                   for t in range(NoTime):    
                       auxlist.append(float(aux1[t]))
                       jsonformat['demandQ'][t].append(auxlist[t])
                     
                elif flags_demandQ and line.split() != [] and line.split()[0] == '];':
                    flags_demandQ = False
                    
                    
                    
                line = fp.readline()
        
        jsonformat['NoBus'] = len(jsonformat['bus']['BUS_I'])
        jsonformat['NoBranch'] = len(jsonformat['branch']['F_BUS'])
        jsonformat['NoGen'] = len(jsonformat['gen']['GEN_BUS'])

        filepath = os.path.join(folder_path, name_json+'.json')
        with open(filepath, 'w') as json_file:
            json.dump(jsonformat, json_file, indent=4)



# # Interaction node
# class _node():
#     def __init__(self):
#         self.value = None
#         self.index = None
#         self.bus = None
#         self.marginal = None
#         self.flag = False

def json_directory():
    ''' Directory contain JSON files for pytest '''
    return os.path.join(os.path.dirname(__file__))

 
# ##################################################################
# ##################################################################
# '''main'''
# # print('Convert to json')
# NoTime = 1
# converter = any2json()

# # country = 'Transmission_Network_UK2'#'Transmission_Network_UK2'
# country = 'Distribution_Network_Rural_UK'#'Transmission_Network_UK2'
# #country = 'case5t'

# converter.matpower2json(folder_path=json_directory(), \
#                         name_matpower=country, name_json=country)
    
    
# # # Load json file
# # mpc = json.load(open(os.path.join(os.path.dirname(__file__), 
# #                                   'tests', 'json', 
# #                                   'Transmission_Network_UK2.json'))) 
    