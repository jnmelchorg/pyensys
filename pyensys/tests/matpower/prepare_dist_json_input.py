import os
import json
import numpy

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

print(abspath)
print(dname)

## function to read MATPOWER files:
# from conversion_model_mat2json import any2json
from pyensys.Optimisers.conversion_model_mat2json import any2json
## function to get demand multipliers for future years
# from scenarios_multipliers import get_multipliers
from pyensys.Optimisers.scenarios_multipliers import get_mult as get_multipliers


converter = any2json()

# country = 'Distribution_Network_Rural_UK'
# country = 'case3' # 3-bus simple case by Nicolas
# country = 'A_KPC_35' # Croatian case HR_Dx_01_2020
# country = 'Distribution_Network_PT1'
country = 'Distribution_Network_Semi_Urban_UK_v2'

converter.matpower2json(folder_path=dname, \
                        name_matpower=country, name_json=country)
    
    

mult = get_multipliers("UK") # select "PT", "UK", or "HR"


# Load network json file
mpc_0 = json.load(open(os.path.join(os.path.dirname(__file__), str(country)+".json"))) 

ref_bus = mpc_0["Slack"] # there could be errors if the load forecast is set in the JSON input for the reference bus

# Years = [2020, 2030] # number of years (snapshots) in the planning horizon
# Years = [2020]
Years = [2020, 2030, 2040, 2050]
N_sc = 1 # number of scenarios to construct the planning decision tree


load_increase = True

attest_input = {
    "columns_names": ["scenario", "year", "bus_index", "p_mw", "q_mvar"],
    "data": []
}

# mult['2030'][0] = 2 # test - manual load increase


## Note that bus numbering starts with "0" in Nicolas format!
## Note that scenarios start with "1" in Nicolas format!
for n_sc in range(N_sc):
    for years in range(len(Years)):
        for i in range(mpc_0["NoBus"]):
            # if mpc_0["bus"]["BUS_I"][i] != ref_bus: # exclude the reference bus
            if mpc_0["bus"]["PD"][i]+mpc_0["bus"]["QD"][i] != 0: # exclude buses with no load
                if load_increase == True:
                    ## Note - multipliers should be fixed to match each scenario!
                    ## Note - 'i-1' must be used to correctly prepare nodes numbering for input json file
                    attest_input['data'].append([n_sc+1, Years[years], mpc_0["bus"]["BUS_I"][i-1], mpc_0["bus"]["PD"][i]*mult[years][0], mpc_0["bus"]["QD"][i]*mult[years][0] ])
                else:       
                    attest_input['data'].append([n_sc+1, Years[years], mpc_0["bus"]["BUS_I"][i-1], mpc_0["bus"]["PD"][i], mpc_0["bus"]["QD"][i]])


with open('attest_input.json', 'w') as outfile:
    json.dump(attest_input, outfile, indent=4)
    