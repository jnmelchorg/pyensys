'''
This script verifies the AC feasibility of the identified clusters
Then, the feasible clusters with lowest costs are selected for each node in the scenario tree

@author: Andrey Churkin
    
'''


from pandapower.converter import from_mpc
from pandapower import runpp, runopp
# from pandapower import from_json
from pandapower import networks, create_gen, delete_std_type, create_empty_network
from os.path import join, dirname
import json
import numpy as np

# # Export the network from a Matpower file:
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_Urban_UK_new.mat'
test_system_path = join(dirname(__file__), "..", "tests\\matpower\\", "Distribution_Network_Urban_UK_new.mat")

# net_test = from_mpc(test_system_path)

# # Read clusters data:
with open(join(dirname(__file__), "..", "tests\\outputs\\")+ "screen_result_final_interv_clust.json", 'r') as f:
    screen_result_final_interv_clust = json.load(f)
# # Read load multipliers:   
with open(join(dirname(__file__), "..", "tests\\outputs\\")+ "mult_bus_export.json", 'r') as f:
    mult_bus = json.load(f)

print('\nmult_bus:')
print(mult_bus)

screen_result_final_interv_clust.append([0]*len(screen_result_final_interv_clust[0])) # create a cluster with no investments

N_clust = len(screen_result_final_interv_clust) # number of clusters
print('\nN_clust:')
print(N_clust)

# clust_costs = np.zeros(N_clust)
clust_costs = [0]*N_clust
for cl in range(N_clust):
    clust_costs[cl] = sum(screen_result_final_interv_clust[cl])*60000 # this cost assumtion £60000 per MVA per km can be verified

# # Sorting clusters by costs in increasing order:
sorted_indices = np.argsort(clust_costs)
screen_result_final_interv_clust = [screen_result_final_interv_clust[i] for i in sorted_indices]
clust_costs = [clust_costs[i] for i in sorted_indices]

print('\nscreen_result_final_interv_clust:')
print(screen_result_final_interv_clust)
print('\nclust_costs:')
print(clust_costs)

N_tree_years = len(mult_bus) # numebr of years (time intervals)
print('\nN_tree_nodes: ',N_tree_years)

node_i = 0
for year in range(N_tree_years):
    for node in range(len(mult_bus[year])):
        print('--------------- year = ',year,', node_i = ',node_i,'---------------')

        # # Read the initial network:
        net_test = from_mpc(test_system_path)

        # # # Increase loads using multipliers for nodes in the scenario tree:
        # net_test.load['p_mw'] = net_test.load['p_mw']*mult_bus[year][node][0] # same load increase for all buses
        # net_test.load['q_mvar'] = net_test.load['q_mvar']*mult_bus[year][node][0] # same load increase for all buses

        # # Manually reduce loads (flexibility):
        net_test.load['p_mw'] = net_test.load['p_mw']*mult_bus[year][node][0]*0.9
        net_test.load['q_mvar'] = net_test.load['q_mvar']*mult_bus[year][node][0]*0.9

        # # Replace the external grid to avoid voltage issues:
        net_test.poly_cost['et'] = 'gen' # could be a problem if we have multiple generators/external grids
        create_gen(net_test, bus=net_test.ext_grid['bus'][0], p_mw=1, vm_pu=1.00, max_q_mvar=net_test.ext_grid['max_q_mvar'][0], min_q_mvar=net_test.ext_grid['min_q_mvar'][0],\
                       min_p_mw=net_test.ext_grid['min_p_mw'][0], max_p_mw=net_test.ext_grid['max_p_mw'][0], scaling=1.0, slack=True, controllable=True)
        empty_net = create_empty_network()
        empty_ext_grid = empty_net.ext_grid
        net_test.ext_grid = empty_net.ext_grid

        # # Solve OPF for different clusters:
        solution_found = 0
        for cl in range(N_clust):
            if solution_found == 0:
                print('Verifying cluster #',cl)
                try:
                    # # Add line capacity upgrades:
                    for l in range(len(screen_result_final_interv_clust[cl])):
                        net_test.line['max_i_ka'][l] += screen_result_final_interv_clust[cl][l]*(10**6)/(11*(10**3))/(10**3)

                    runopp(net_test,numba=False, verbose=0)
                    solution_found = 1
                except:
                    pass

                if solution_found == 1:
                    print('OPF model converged!')
                    # print('\nnet_test:')
                    # print(net_test)
                    # print('\nnet_test.res_bus:')
                    # print(net_test.res_bus)
                else:
                    print('No solution found - the OPF model did not converged!')
        node_i += 1
