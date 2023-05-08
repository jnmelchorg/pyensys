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
import pandas as pd # to read new EV load data

def clust_verification_function(case_path,use_load_data_update,add_load_data_case_name,line_tap_vector): # Note: multipliers and clusters are saved in json and then read whitin this function

    # # Export the network from a Matpower file:
    # test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_Urban_UK_new.mat'
    # test_system_path = join(dirname(__file__), "..", "tests\\matpower\\", "Distribution_Network_Urban_UK_new.mat")

    if case_path[-1] == 'm':
        test_system_path = case_path[:-1] + 'mat' # read .mat file, not .m file
    else:
        test_system_path = case_path

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

    N_lines = len(screen_result_final_interv_clust[0]) # number of lines in the system
    print('\nN_lines: ',N_lines)

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

    N_tree_years = len(mult_bus) # number of years (time intervals)
    print('\nN_tree_years: ',N_tree_years)
    N_tree_nodes = 0 # number of nodes in the tree
    for n in range(N_tree_years):
        N_tree_nodes += 2**n
    print('\nN_tree_nodes: ',N_tree_nodes)

    solutions_lines = [[0]*N_lines for i in range(N_tree_nodes)]
    solutions_investment_costs = [[0] for i in range(N_tree_nodes)]
    solutions_operation_costs = [[0] for i in range(N_tree_nodes)]

    year_name = [2020, 2030, 2040, 2050] # to navigate in the EV-PV Excel table

    print('\nline_tap_vector:')
    print(line_tap_vector)

    node_i = 0
    no_feasible_plan = False # indicate if there are no feasible clusters found for a specific year/scenario
    for year in range(N_tree_years):
        for node in range(len(mult_bus[year])):
            print('--------------- year = ',year,', node_i = ',node_i,'---------------')

            # # Read the initial network:
            net_test = from_mpc(test_system_path)

            # print('\nnet_test:')
            # print(net_test['bus']['name'])
            # print(len(net_test['bus']['name']))

            if use_load_data_update == 1:
                if year != 0:
                    # EV_data_path = "C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests" # to be updated
                    EV_data_path = join(dirname(__file__), "..", "tests\\")
                    EV_data_file_name = 'EV-PV-Storage_Data_for_Simulations.xlsx'
                    EV_data_file_path = join(EV_data_path, EV_data_file_name)
                    EV_data_sheet_names = add_load_data_case_name
                    # EV_data_sheet_names = 'PT_Dx_01_' # set via CLI
                    # EV_data_sheet_names = 'HR_Dx_01_' # set via CLI
                    # EV_data_sheet_names = 'UK_Dx_01_' # set via CLI

                    EV_load_data = pd.read_excel(EV_data_file_path, sheet_name = str(EV_data_sheet_names) + str(year_name[year]), skiprows = 1)
                    EV_load_data_MW_profile = EV_load_data["EV load (MW)"]
                    EV_load_data_MW_max = np.max(EV_load_data_MW_profile[0:24])
                    Pd_additions = EV_load_data["Node Ratio"]*EV_load_data_MW_max # how much new load per node
                else:
                    Pd_additions = [0] * len(net_test['bus']['name']) # zero additional EV load
            else:
                Pd_additions = [0] * len(net_test['bus']['name']) # zero additional EV load
            

            # print('\nnet_test.load[p_mw]:')
            # print(net_test.load['p_mw'])
            # print('\nnet_test.load[q_mvar]:')
            # print(net_test.load['q_mvar'])
            # print('\nnet_test.load:')
            # print(net_test.load)
            # print('\nPd_additions:')
            # print(Pd_additions)

            # # Increase loads using multipliers for nodes in the scenario tree:
            net_test.load['p_mw'] = net_test.load['p_mw']*mult_bus[year][node][0] # same load increase for all buses
            net_test.load['q_mvar'] = net_test.load['q_mvar']*mult_bus[year][node][0] # same load increase for all buses

            if use_load_data_update == 1: ## add new EV loads
                for lx in range(len(net_test.load['p_mw'])):
                    net_test.load['p_mw'][lx] += Pd_additions[net_test.load['bus'][lx]]

            print('Total P: ',sum(net_test.load['p_mw']),'(MW), Total Q: ',sum(net_test.load['q_mvar']),'(MVAr)')

            # # # Manually reduce loads (flexibility tests):
            # net_test.load['p_mw'] = net_test.load['p_mw']*mult_bus[year][node][0]*0.9
            # net_test.load['q_mvar'] = net_test.load['q_mvar']*mult_bus[year][node][0]*0.9

            # # Replace the external grid to avoid voltage issues:
            replace_ext_grid = True # to make the external grid a generator (to control voltages beyond 1.0pu)
            # replace_ext_grid = False

            if replace_ext_grid == True:
                if len(net_test.ext_grid) == 1: ## if there is only one external grid (interface), e.g., in UK distribution networks
                    net_test.poly_cost['et'] = 'gen' # could be a problem if we have multiple generators/external grids
                    # print('\nnet_test.poly_cost:')
                    # print(net_test.poly_cost)

                    # # let's try to create a gerenator instead of ext_grid
                    create_gen(net_test, bus=net_test.ext_grid['bus'][0], p_mw=1, vm_pu=1.00, max_q_mvar=net_test.ext_grid['max_q_mvar'][0], min_q_mvar=net_test.ext_grid['min_q_mvar'][0],\
                                        min_p_mw=net_test.ext_grid['min_p_mw'][0], max_p_mw=net_test.ext_grid['max_p_mw'][0], scaling=1.0, slack=True, controllable=True)

                    # del net_test.ext_grid # impossible to just delete - need to replace with an empty ext_grid
                    empty_net = create_empty_network()
                    empty_ext_grid = empty_net.ext_grid

                    net_test.ext_grid = empty_net.ext_grid
                else:
                    for ext_grid_i in range(len(net_test.ext_grid)):
                        net_test.poly_cost['et'] = 'gen'
                        # print('\nnet_test.poly_cost:')
                        # print(net_test.poly_cost)

                        # # let's try to create a gerenator instead of ext_grid
                        create_gen(net_test, bus=net_test.ext_grid['bus'][ext_grid_i], p_mw=1, vm_pu=1.00, max_q_mvar=net_test.ext_grid['max_q_mvar'][ext_grid_i], min_q_mvar=net_test.ext_grid['min_q_mvar'][ext_grid_i],\
                                            min_p_mw=net_test.ext_grid['min_p_mw'][ext_grid_i], max_p_mw=net_test.ext_grid['max_p_mw'][ext_grid_i], scaling=1.0, slack=True, controllable=True)

                    ## now remove all external grids:
                    empty_net = create_empty_network()
                    empty_ext_grid = empty_net.ext_grid
                    net_test.ext_grid = empty_net.ext_grid

                    ## remove multiple costs assigned to some elements (e.g., external grid and generator at the same bus):
                    existing_gen = [] # write down generators to find duplicates
                    for poly_cost_i in range(len(net_test.poly_cost)):
                        duplicate = False
                        next_gen = net_test.poly_cost['element'][poly_cost_i]
                        # print('existing_gen: ',existing_gen)
                        # print('next_gen: ',next_gen)

                        try:
                            find_next_gen_in_existing = existing_gen.index(next_gen)
                            # print('find_next_gen_in_existing: ',find_next_gen_in_existing)
                            duplicate = True
                        except:
                                pass
                        
                        if duplicate == False:
                            existing_gen += [net_test.poly_cost['element'][poly_cost_i]]
                        else:   
                            net_test.poly_cost.drop(poly_cost_i, inplace=True)


            # # Solve OPF for different clusters:
            solution_found = 0
            for cl in range(N_clust):
                if solution_found == 0:
                    print('Verifying cluster #',cl)

                    # screen_result_final_interv_clust[cl] = [x/2 for x in screen_result_final_interv_clust[cl]] # <--- use this to create infeasibility (for testing purposes)

                    
                    try:
                        ## Add line capacity upgrades:
                        if len(net_test.trafo) == 0: # if there are no transformers - update only line capacities
                            for l in range(len(screen_result_final_interv_clust[cl])):
                                # net_test.line['max_i_ka'][l] += screen_result_final_interv_clust[cl][l]*(10**6)/(11*(10**3))/(10**3) # ! works only for 11kV network tests

                                v_kv = net_test.bus['vn_kv'][net_test.line['from_bus'][l]]
                                
                                net_test.line['max_i_ka'][l] += screen_result_final_interv_clust[cl][l]*(10**6)/(v_kv*(10**3))/(10**3)

                        else: # update both line and transformer capacities
                            l_count = 0
                            t_count = 0
                            for l in range(len(screen_result_final_interv_clust[cl])):
                                if line_tap_vector[l] == 0: # this is a line
                                    # print('updating line ',net_test.line['from_bus'][l_count],'-',net_test.line['to_bus'][l_count],' by ',screen_result_final_interv_clust[cl][l],'MVA')

                                    v_kv = net_test.bus['vn_kv'][net_test.line['from_bus'][l_count]]
                                    net_test.line['max_i_ka'][l_count] += screen_result_final_interv_clust[cl][l]*(10**6)/(v_kv*(10**3))/(10**3)
                                    l_count += 1
                                else: # this is a transformer
                                    # print('updating transformer ',net_test.trafo['hv_bus'][t_count],'-',net_test.trafo['lv_bus'][t_count],' by ',screen_result_final_interv_clust[cl][l],'MVA')

                                    net_test.trafo['sn_mva'][t_count] += screen_result_final_interv_clust[cl][l]
                                    t_count += 1

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
                        solutions_lines[node_i] = screen_result_final_interv_clust[cl]
                        solutions_investment_costs[node_i] = clust_costs[cl]
                    else:
                        print('No solution found - the OPF model did not converge!')

                if cl == N_clust-1 and solution_found == 0:
                    print('No feasible solutions found - all clusters lead to infeasible network operation!')
                    no_feasible_plan = True
            node_i += 1

    return ([solutions_lines, solutions_investment_costs, solutions_operation_costs], no_feasible_plan)