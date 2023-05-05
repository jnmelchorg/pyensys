from pandapower.converter import from_mpc
from pandapower import runpp, runopp
from pandapower import from_json
from pandapower import networks, create_gen, delete_std_type, create_empty_network

# # export from Matpower:
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_Urban_UK_new.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\case33bw (mod) v1.1.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\case33bw.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\A_KPC_35_v2.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\test_save.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\A_KPC_35_v2b.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\A_KPC_35_v2c.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_PT1_new_v2.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_Urban_UK_new_contingency_7_2.mat'
test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\A_BJ_35_v2c.mat'


net_test = from_mpc(test_system_path)

# # to export from json:
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\self_network_export.json'
# net_test = from_json(test_system_path)

# # Some defined Panda Power cases:
# net_test = networks.case33bw()

# print('net_test:')
# print(net_test)


# # Increase loads manually (using a load growth multiplier):
# net_test.load['p_mw'] = net_test.load['p_mw']*2.2     #*1.93
# net_test.load['q_mvar'] = net_test.load['q_mvar']*2.2    #*1.93

# net_test.load['controllable'][0] = True

# print('\nnet_test.load:')
# print(net_test.load['p_mw'])
# print(net_test.load)

# print('\nnet_test.ext_grid:')
# print(net_test.ext_grid)
# print('\nlen(net_test.ext_grid): ', len(net_test.ext_grid))


# net_test.bus['max_vm_pu'][0] = 1.0
# net_test.bus['max_vm_pu'][6] = 1.06
# print('\nnet_test.bus:')
# print(net_test.bus)

replace_ext_grid = True # to make the external grid a generator (to control voltages beyond 1.0pu)
# replace_ext_grid = False

if replace_ext_grid == True:
    if len(net_test.ext_grid) == 1: ## if there is only one external grid (interface), e.g., in UK distribution networks
        net_test.poly_cost['et'] = 'gen' # could be a problem if we have multiple generators/external grids
        print('\nnet_test.poly_cost:')
        print(net_test.poly_cost)

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
            print('existing_gen: ',existing_gen)
            print('next_gen: ',next_gen)

            try:
                find_next_gen_in_existing = existing_gen.index(next_gen)
                print('find_next_gen_in_existing: ',find_next_gen_in_existing)
                duplicate = True
            except:
                    pass
            
            if duplicate == False:
                existing_gen += [net_test.poly_cost['element'][poly_cost_i]]
            else:   
                net_test.poly_cost.drop(poly_cost_i, inplace=True)


# # Add line capacity upgrades:
# line_upgrades = [14-1,15-1,16-1,29-1]
# upgrades_MVA = [5.0,0.75,0.75,2.0]

# lf = 0
# for l in line_upgrades:
#     net_test.line['max_i_ka'][l] += upgrades_MVA[lf]*(10**6)/(11*(10**3))/(10**3) # ! works only for 11kV network tests
#     lf += 1

# print('\nnet_test.line:')
# print(net_test.line)
# print(net_test.line['max_i_ka'][0])

# net_test.ext_grid['vm_pu'] = 1.06 # manually adjust voltage

# net_test.ext_grid['in_service'] = False
# delete_std_type(net_test, 'ext_grid')
# del net_test.
# net_test.ext_grid = []






print('net_test:')
print(net_test)

print('net_test.ext_grid:')
print(net_test.ext_grid)

print('net_test.bus:')
print(net_test.bus)

print('net_test.line:')
print(net_test.line)

print('net_test.gen:')
print(net_test.gen)

print('net_test.sgen:')
print(net_test.sgen)

print('net_test.trafo:')
print(net_test.trafo)

print('net_test.poly_cost:')
print(net_test.poly_cost)


net_test.gen['slack'][2] = False # <---- manual fix for some cases, like A_BJ_35.m 


# # runpp(net_test,numba=False) # run power flow
runopp(net_test,numba=False, verbose=1) # run OPF

# solution_found = 0
# try:
#     runopp(net_test,numba=False, verbose=1)
#     solution_found = 1
# except:
#     pass

# if solution_found == 1:
#     print('\nA solution was found - the OPF model converged!')
#     print('\nnet_test:')
#     print(net_test)
#     print('\nnet_test.res_bus:')
#     print(net_test.res_bus)
# else:
#     print('\nNo solution found - the OPF model did not converged!')

