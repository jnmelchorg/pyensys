from pandapower.converter import from_mpc
from pandapower import runpp, runopp
from pandapower import from_json
from pandapower import networks, create_gen, delete_std_type, create_empty_network

# # export from Matpower:
test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_Urban_UK_new.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\case33bw (mod) v1.1.mat'
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\case33bw.mat'

net_test = from_mpc(test_system_path)

# # to export from json:
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\self_network_export.json'
# net_test = from_json(test_system_path)

# # Some defined Panda Power cases:
# net_test = networks.case33bw()

# print('net_test:')
# print(net_test)


# # Increase loads manually (using a load growth multiplier):
# net_test.load['p_mw'] = net_test.load['p_mw']*1.20
# net_test.load['q_mvar'] = net_test.load['q_mvar']*1.93

# net_test.load['controllable'][0] = True

print('\nnet_test.load:')
# print(net_test.load['p_mw'])
print(net_test.load)


# net_test.bus['max_vm_pu'][0] = 1.0
# net_test.bus['max_vm_pu'][6] = 1.06
print('\nnet_test.bus:')
print(net_test.bus)

net_test.poly_cost['et'] = 'gen' # could be a problem if we have multiple generators/external grids
print('\nnet_test.poly_cost:')
print(net_test.poly_cost)

# # let's try to create a gerenator instead of ext_grid
create_gen(net_test, bus=net_test.ext_grid['bus'][0], p_mw=1, vm_pu=1.00, max_q_mvar=net_test.ext_grid['max_q_mvar'][0], min_q_mvar=net_test.ext_grid['min_q_mvar'][0],\
                       min_p_mw=net_test.ext_grid['min_p_mw'][0], max_p_mw=net_test.ext_grid['max_p_mw'][0], scaling=1.0, slack=True, controllable=True)

# del net_test.ext_grid # impossible to just delete - need to replace with an empty ext_grid

# # Add line capacity upgrades:
# line_upgrades = [14-1,15-1,16-1,29-1]
# upgrades_MVA = [5.0,0.75,0.75,2.0]

# lf = 0
# for l in line_upgrades:
#     net_test.line['max_i_ka'][l] += upgrades_MVA[lf]*(10**6)/(11*(10**3))/(10**3)
#     lf += 1

# print('\nnet_test.line:')
# print(net_test.line)
# print(net_test.line['max_i_ka'][0])

# net_test.ext_grid['vm_pu'] = 1.06 # manually adjust voltage

# net_test.ext_grid['in_service'] = False
# delete_std_type(net_test, 'ext_grid')
# del net_test.
# net_test.ext_grid = []

empty_net = create_empty_network()
empty_ext_grid = empty_net.ext_grid

net_test.ext_grid = empty_net.ext_grid




print('net_test:')
print(net_test)

# print('net_test.sgen:')
# print(net_test.sgen)

# print('net_test.ext_grid:')
# print(net_test.ext_grid)

# print('net_test.bus:')
# print(net_test.bus)

print('net_test.gen:')
print(net_test.gen)



# # runpp(net_test,numba=False) # run power flow
runopp(net_test,numba=False) # run OPF

print('\nnet_test:')
print(net_test)
print('\nnet_test.res_bus:')
print(net_test.res_bus)