from pandapower.converter import from_mpc
from pandapower import runpp, runopp
from pandapower import from_json
from pandapower import networks

# # export from Matpower:
test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_Urban_UK_new.mat'
net_test = from_mpc(test_system_path)

# # to export from json:
# test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\self_network_export.json'
# net_test = from_json(test_system_path)

# # Some defined Panda Power cases:
# net_test = networks.case33bw()

print('net_test:')
print(net_test)

# Increase loads manually (using a load growth multiplier):
net_test.load['p_mw'] = net_test.load['p_mw']*1.93
# net_test.load['q_mvar'] = net_test.load['q_mvar']*1.93

# print('\nnet_test.load:')
# print(net_test.load['p_mw'])
# print(net_test.load)

# # Add line capacity upgrades:
line_upgrades = [14-1,15-1,16-1,29-1]
upgrades_MVA = [5.0,0.75,0.75,2.0]

lf = 0
for l in line_upgrades:
    net_test.line['max_i_ka'][l] += upgrades_MVA[lf]*(10**6)/(11*(10**3))/(10**3)
    lf += 1

# print('\nnet_test.line:')
# print(net_test.line)
# print(net_test.line['max_i_ka'][0])

# net_test.ext_grid['vm_pu'] = 1.06 # manually adjust voltage

print('net_test.ext_grid:')
print(net_test.ext_grid)

print('net_test.bus:')
print(net_test.bus)



# # runpp(net_test,numba=False) # run power flow
runopp(net_test,numba=False) # run OPF

print('\nnet_test:')
print(net_test)
print('\nnet_test.res_bus:')
print(net_test.res_bus)