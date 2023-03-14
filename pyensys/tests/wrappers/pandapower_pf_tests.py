from pandapower.converter import from_mpc
from pandapower import runpp, runopp

test_system_path = 'C:\\Users\\m36330ac\\Documents\\MEGA\\Eduardo Alejandro Martinez Cesena\\WP3\\Python\\from Nicolas\\pyensys\\pyensys\\tests\\matpower\\Distribution_Network_Urban_UK_new.mat'

net_test = from_mpc(test_system_path)

# # Increase loads manually (using a load growth multiplier):
# net_test.load['p_mw'] = net_test.load['p_mw']*1.93

print('\nnet_test.load:')
print(net_test.load['p_mw'])

# # Add line capacity upgrades:
# line_upgrades = [14-1,15-1,16-1,29-1]
# upgrades_MVA = [5.0,0.75,0.75,2.0]

# lf = 0
# for l in line_upgrades:
#     net_test.line['max_i_ka'][l] += upgrades_MVA[lf]*(10**6)/(11*(10**3))/(10**3)
#     lf += 1

print('\nnet_test.line:')
print(net_test.line)
# print(net_test.line['max_i_ka'][0])


# runpp(net_test,numba=False) # run power flow
runopp(net_test,numba=False) # run OPF

print('\nnet_test:')
print(net_test)
print('\nnet_test.res_bus:')
print(net_test.res_bus)