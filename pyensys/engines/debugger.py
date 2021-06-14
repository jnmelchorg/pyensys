from pyensys.engines.main import pyeneClass
import os
import numpy as np

import time
begin = time.time()

# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\Pyene format proposal v1.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\118bus.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\Input Data - Myanmar_v1.xlsx'
# path = "C:\\Users\\f09903jm\\Downloads pc\\EAPP_pyene.xlsx"
# path = "C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\EAPP\\EAPP Data\\EAPP\\EAPP_1h.xlsx"

# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\Pyene format proposal v1.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\Pyene format proposal v3.xlsx'

# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\case4.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\case30.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\case118.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\case300.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\case1354pegase.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\case2736sp.xlsx'

path = 'C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\excel\\case4.xlsx'

opt = pyeneClass()
opt.initialise(path=path)
end1 = time.time()

subscripts = {
    "pt" : [["week", "weekday"], "v_string"],
    "hour" : [17.0, "double"]
}

opt.run(subscripts=subscripts)
end2 = time.time()

opt.save_outputs(sim_no=0)
end3 = time.time()

# information = {}
# information["name"] = ["string", "Pmax"]
# information["pt"] = ["v_string", ["week", "weekday"]]
# information["hour"] = ["double", 17.0]

# information["problem"] = ["string", "DC OPF"]
# information["ID"] = ["string", "T37"]
# information["value"] = ["double", 0]

# opt.update_parameter(information=information)
# opt.run(subscripts=subscripts)
# opt.save_outputs(sim_no=1)
# end4 = time.time()

# information = {}
# information["name"] = ["string", "Pmax"]
# information["pt"] = ["v_string", ["week", "weekday"]]
# information["hour"] = ["double", 1.0]
# information["problem"] = ["string", "DC OPF"]
# information["ID"] = ["string", "T37"]
# information["value"] = ["double", 300]

# opt.update_parameter(information=information)
# opt.run(subscripts=subscripts)
# opt.save_outputs(sim_no=2)
# end5 = time.time()

# subscripts = {
#     "pt" : [["week", "weekend"], "v_string"],
#     "hour" : [1.0, "double"]
# }

# opt.run(subscripts=subscripts)
# end6 = time.time()

# opt.save_outputs(sim_no=3)
# end7 = time.time()

# information = {}
# information["name"] = ["string", "Pmax"]
# information["problem"] = ["string", "DC OPF"]
# information["ID"] = ["string", "H2"]

# information["value"] = ["double", 100]
# information["pt"] = ["v_string", ["week", "weekend"]]
# information["hour"] = ["double", 1.0]

# opt.update_parameter(information=information)
# end8 = time.time()
# opt.run(subscripts=subscripts)
# opt.save_outputs(sim_no=4)
# end9 = time.time()


print("TIEMPO 1 {}".format(end1-begin))
print("TIEMPO 2 {}".format(end2-end1))
print("TIEMPO 3 {}".format(end3-end2))
# print("TIEMPO 4 {}".format(end4-end3))
# print("TIEMPO 5 {}".format(end5-end4))
# print("TIEMPO 6 {}".format(end6-end5))
# print("TIEMPO 7 {}".format(end7-end6))
# print("TIEMPO 8 {}".format(end8-end7))
# print("TIEMPO 9 {}".format(end9-end8))

# IDs, names, min_bnd, max_bnd = opt.get_moea_variables()
# print(IDs, names, min_bnd, max_bnd)

# names = opt.get_moea_objectives()
# print(names)

# information = {}
# information["name"] = ["string", "input"]
# information["pt"] = ["v_string", ["week"]]
# information["reference"] = ["string", "H2"]
# information["name_node"] = ["string", "week"]
# information["resource"] = ["string", "energy"]
# information["problem"] = ["string", "BT"]
# information["value"] = ["double", 1e10]

# opt.update_parameter(information=information)
# opt.run()
# opt.save_outputs(sim_no=2)

# import h5py
# a_file = h5py.File(os.path.join(os.path.dirname(path), "outputs1.h5"), "r")
# print(a_file.keys())
# print(a_file['Simulation_00000'].keys())
# dataset = np.array(a_file['Simulation_00000']['Pd_week'])
# print(dataset)
# # print("")
# # dataset = np.array(a_file['Simulation_00000']['voltage angle_week|weekend'])
# # print(dataset)
# print("")
# dataset = np.array(a_file['Simulation_00000']['surplus'])
# print(dataset)
# a_file.close()