from pyene.engines.main import pyeneClass
import os
import numpy as np

import time
begin = time.time()

# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\Pyene format proposal v1.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\118bus.xlsx'
# path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\Input Data - Myanmar_v1.xlsx'
# path = "C:\\Users\\f09903jm\\Downloads pc\\EAPP_pyene.xlsx"
# path = "C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\EAPP\\EAPP Data\\EAPP\\EAPP_1h.xlsx"

path = 'C:\\Users\\f09903jm\\OneDrive - The University of Manchester\\Pyene\\Inputs\\Pyene format proposal v1.xlsx'

opt = pyeneClass()
opt.initialise(path=path)
opt.save_outputs(sim_no=0)
end1 = time.time()


opt.run()

end2 = time.time()

print("TIEMPO 1 {}".format(begin-end1))
print("TIEMPO 2 {}".format(end1-end2))


# IDs, names, min_bnd, max_bnd = opt.get_moea_variables()
# print(IDs, names, min_bnd, max_bnd)

# names = opt.get_moea_objectives()
# print(names)

# information = {}
# information["name"] = ["string", "input"]
# information["pt"] = ["v_string", ["week"]]
# information["reference"] = ["string", "H1"]
# information["name_node"] = ["string", "week"]
# information["resource"] = ["string", "energy"]
# information["problem"] = ["string", "BT"]
# information["value"] = ["double", 1e10]

# opt.update_parameter(information=information)
# opt.run()
# opt.save_outputs(sim_no=1)

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