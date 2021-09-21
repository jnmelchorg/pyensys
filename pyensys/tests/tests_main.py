from pyensys.engines.main import pyeneClass
from os.path import join, dirname
import time

begin = time.time()

path = dirname(__file__)
path = join(path, "excel")
path = join(path, "excel_format_proposal.xlsx")

opt = pyeneClass()
opt.initialise(path=path)

end1 = time.time()

subscripts = {}
subscripts["pt"] = ["v_string", ["week", "weekday"]]
subscripts["hour"] = ["double", 0.0]

opt.run(subscripts=subscripts)
opt.save_outputs(sim_no=0)

end2 = time.time()

parameter = {}
parameter["name"] = ["string", "active power max limit"]
parameter["pt"] = ["v_string", ["week", "weekday"]]
parameter["hour"] = ["double", 0.0]
parameter["value"] = ["double", 200.0]
parameter["ID"] = ["string", "PV1"]
parameter["problem"] = ["string", "DC OPF"]


opt.update_parameter(information=parameter)
opt.run(subscripts=subscripts)
opt.save_outputs(sim_no=1)

end3 = time.time()

print("TIEMPO 1 {}".format(-begin+end1))
print("TIEMPO 2 {}".format(-end1+end2))
print("TIEMPO 3 {}".format(-end2+end3))

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