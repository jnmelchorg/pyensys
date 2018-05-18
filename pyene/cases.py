""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass as pe


# Energy balance test
def test_pyeneE(config):
    """ Execute pyene to access pyeneE - Full json based simulation."""
    EN = pe()
    # Avoid loading file
    if config.init:
        EN.fRea = False
    (EM, EModel, results) = EN.ESim(config.TreeFile)
    EM.print(EModel)


# Network simulation test
def test_pyeneN(config):
    """ Execute pyene to access pyeneN - Full json based simulation."""
    EN = pe()
    # Avoid loading file
    if config.init:
        EN.fRea = False
    (NM, NModel, results) = EN.NSim(config.NetworkFile)
    NM.print(NModel)


# pyene simulation test
def test_pyene(conf):
    """ Execute pyene to access pyeneN - Full json based simulation."""

    # Create object
    EN = pe()

    # Initialise with selected configuration
    EN.initialise(conf)

    # Fake weather engine
    FileName = 'TimeSeries.json'
    (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
     LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
     Nohr) = EN.ReadTimeS(FileName)

    # Single demand node
    demandNode = {
            'type': 'Demand',
            'value': DemandProfiles[0][:],
            'link': BusDem
            }
    EN.loadDemand(demandNode)

    # Several RES nodes
    for xr in range(conf.NoRES):
        resNode = {
                'type': 'RES',
                'value': RESProfs[xr][:],
                'link': xr+1
                }
        EN.loadRES(resNode, conf.NoRES, conf.Time)

    # Several hydro nodes
    for xh in range(conf.NoHydro):
        hydNode = {
                'type': 'Hydro',
                'value': 1,
                'link': xh+1
                }
        EN.loadHydro(hydNode, conf.NoHydro)

    # Run integrated pyene
    mod = EN.run()

    # Collect output of pumps
    indexPump=1
    pumNode ={
            'type': 'Pump',
            'value': EN.getPump(mod, indexPump),
            'link': indexPump
            }
    

    # Print results
    EN.Print_ENSim(mod, EN.EM, EN.NM)
