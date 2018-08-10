""" Test the pyeneE engine. """
from fixtures import testConfig
from pyene.engines.pyene import pyeneClass as pe


def test_pyeneH_Time():
    ''' Check time dependence in default network '''
    print('test_pyeneH_Time')
    conf = testConfig()
    EN = pe(conf.EN)
    (HM, HModel, results) = EN.HSim(conf)
    HM.print_outputs(HModel)

    assert 0.0001 >= abs(HModel.OF.expr()-6766.6666) and \
        0.0001 >= abs(HModel.vHout[0, 0].value-66.6666) and \
        0.0001 >= abs(HModel.vHout[0, 1].value-438.2045) and \
        0.0001 >= abs(HModel.vHout[0, 2].value-155.0717) and \
        0.0001 >= abs(HModel.vHout[1, 2].value-66.6666) and \
        0.0001 >= abs(HModel.vHout[1, 3].value-278.1102) and \
        0.0001 >= abs(HModel.vHout[1, 4].value-91.8225) and \
        0.0001 >= abs(HModel.vHout[2, 2].value-100.0000) and \
        0.0001 >= abs(HModel.vHout[2, 3].value-417.1653) and \
        0.0001 >= abs(HModel.vHout[2, 4].value-137.7337)


def test_pyeneH_Scenarios():
    ''' Check multiple scenario analysis '''
    print('test_pyeneH_Scenarios')
    conf = testConfig()
    conf.HM.settings['NoTime'] = 5  # Five periods
    conf.HM.connections['Number'] = 2  # Two scenarios
    # Ading additional water in both scenarios
    conf.HM.settings['In'] = [[0, 1, 600], [3, 3, 800]]
    # Connecting the reservoirs at the beginning and end of each scenario
    conf.HM.connections['LinksF'] = [[0, 0], [1, 0]]
    conf.HM.connections['LinksT'] = [[0, 1], [1, 1]]
    EN = pe(conf.EN)
    (HM, HModel, results) = EN.HSim(conf)
    HM.print_outputs(HModel)
    print('OF: ', HModel.OF.expr())

    assert 0.0001 >= abs(HModel.OF.expr()-3500) and \
        0.0001 >= abs(HModel.vHout[0, 1].value-441.8511) and \
        0.0001 >= abs(HModel.vHout[0, 2].value-156.6638) and \
        0.0001 >= abs(HModel.vHout[4, 3].value-278.7574) and \
        0.0001 >= abs(HModel.vHout[4, 4].value-92.0810)


def test_pyeneH_ScenParts():
    ''' Check multiple scenario analysis '''
    print('test_pyeneH_ScenParts')
    conf = testConfig()
    conf.HM.settings['NoTime'] = 5  # Five periods
    conf.HM.connections['Number'] = 2  # Two scenarios
    # Ading additional water in both scenarios
    conf.HM.settings['In'] = [[0, 1, 600], [3, 3, 800]]
    # Connecting the reservoirs at the beginning and end of each scenario
    conf.HM.connections['LinksF'] = [[0, 0], [1, 0]]
    conf.HM.connections['LinksT'] = [[0, 1], [1, 1]]
    conf.HM.rivers['Parts'] = [5]  # Model each river using several parts
    EN = pe(conf.EN)
    (HM, HModel, results) = EN.HSim(conf)
    HM.print_outputs(HModel)
    print('OF: ', HModel.OF.expr())

    assert 0.0001 >= abs(HModel.OF.expr()-3500) and \
        0.0001 >= abs(HModel.vHout[0, 1].value-452.0168) and \
        0.0001 >= abs(HModel.vHout[0, 2].value-146.9597) and \
        0.0001 >= abs(HModel.vHout[4, 3].value-281.6114) and \
        0.0001 >= abs(HModel.vHout[4, 4].value-89.0123)
