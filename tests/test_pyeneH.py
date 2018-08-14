""" Test the pyeneE engine. """
from fixtures import testConfig
from pyene.engines.pyene import pyeneClass as pe


def test_pyeneH_Time():
    ''' Check time dependence in default network '''
    print('test_pyeneH_Time')
    conf = testConfig()
    EN = pe(conf.EN)
    conf.HM.settings['Flag'] = True
    (HM, HModel, results) = EN.HSim(conf)
    HM.print(HModel)

    assert 0.0001 >= abs(HModel.OF.expr()-6766.6666) and \
        0.0001 >= abs(HModel.vHout[0, 0].value-66.6680) and \
        0.0001 >= abs(HModel.vHout[0, 1].value-305.5747) and \
        0.0001 >= abs(HModel.vHout[0, 2].value-171.7524) and \
        0.0001 >= abs(HModel.vHout[1, 2].value-66.6668) and \
        0.0001 >= abs(HModel.vHout[1, 3].value-236.2205) and \
        0.0001 >= abs(HModel.vHout[1, 4].value-103.9565) and \
        0.0001 >= abs(HModel.vHout[2, 2].value-100.0001) and \
        0.0001 >= abs(HModel.vHout[2, 3].value-354.3308) and \
        0.0001 >= abs(HModel.vHout[2, 4].value-155.9348)


def test_pyeneH_Scenarios():
    ''' Check multiple scenario analysis '''
    print('test_pyeneH_Scenarios')
    conf = testConfig()
    conf.HM.settings['Flag'] = True
    conf.HM.settings['NoTime'] = 5  # Five periods
    conf.HM.connections['Number'] = 2  # Two scenarios
    # Ading additional water in both scenarios
    conf.HM.settings['In'] = [[0, 1, 600], [3, 3, 800]]
    # Connecting the reservoirs at the beginning and end of each scenario
    conf.HM.connections['LinksF'] = [[0, 0], [1, 0]]
    conf.HM.connections['LinksT'] = [[0, 1], [1, 1]]
    EN = pe(conf.EN)
    (HM, HModel, results) = EN.HSim(conf)
    HM.print(HModel)
    print('OF: ', HModel.OF.expr())

    assert 0.0001 >= abs(HModel.OF.expr()-3500) and \
        0.0001 >= abs(HModel.vHout[0, 1].value-325.0060) and \
        0.0001 >= abs(HModel.vHout[0, 2].value-183.6914) and \
        0.0001 >= abs(HModel.vHout[4, 3].value-239.9525) and \
        0.0001 >= abs(HModel.vHout[4, 4].value-106.0274)


def test_pyeneH_ScenParts():
    ''' Check multiple scenario analysis '''
    print('test_pyeneH_ScenParts')
    conf = testConfig()
    conf.HM.settings['Flag'] = True
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
    HM.print(HModel)
    print('OF: ', HModel.OF.expr())

    assert 0.0001 >= abs(HModel.OF.expr()-3500) and \
        0.0001 >= abs(HModel.vHout[0, 1].value-383.7028) and \
        0.0001 >= abs(HModel.vHout[0, 2].value-124.6046) and \
        0.0001 >= abs(HModel.vHout[4, 3].value-259.6517) and \
        0.0001 >= abs(HModel.vHout[4, 4].value-83.0653)
