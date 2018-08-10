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
