""" Test the pyeneE engine. """
from fixtures import *
from pyene.engines.pyene import pyeneClass as pe
import os


# Single vector energy balance test
def test_pyeneE_TreeMonth01(conf):
    print('test_pyeneE_TreeMonth01: TreeMonth01.json')
    conf.EM.settings['File'] = os.path.join(os.path.dirname(__file__), '..',
                                            'tests', 'json',
                                            'TreeMonth01.json')
    EN = pe()

    (EM, EModel, results) = EN.ESim(conf)
    EM.print(EModel)

    # 1000 - 4.25*(5*1 + 2*2) = 961.75
    assert EModel.vSoC[1, 1, 0].value == 961.75


# Multiple vector test
def test_pyeneE_TreeYear02(conf):
    print('test_pyeneE_TreeYear02: TreeYear02.json')
    conf.EM.settings['File'] = os.path.join(os.path.dirname(__file__), '..',
                                            'tests', 'json', 'TreeYear02.json')
    EN = pe()

    (EM, EModel, results) = EN.ESim(conf)
    EM.print(EModel)

    # 1900-30*(5*3+2*6)+22*(3-(5*3+2*6)) = 562
    # 562+1000-30*(5*3+2*6)+22*(3-(5*3+2*6)) = 224
    # 800-30*(5*1+2*4)+22*(1-(5*1+2*4)) = 146
    # 146+1700-30*(5*1+2*4)+22*(1-(5*1+2*4)) = 1192
    assert (EModel.vSoC[1, 1, 0].value == 562 and
            EModel.vSoC[2, 1, 0].value == 224 and
            EModel.vSoC[1, 1, 1].value == 146 and
            EModel.vSoC[2, 1, 1].value == 1192)


# Consideration of uncertainty
def test_pyeneE_Uncertainty(conf):
    print('test_pyeneE_Uncertainty: TreeMonth01Unc.json')
    conf.EM.settings['File'] = os.path.join(os.path.dirname(__file__), '..',
                                            'tests', 'json',
                                            'TreeMonth01Unc.json')
    EN = pe(conf.EN)

    (EM, EModel, results) = EN.ESim(conf)
    EM.print(EModel)

    # 10-5*.71 = 6.45
    # 10-8*.29 = 7.68
    # 10-0.2*(5*.71+8*.29) = 8.826
    # 10-0.3*(5*.71+8*.29) = 8.239
    # 20 -0.2*(5*.71+8*.29)-0.2*(5*.71+8*.29)-0.3*(5*.71+8*.29) -
    # 0.3*(5*.71+8*.29) = 4.13
    assert (EModel.vSoC[1, 1, 0].value == 4.130 and
            EModel.vSoC[2, 1, 0].value == 8.826 and
            EModel.vSoC[3, 1, 0].value == 8.239 and
            EModel.vSoC[6, 1, 0].value == 6.450 and
            EModel.vSoC[7, 1, 0].value == 7.680)
