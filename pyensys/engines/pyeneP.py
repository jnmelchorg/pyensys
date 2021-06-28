# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:23:14 2018

Pyene for Printing outputs

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import numpy as np

class PrintClass:
    def __init__(self, m, EN):
        ''' Initialize for pyomo, pypsa or glpk class '''
        
        self.data = {
                'GenerationP': None,
                'GenerationQ': None,
                'GenerationUC': None,
                'GenerationCost': None,
                'Line_FlowP0': None,
                'Line_FlowP1': None,
                'Line_FlowQ0': None,
                'Line_FlowQ1': None,
                'Line_LossP': None,
                'Line_LossQ': None,
                'Voltage_pu': None,
                'Voltage_ang': None,
                'Pumps': None,
                'Curtailment': None,
                'Services': None,
                'OF': None
                }
        
        from pyomo.core import ConcreteModel
        import pypsa

        if isinstance(m, ConcreteModel):
            self.type = 1
        elif isinstance(m[0], pypsa.Network):
            self.type = 2
        else:
            print('Model not recognised')

        # Initialize network variables
        from .main import pyeneClass
        from .pyeneN import ENetworkClass
        if isinstance(EN, pyeneClass):
            OFaux = EN._Calculate_OFaux()
            self.initializeNM(m, EN.NM, OFaux)
        elif isinstance(EN, ENetworkClass):
            OFaux = np.ones(len(EN.connections['set']),dtype=float)
            self.initializeNM(m, EN, OFaux)

    def initializeNM(self, m, NM, OFaux):
        Noh = len(NM.connections['set'])
        self.data['GenerationP'] = \
            np.zeros((Noh*NM.Gen.get_NoGen(), NM.settings['NoTime']),
                     dtype=float)
        self.data['GenerationQ'] = \
            np.zeros((Noh*NM.Gen.get_NoGen(), NM.settings['NoTime']),
                     dtype=float)
        self.data['GenerationCost'] = \
            np.zeros((Noh*NM.Gen.get_NoGen(), NM.settings['NoTime']),
                     dtype=float)
        self.data['GenerationUC'] = \
            np.ones((Noh*NM.Gen.get_NoGen(), NM.settings['NoTime']),
                     dtype=bool)
        self.data['Voltage_pu'] = \
            np.ones((Noh*NM.NoBuses, NM.settings['NoTime']),
                     dtype=float)
        self.data['Voltage_ang'] = \
            np.ones((Noh*NM.NoBuses, NM.settings['NoTime']),
                     dtype=float)
        self.data['Pumps'] = np.zeros((Noh*(NM.pumps['Number']+1),
                                      NM.settings['NoTime']), dtype=float)
        self.data['Curtailment'] = \
            np.zeros((Noh*NM.NoFea, NM.settings['NoTime']), dtype=float)
        self.data['Services'] = np.zeros((Noh*NM.p['GServices'],
                                        NM.settings['NoTime']), dtype=float)
        self.data['Line_FlowP0'] = \
            np.zeros((Noh*NM.NoBranch, NM.settings['NoTime']), dtype=float)
        self.data['Line_FlowP1'] = \
            np.zeros((Noh*NM.NoBranch, NM.settings['NoTime']), dtype=float)
        self.data['Line_FlowQ0'] = \
            np.zeros((Noh*NM.NoBranch, NM.settings['NoTime']), dtype=float)
        self.data['Line_FlowQ1'] = \
            np.zeros((Noh*NM.NoBranch, NM.settings['NoTime']), dtype=float)
        self.data['Line_LossP'] = \
            np.zeros((Noh*NM.connections['Branches'], NM.settings['NoTime']),
                     dtype=float)
        self.data['Line_LossQ'] = \
            np.zeros((Noh*NM.connections['Branches'], NM.settings['NoTime']),
                     dtype=float)

        if self.type == 1:
            # Pyomo - DC model
            for x1 in range(Noh*NM.Gen.get_NoGen()):
                for x2 in range(NM.settings['NoTime']):
                     self.data['GenerationP'][x1][x2] = \
                            m.vNGen[x1, x2].value*NM.ENetwork.get_Base()

            # UC
            if 'vNGen_Bin' in m.__dir__():
                for x1 in range(Noh*NM.Gen.get_NoGen()):
                    for x2 in range(NM.settings['NoTime']):
                        self.data['GenerationUC'][x1][x2] = \
                            m.vNGen_Bin[x1, x2].value

            # Voltage angle
            if 'vNVolt' in m.__dir__():
                for x1 in range(Noh*NM.NoBuses):
                    for x2 in range(NM.settings['NoTime']):
                        self.data['Voltage_ang'][x1][x2] = \
                                m.vNVolt[x1, x2].value

            # Pumps
            if 'vNPump' in m.__dir__():
                for x1 in range(Noh*(NM.pumps['Number']+1)):
                    for x2 in range(NM.settings['NoTime']):
                        self.data['Pumps'][x1][x2] = \
                                m.vNPump[x1, x2].value*NM.ENetwork.get_Base()

            # Dummy genertors / curtailment / feasibility constraints
            if 'vNFea' in m.__dir__():
                for x1 in range(Noh*NM.NoFea):
                    for x2 in range(NM.settings['NoTime']):
                        self.data['Curtailment'][x1][x2] = \
                                m.vNFea[x1, x2].value*NM.ENetwork.get_Base()

            # Services
            if 'vNServ' in m.__dir__():
                for x1 in range(Noh*NM.p['GServices']):
                    for x2 in range(NM.settings['NoTime']):
                        self.data['Services'][x1][x2] = \
                                m.vNServ[x1, x2].value*NM.ENetwork.get_Base()

            # Lines
            if 'vNFlow' in m.__dir__():
                for x1 in range(Noh*NM.NoBranch):
                    for x2 in range(NM.settings['NoTime']):
                        self.data['Line_FlowP0'][x1][x2] = \
                            m.vNFlow[x1, x2].value*NM.ENetwork.get_Base()

            # Losses
            if 'vNLoss' in m.__dir__():
                for x1 in range(Noh*NM.connections['Branches']):
                    for x2 in range(NM.settings['NoTime']):
                        self.data['Line_LossP'][x1][x2] = \
                        m.vNLoss[x1, x2].value*NM.ENetwork.get_Base()

            # OF
            self.data['OF'] = m.OF.expr()

        elif self.type ==2:
            # pypsa - AC model
            for xh in range(Noh):
                # Generators
                for xg in range(NM.Gen.get_NoGen()):
                    x1 = NM.connections['Generation'][xh]+xg
                    self.data['GenerationP'][x1][:] = \
                        getattr(m[xh].generators_t.p, 'Gen{}'.format(xg+1))
                    self.data['GenerationQ'][x1][:] = \
                        getattr(m[xh].generators_t.q, 'Gen{}'.format(xg+1))

                # Generation costs
                NM.s['Gen'] = range(NM.Gen.get_NoGen()) # xg
                range(NM.Gen.get_NoPieces()) # xc
                NM.s['Tim'] = range(NM.settings['NoTime']) # xt
                NM.s['Con'] = NM.connections['set'] # xh
                ConC = NM.connections['Cost'][xh]
                ConG = NM.connections['Generation'][xh]
                for xg in range(NM.Gen.get_NoGen()):
                    for xc in range(NM.Gen.get_NoPieces()):
                        (flg, x1, x2, M1, M2) = \
                        NM.Gen.cNEGenC_Auxrule(xg, xc, ConC, ConG)
                        if flg:
                            for xt in range(NM.settings['NoTime']):
                                w = NM.scenarios['Weights'][xt]
                                self.data['GenerationCost'][x1][xt] = \
                                    max(self.data['GenerationCost'][x1][xt],
                                        self.data['GenerationQ'][x1][xt] *
                                        M1*w+M2)
                # Buses
                for xn in range(NM.ENetwork.get_NoBus()):
                    x1 = NM.connections['Voltage'][xh]+xn
                    self.data['Voltage_pu'][x1][:] = \
                        getattr(m[xh].buses_t.v_mag_pu, 'Bus{}'.format(xn+1))
                    self.data['Voltage_ang'][x1][:] = \
                        getattr(m[xh].buses_t.v_ang, 'Bus{}'.format(xn+1))

                # Lines
                xb = 0
                for ob in NM.ENetwork.Branch:
                    # Is this a line?
                    if ob.get_Tap() == 0:
                        aux = 'Line{}'.format(ob.get_Number())
                        P0 = getattr(m[xh].lines_t.p0, aux)
                        P1 = getattr(m[xh].lines_t.p1, aux)
                        Q0 = getattr(m[xh].lines_t.q0, aux)
                        Q1 = getattr(m[xh].lines_t.q1, aux)
                    else:  # Transformer
                        aux = 'Trs{}'.format(ob.get_Number())
                        P0 = getattr(m[xh].transformers_t.p0, aux)
                        P1 = getattr(m[xh].transformers_t.p1, aux)
                        Q0 = getattr(m[xh].transformers_t.q0, aux)
                        Q1 = getattr(m[xh].transformers_t.q1, aux)

                    x1 = NM.connections['Flow'][xh]+xb
                    self.data['Line_FlowP0'][x1][:] = P0
                    self.data['Line_FlowP1'][x1][:] = P1
                    self.data['Line_FlowQ0'][x1][:] = Q0
                    self.data['Line_FlowQ1'][x1][:] = Q1
                    self.data['Line_LossP'][x1][:] = P0+P1
                    self.data['Line_LossQ'][x1][:] = Q0+Q1
                    xb += 1

            # OF
            self.data['OF'] = 0
            for xh in range(Noh):
                for xg in range(NM.Gen.get_NoGen()):
                    x1 = NM.connections['Generation'][xh]+xg
                    self.data['OF'] += self.data['GenerationCost'][x1][xt] * \
                        OFaux[xh]

    def get_Curtailment(self, x1, x2):
        return self.data['Curtailment'][x1][x2]

    def get_GenerationP(self, x1, x2):
        return self.data['GenerationP'][x1][x2]

    def get_GenerationQ(self, x1, x2):
        return self.data['GenerationQ'][x1][x2]

    def get_GenerationUC(self, x1, x2):
        return self.data['GenerationUC'][x1][x2]

    def get_Line_FlowP0(self, x1, x2):
        return self.data['Line_FlowP0'][x1][x2]

    def get_Line_FlowP1(self, x1, x2):
        return self.data['Line_FlowP1'][x1][x2]

    def get_Line_FlowQ0(self, x1, x2):
        return self.data['Line_FlowQ0'][x1][x2]

    def get_Line_FlowQ1(self, x1, x2):
        return self.data['Line_FlowQ1'][x1][x2]

    def get_Line_LossP(self, x1, x2):
        return self.data['Line_LossP'][x1][x2]

    def get_Line_LossQ(self, x1, x2):
        return self.data['Line_LossQ'][x1][x2]

    def get_OF(self):
        return self.data['OF']

    def get_Pumps(self, x1, x2):
        return self.data['Pumps'][x1][x2]

    def get_Services(self, x1, x2):
        return self.data['Services'][x1][x2]

    def get_Voltage_pu(self, x1, x2):
        return self.data['Voltage_pu'][x1][x2]

    def get_Voltage_ang(self, x1, x2):
        return self.data['Voltage_ang'][x1][x2]

