# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:23:14 2018

Pyene Interfaces provides methods for exchanging information and models with
external tools, such as pypsa and pypower

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import numpy as np
try:
    import pypsa
except ImportError:
    print('pypsa has not been installed - functionalities unavailable')


class EInterfaceClass:
    def pyene2pypsa(self, NM, xscen):
        '''Convert pyene files to pypsa format'''
        # Create pypsa network
        try:
            nu = pypsa.Network()
        except ImportError:
            return (0, False)

        nu.set_snapshots(range(NM.settings['NoTime']))
        baseMVA = NM.networkE.graph['baseMVA']

        # Names
        auxtxtN = 'Bus'
        auxtxtG = 'Gen'
        auxtxtLd = 'Load'

        '''                             NETWORK
        name - network name
        snapshots - list of snapshots or time steps
        snapshot_weightings - weights to control snapshots length
        now - current snapshot
        srid - spatial reference
        '''

        '''                               BUS
        Missing attributes:
            type - placeholder (not yet implemented in pypsa)
            carrier - 'AC' or 'DC'

        Implemented attributes:
            Name auxtxtN+str(xn)
            v_nom - Nominal voltage
            v_mag_pu_set - per unit voltage set point
            v_mag_pu_min - per unit minimum voltage
            v_mag_pu_max - per unit maximum voltage
            auxtxtN = 'Bus'  # Generic name for the nodes
            x - coordinates
            y - coordinates
        '''
        PVBus = np.zeros(NM.networkE.number_of_nodes(), dtype=float)
        aux = (NM.generationE['Number']-NM.hydropower['Number'] -
               NM.RES['Number'])
        for xn in NM.generationE['Data']['GEN_BUS'][0:aux]:
            if NM.networkE.node[xn]['BUS_TYPE'] == 2:
                PVBus[xn-1] = NM.generationE['Data']['VG'][xn]
        for xn in NM.networkE.node:
            if NM.networkE.node[xn]['BASE_KV'] == 0:
                aux1 = 1
            else:
                aux1 = NM.networkE.node[xn]['BASE_KV']
            if NM.networkE.node[xn]['BUS_TYPE'] == 2 or \
               NM.networkE.node[xn]['BUS_TYPE'] == 3:
                aux2 = NM.networkE.node[xn]['VM']
            else:
                aux2 = PVBus[xn-1]
            if 'BUS_X' in NM.networkE.node[xn]:
                nu.add('Bus', auxtxtN+str(xn),
                       v_nom=aux1, v_mag_pu_set=aux2,
                       v_mag_pu_min=NM.networkE.node[xn]['VMIN'],
                       v_mag_pu_max=NM.networkE.node[xn]['VMAX'],
                       x=NM.networkE.node[xn]['BUS_X'],
                       y=NM.networkE.node[xn]['BUS_Y'])
            else:
                nu.add('Bus', auxtxtN+str(xn), v_nom=aux1, v_mag_pu_set=aux2,
                       v_mag_pu_min=NM.networkE.node[xn]['VMIN'],
                       v_mag_pu_max=NM.networkE.node[xn]['VMAX'])

        '''                            GENERATOR
        Missing attributes:
            type - placeholder for generator type
            p_nom_extendable - boolean switch to increase capacity
            p_nom_min - minimum value for extendable capacity
            p_nom_max - maximum value for extendable capacity
            sign - power sign
            carrier - required for global constraints
            capital_cost - cost of extending p_nom by 1 MW
            efficiency - required for global constraints
            committable - boolean (only if p_nom is not extendable)
            start_up_cost - only if commitable is true
            shut_down_cost - only if commitable is true
            min_up_time - only if commitable is true
            min_down_time - only if commitable is true
            initial_status  - only if commitable is true
            ramp_limit_up - only if commitable is true
            ramp_limit_down - only if commitable is true
            ramp_limit_start_up - only if commitable is true
            ramp_limit_shut_down - only if commitable is true

        Implemented attributes:
            name - generator name
            bus - bus name
            control - 'PQ', 'PV' or 'Slack'
            p_min_pu - use default 0 for modelling renewables
            p_max_pu - multpier for intermittent maximum generation
            marginal_cost - linear model
        '''
        # Fuel based generation
        aux = (NM.generationE['Number']-NM.hydropower['Number'] -
               NM.RES['Number'])
        xg = -1
        for xn in NM.generationE['Data']['GEN_BUS'][0:aux]:
            xg += 1
            if NM.networkE.node[xn]['BUS_TYPE'] == 1:
                aux1 = 'PQ'
            elif NM.networkE.node[xn]['BUS_TYPE'] == 2:
                aux1 = 'PV'
            else:
                aux1 = 'Slack'
            aux2 = (NM.generationE['Data']['PMAX'][xg] +
                    NM.generationE['Data']['PMIN'][xg])/2*baseMVA
            if NM.generationE['Costs']['MODEL'][xg] == 1:
                xi = 2
                while NM.generationE['Costs']['COST'][xg][xi] <= aux2:
                    xi += 2
                aux3 = ((NM.generationE['Costs']['COST'][xg][xi+1] -
                         NM.generationE['Costs']['COST'][xg][xi-1]) /
                        (NM.generationE['Costs']['COST'][xg][xi] -
                         NM.generationE['Costs']['COST'][xg][xi-2]))
            else:
                aux3 = 0
                for xi in range(NM.generationE['Costs']['NCOST'][xg]-1):
                    aux3 += ((NM.generationE['Costs']['NCOST'][xg]-xi-1) *
                             NM.generationE['Costs']['COST'][xg][xi] *
                             aux2**(NM.generationE['Costs']['NCOST']
                                    [xg]-xi-2))
            nu.add('Generator', auxtxtG+str(xg+1),
                   bus=auxtxtN+str(xn),
                   control=aux1,
                   p_nom_max=NM.generationE['Data']['PMAX'][xg]*baseMVA,
                   p_set=NM.generationE['Data']['PG'][xg],
                   q_set=NM.generationE['Data']['QG'][xg],
                   marginal_cost=aux3
                   )

        # Renewable generation
        aux = NM.generationE['Number']-NM.RES['Number']
        aux1 = 'PQ'
        xg = aux-1
        xr = -1
        yres = np.zeros(NM.settings['NoTime'], dtype=float)
        for xn in (NM.generationE['Data']['GEN_BUS']
                   [aux:NM.generationE['Number']]):
            xg += 1
            xr += 1
            for xt in range(NM.settings['NoTime']):
                yres[xt] = (NM.scenarios['RES']
                            [NM.resScenario[xr][xscen][1]+xt])
            nu.add('Generator', auxtxtG+str(xg+1),
                   bus=auxtxtN+str(xn),
                   control=aux1,
                   p_nom_max=yres,
                   p_nom=NM.RES['Max'][xr],
                   p_set=NM.generationE['Data']['PG'][xg],
                   q_set=NM.generationE['Data']['QG'][xg],
                   marginal_cost=NM.generationE['Costs']['COST'][xg][0]
                   )

        '''                             CARRIER
        name - Energy carrier
        co2_emissions - tonnes/MWh
        carrier_attribute
        '''

        '''                         GLOBAL CONSTRAINT
        name - constraint
        type - only 'primary_energy' is supported
        carrier_attribute - attributes such as co2_emissions
        sense - either '<=', '==' or '>='
        constant - constant for right hand-side of constraint
        '''

        #                           STORAGE UNIT
        #                              STORE
        '''                              LOAD
        Missing attributes:
            type - placeholder for type
            sign - change to -1 for generator

        Implemented attributes:
            name - auxtxtLd+str(xL)
            bus - Name of bus
        '''
        xL = 0
        ydemP = np.zeros(NM.settings['NoTime'], dtype=float)
        ydemQ = np.zeros(NM.settings['NoTime'], dtype=float)
        for xn in NM.networkE.node:
            if NM.demandE['PD'][xn-1] != 0:
                xL += 1
                for xt in range(NM.settings['NoTime']):
                    aux = (NM.scenarios['Demand']
                           [NM.busScenario[xn-1][xscen]+xt])
                    ydemP[xt] = NM.demandE['PD'][xn-1]*aux
                    ydemQ[xt] = NM.demandE['QD'][xn-1]*aux
                nu.add('Load', auxtxtLd+str(xL),
                       bus=auxtxtN+str(xn),
                       p_set=ydemP,
                       q_set=ydemQ
                       )

        #                         SHUNT IMPEDANCE

        '''                              LINE
        Missing attributes:
            type - assign string to re-calculate line parameters
            g - shunt conductivity
            s_nom_extendable - boolean switch to allow s_nom to be extended
            s_nom_min - minimum capacity for extendable s_nom
            s_nom_max - maximum capacity for extendable s_nom
            s_max_pu - multiplier (time series) for considering DLR
            capital_cost - cost for extending s_nom by 1
            length - line length
            terrain_factor - terrain factor for increasing capital costs
            num_parallel - number of lines in parallel
            v_ang_min - placeholder for minimum voltage angle drop/increase
            v_and_max - placeholder for maximum voltage angle drop/increase
            b - shunt susceptance

        Implemented attributes:
            Name - auxtxtL+str(xb)
            bus0 - One of the buses connected to the line
            bus1 - The other bus connected to the line
            x - series reactance
            r - series resistance
            snom - MVA limit
        '''
        auxtxtL = 'Line'
        xb = 0
        for (xf, xt) in NM.networkE.edges:
            if NM.networkE[xf][xt]['TAP'] == 0:
                xb += 1
                auxpu = (nu.buses['v_nom']['Bus{}'.format(xf)]**2 /
                         NM.networkE.graph['baseMVA'])
                nu.add('Line', auxtxtL+str(xb),
                       bus0=auxtxtN+str(xf),
                       bus1=auxtxtN+str(xt),
                       x=NM.networkE[xf][xt]['BR_X']*auxpu,
                       r=NM.networkE[xf][xt]['BR_R']*auxpu,
                       s_nom=NM.networkE[xf][xt]['RATE_A']
                       )

        #                           LINE TYPES
        '''                           TRANSFORMER
        Missing attributes:
            type - assign string to re-calculate line parameters
            g - shunt conductivity
            s_nom_extendable - boolean switch to allow s_nom to be extended
            s_nom_min - minimum capacity for extendable s_nom
            s_nom_max - maximum capacity for extendable s_nom
            s_max_pu - multiplier (time series) for considering DLR
            capital_cost - cost for extending s_nom by 1
            num_parallel - number of lines in parallel
            tap_position - if a type is defined, determine tap position
            phase_shift - voltage phase angle
            v_ang_min - placeholder for minimum voltage angle drop/increase
            v_and_max - placeholder for maximum voltage angle drop/increase

        Implemented attributes:
            Name - auxtxtT+str(xb)
            bus0 - One of the buses connected to the line
            bus1 - The other bus connected to the line
            model - Matpower and pypower use pi admittance models
            tap_ratio - transformer ratio
            tap_side - taps at from bus in Matlab
        '''
        auxtxtT = 'Trs'
        xb = 0
        for (xf, xt) in NM.networkE.edges:
            if NM.networkE[xf][xt]['TAP'] != 0:
                xb += 1
                nu.add('Transformer', auxtxtT+str(xb),
                       bus0=auxtxtN+str(xf),
                       bus1=auxtxtN+str(xt),
                       model='pi',
                       x=NM.networkE[xf][xt]['BR_X'],
                       r=NM.networkE[xf][xt]['BR_R'],
                       b=NM.networkE[xf][xt]['BR_B'],
                       s_nom=NM.networkE[xf][xt]['RATE_A'],
                       tap_ratio=NM.networkE[xf][xt]['TAP'],
                       tap_side=0
                       )

        #                        TRANSFORMER TYPES
        #                              LINK

        return (nu, True)

    def pypsa2pyene(self):
        '''Convert pyene files to pypsa format'''
        print('To be finalised pypsa2pyene')

    def pyene2pypower(self):
        '''Convert pyene files to pypsa format'''
        print('To be finalised pyene2pypower')

    def pypower2pyene(self):
        '''Convert pyene files to pypsa format'''
        print('To be finalised pypower2pyene')
