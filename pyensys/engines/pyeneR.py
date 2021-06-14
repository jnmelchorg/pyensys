# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:19:20 2018

Pyene Renewable energy sources provides methods to calculate electricity
generation from different energy resources, currently PV and wind

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import numpy as np
import math


class pyeneRConfig():
    ''' Overall default configuration '''
    def __init__(self):
        '''Inital assumptions'''

        # Solar pannels
        self.solar = {}
        self.solar['Latitude'] = 52  # Latitude
        self.solar['Albedo'] = 0.2  # Reflectance coefficient
        self.solar['day'] = [1]  # Day to simulate
        self.solar['hour'] = [1]
        # Collector type: 1:Fixed 2: Single axis tracking 3: two axis tracking
        self.solar['Collector'] = 1
        self.solar['face'] = 0  # Collector face angle
        self.solar['tilt'] = self.solar['Latitude']  # Collector tilt angle

        # Wind generators
        self.wind = {}
        self.wind['height'] = 10  # Height of wind measurements
        self.wind['alpha'] = 0.1429  # Friction coefficient
        self.wind['CutIN'] = 3  # Cut in wind speed (m/s)
        self.wind['Rated'] = 10  # Rated wind speed (m/s)
        self.wind['cutOUT'] = 24  # cut out wind speed (m/s)
        self.wind['Park'] = 0.95  # Wind park efficiency
        self.wind['Model'] = 1  # 1: Linear, 2: Parabolic, 3: Cubic
        self.wind['tower'] = 50  # Height of the turbines

        # Water for cooling
        self.cooling = {
                'Flag': False,  # Consider cooling
                'Types': ['Nuclear', 'Coal','CCGT'],
                'Gen2Temp': [],  # Link generators to temp_w profiles
                'GenType': [],  # Generation types
                'temp_w': [[13.3876, 12.6934, 12.2074, 12.1380, 12.1380,
                           12.4851, 12.7628, 13.6652, 14.9148, 15.8173,
                           16.5809, 17.1362, 17.2057, 17.6222, 17.4833,
                           17.4833, 17.2057, 16.7891, 16.3726, 15.8173,
                           14.8454, 14.2206, 13.1793, 13.2487]],
                'TempThres': [15.0, 15.0, 15.0],  # Temp not affect CF
                'TempShut': [32.0, 32.0, 32.0],  # Tempe that leads to shutdown
                'TempMax': [35.0, 35.0, 35.0],  # Max outlet temperature
                'Delta': [10.0, 10.0, 10.0],  # Max temp intake and outlet
                'Scarcity': 1,  # Ratio available/required
                'Lambda': [0.00444, 0.0097, 0.0097]  # Derating factor
                }
        


class RESprofiles():
    ''' Produce time series for different RES '''
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = pyeneRConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def buildCF(self):
        ''' Build parameters for generator's CF '''
        aux = len(self.cooling['Gen2Temp'])

            

    def buildPV(self, direct, diffuse):
        ''' Produce PV generation multiplier '''
        datalength = len(direct)
        PVMultiplier = np.zeros(datalength, dtype=float)
        # Changing to radians
        latitude = math.radians(self.solar['Latitude'])
        face = math.radians(self.solar['face'])
        tilt = math.radians(self.solar['tilt'])

        # Produce multipliers
        xt = 0
        for xd in self.solar['day']:
            for xh in self.solar['hour']:
                PVMultiplier[xt] = \
                    self.getPVGeneration(direct[xt], diffuse[xt], xd, xh,
                                         latitude, face, tilt)
                xt += 1

        return PVMultiplier

    def buildWind(self, val):
        ''' Produce wind generation multiplier '''
        datalength = len(val)
        windMultiplier = np.zeros(datalength, dtype=float)

        # Produce multipliers
        for xw in range(datalength):
            # Height correction
            ws = val[xw]*(self.wind['tower']/self.wind['height']) ** \
                self.wind['alpha']

            # The turbines do not generate for winds under the cut-in speed
            if ws > self.wind['CutIN']:
                # Produce at rated capacity
                if ws > self.wind['Rated']:
                    if ws <= self.wind['cutOUT']:
                        windMultiplier[xw] = self.wind['Park']
                else:
                    # Use function
                    windMultiplier[xw] = self.wind['Park'] * \
                        (ws**self.wind['Model'] -
                         self.wind['CutIN']**self.wind['Model']) / \
                        (self.wind['Rated']**self.wind['Model'] -
                         self.wind['CutIN']**self.wind['Model'])

        return windMultiplier

    def get_CF(self, **kwargs):
        ''' Calculate capacity factor '''

        if 'temp_w' in kwargs:
            temp_w = kwargs.pop('temp_w')
        else:
            temp_w = self.cooling['temp_w']

        if 'type' in kwargs:
            index = self.cooling['Types'].index(kwargs.pop('type'))
        elif 'index' in kwargs:
            index = kwargs.pop('index')
        else:
            index = self.cooling['Types'].index('Coal')

        # Select generator data
        temp_w_health = self.cooling['TempThres'][index]
        temp_w_shutdown = self.cooling['TempShut'][index]
        temp_w_max = self.cooling['TempMax'][index]
        delta_temp_w_max = self.cooling['Delta'][index]
        w_scarcity_factor = self.cooling['Scarcity']
        lambda_derating4temp = self.cooling['Lambda'][index]

        #  Temperature over which the CF would be affected severely
        temp_w_risk = temp_w_max-(delta_temp_w_max/w_scarcity_factor)
        
        #  calculate the derating factor relates to the cooling water scarcity
        lambda_derating4scarcity = \
            (1.0-lambda_derating4temp*(temp_w_max - delta_temp_w_max /
                                       w_scarcity_factor - temp_w_health)) / \
            (delta_temp_w_max/w_scarcity_factor)

        #  create the output parameter
        s = len(temp_w)
        capacity_factor = np.zeros(s, dtype=float)

        #  capacity factor is equal to 1.0, if temp_w < temp_w_health
        for x in range(s):
            if temp_w[x] <= temp_w_health:
                capacity_factor[x] = 1.0
            elif temp_w[x] < temp_w_shutdown:
                if temp_w[x] <= temp_w_risk:
                    capacity_factor[x] = 1.0-lambda_derating4temp * \
                        (temp_w[x]-temp_w_health)
                else:
                    capacity_factor[x] = lambda_derating4scarcity * \
                        (temp_w_max-temp_w[x])
            if capacity_factor[x] > 1:
                capacity_factor[x] = 1

        return capacity_factor

    def getPVAngles(self, xd, xh, latitude, face, tilt):
        ''' Calculate solar angles '''
        # Solar declination
        Delta = 0.409279*math.sin(2*math.pi/365*(xd-81))

        # Hour angle
        HourAngle = math.pi*(1-xh/12)

        # Altitude angle
        Beta = math.asin(math.cos(latitude)*math.cos(Delta) *
                         math.cos(HourAngle)+math.sin(latitude) *
                         math.sin(Delta))
        # Azimuth
        azimuth = math.asin(math.cos(Delta)*math.sin(HourAngle)/math.cos(Beta))
        if math.cos(HourAngle) < math.tan(Delta)/math.tan(latitude):
            azimuth = math.pi-azimuth

        # Cosine of incident angle between the sun and the collectro's face
        Theta = math.cos(Beta)*math.cos(azimuth-face)*math.sin(tilt) + \
            math.sin(Beta)*math.cos(tilt)
        if Theta < 0:
            Theta = 0
        if Beta < 0:
            Beta = 0

        return (Beta, Theta, Delta)

    def getPVGeneration(self, direct, diffuse, xd, xh, latitude, face, tilt):
        ''' Get power generation kW/m2 '''
        # Get solar angles
        [Beta, Theta, Delta] = self.getPVAngles(xd, xh, latitude, face, tilt)

        # Choose collector model
        if self.solar['Collector'] == 1:
            # Fixed collector
            aux1 = (1-math.cos(tilt))/2
            aux2 = Theta
        elif self.solar['Collector'] == 2:
            # One axis tracking
            aux1 = (1-math.cos(math.pi/2-Beta+Delta))/2
            aux2 = math.cos(Delta)
        elif self.solar['Collector'] == 3:
            # Two axis tracking
            aux1 = (1-math.cos(math.pi/2-Beta))/2
            aux2 = 1

        PV = aux1*(diffuse+self.solar['Albedo']*(diffuse+direct))+aux2*direct
        return PV/1000

    def setPVday(self):
        ''' Set a 24h day '''
        self.solar['hour'] = np.zeros(24, dtype=int)
        for xt in range(24):
            self.solar['hour'][xt] = xt+1
