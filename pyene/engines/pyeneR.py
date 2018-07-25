# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:19:20 2018

@author: mchihem2
"""
import numpy as np
import math


class RESprofiles():
    ''' Produce time series for different RES '''
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
