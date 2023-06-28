# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:26:08 2023

@author: Admin
"""

import numpy as np
from profiley.nfw import TNFW, NFW
import matplotlib.pyplot as plt
from colossus.halo import concentration, profile_nfw
from colossus.cosmology import cosmology


import random


params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

z_array=[0.012, 0.05, 0.043, 0.034, 0.015, 0.008, 0.08, 0.017, 0.014]
# lens_z = random.choice(z_array)
lens_z = 0.334
mass_array=[1.5e12, 1.2e12, 5e10, 6.6e12, 1e11, 1.2e11, 1.1e11, 4e10, 0.9e11, 2e12]
# mass = random.choice(mass_array)
mass = 179405003945.4444

concentration_model="duffy08"
c=concentration.concentration(
        M=mass, mdef="200m", z=lens_z, model=concentration_model
        )

halo_profile = profile_nfw.NFWProfile(M=mass, c=c, z=lens_z, mdef="200m")

scale_radius = halo_profile.getParameterArray()[1]/1000
trunc_array=np.linspace(0.01, 0.36, 10+1)
trunc_radius=random.choice(trunc_array)
# trunc_radius=0.175
# tau=trunc_radius/scale_radius
tau=38482.44389563829
eta=2

tnfw = TNFW(mass, c, lens_z, tau, eta)


R = np.linspace(0.01, 1.5, 75)
dSigma=np.squeeze(tnfw.projected_excess(R))/1000000
# dSigma=nfw.projected_excess(R)
halo_dSigma = np.genfromtxt('0609.txt', delimiter='\t', usecols=(1))
summed_halo=np.add(dSigma,halo_dSigma[::-1])/1000000
# summed_halo=np.mean([dSigma, halo_dSigma[::-1]], axis=0)
# summed_halo=np.subtract(dSigma,halo_dSigma[::-1])


plt.plot(R, summed_halo, label='combined')

plt.plot(R,halo_dSigma[::-1]/1000000, label='halo', linestyle='--')

plt.plot(R,dSigma/1000000,label='subhalo',  linestyle='--')
# plt.ylim(np.min(halo_dSigma),np.amax(halo_dSigma))
plt.title(f'lens mass: {mass:.2e}, Z: {lens_z}, Rt/Rs: {tau:.2}')
plt.legend()