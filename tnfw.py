# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:55:02 2023

@author: Admin
"""

import math
import numpy as np
from profiley.nfw import TNFW, NFW
import matplotlib.pyplot as plt
from colossus.halo import concentration, profile_nfw
from colossus.cosmology import cosmology

mass=math.pow(10,12.77)
params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")


lens_z = 0.23



concentration_model="duffy08"
c=concentration.concentration(
        M=mass, mdef="200m", z=lens_z, model=concentration_model
        )

NFW_profile = profile_nfw.NFWProfile(M=mass, c=c, z=lens_z, mdef="200m")
# halo_profile = NFW(mass,c,lens_z)

scale_radius = NFW_profile.getParameterArray()[1]/1000
# trunc_array=np.linspace(0.01, 0.36, 10+1)
# trunc_radius=random.choice(trunc_array)
eta=2

# trunc_radius=1.92
# tau=trunc_radius/scale_radius

# print(tau)

fig, ax = plt.subplots()
for trunc_radius in np.arange(0,1, 0.05):

    tau=trunc_radius/scale_radius
    
    # ax.set_yscale('log')
    print(tau)
    
    
    tnfw = TNFW(mass, c, lens_z, tau, eta)
    
    nfw=NFW(mass, c, lens_z)
    
    
    R = np.linspace(0.01, 1.5, 75)
    subhalo=np.squeeze(tnfw.projected_excess(R))/1000000/1000000
    esd = nfw.projected_excess(R)
    plt.ylim(-10,40)
    plt.xlim(0,0.4)
    
    ax.plot(R,subhalo,  linestyle='--',label=f'{tau}')
    
    # plt.ylim(-40,120)
ax.plot(R,esd/1000000/1000000, color='r',label='ESD')    

# ax.legend()
plt.grid()
plt.show()