# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:42:59 2023

@author: Admin
"""

import pandas as pd
from profiley.nfw import TNFW, NFW
import matplotlib.pyplot as plt
from colossus.halo import concentration, profile_nfw
from colossus.cosmology import cosmology
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit

bin='0306'

lenses = Table.read("./data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits")
data_mask = (
        (lenses["R"] >= 0.3)
        & (lenses["R"] < 0.6)
        # & (lenses["PMem"] > 0.8)
        & (lenses["zspec"] > -1)
       
    )
lenses = lenses[data_mask]

# z=np.mean(lenses['zspec'])

def subhalo_profile(r,mass,tau,z ):
    print(mass)
    print(tau)
    print(z)
    
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )

    eta=2
    tnfw = TNFW(mass, c, z, tau, eta)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    dSigma=np.squeeze(tnfw.projected_excess(r))/100000


    
    # dSigma=nfw.projected_excess(R)
    halo_table = np.genfromtxt(f'{bin}(zspec).txt', delimiter='\t', usecols=(0, 1), dtype=float)
    halo_r=halo_table[:,0]/1000
    
    halo_ds=halo_table[:,1]
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)

    summed_halo=np.add(dSigma,halo_dSigma)/1000000

    return summed_halo

# def custom_func(r,mass,tau,z):
#     if tau < 0 :
#         return np.inf  # Return infinity if the condition is violated
#     if z<0.05 or z>0.8:
#         return np.inf
#     return subhalo_profile(r,mass,tau,z)

# Read the CSV file
df = pd.read_csv(f'D:/GitHub/summer-research/data/dsigma_measurements/output/ShapePipe{bin}.csv')

# Save the "ds" and "rp" columns as variables
ds = df['ds']
rp = df['rp']
ds_err=df['ds_err']

init=[1e12, 5.83, 0.3]
param_bounds=([0, 0, 0], [np.inf, 35, 0.8])
# Fit the curve to the data
# popt, pcov = curve_fit(custom_func, rp, ds, p0=init)
popt, pcov = curve_fit(subhalo_profile, rp, ds, p0=init, bounds=param_bounds, sigma=ds_err, absolute_sigma=True)

# Extract the optimized parameters

lens_mass, lens_tau , lens_z = popt

fit = subhalo_profile(rp, lens_mass, lens_tau, lens_z)


# plt.plot(rp, ds, 'bo', label='Isaac Data')
plt.plot(rp, fit, 'r-', label='Fitted Curve')
plt.errorbar(rp, ds, ds_err, label='Isaac Data')
plt.xlabel('R (Mpc)')
plt.ylabel('M/pc^2')
plt.title(f'{bin} lens mass: {lens_mass:.2e}, Z: {lens_z:.2}, Rt/Rs: {lens_tau:.2}')
plt.legend()
plt.show()

