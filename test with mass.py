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
from subhalo_profile import make_profile
import math

bin='0609'

if bin=='0609':
    lowlim=0.6
    highlim=0.9
elif bin=='0306':
    lowlim=0.3
    highlim=0.6
elif bin=='0103':
    lowlim=0.1
    highlim=0.3 

lenses = Table.read("./data/redmapper_mnc_allz.fits")
data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
       
    )
lenses = lenses[data_mask]

z=np.mean(lenses['z_any'])
# z=np.mean(lenses['zspec'])

def summed_profile(r,mass,tau,A):
# def subhalo_profile(r,mass, A):
    # print(mass)
    # print(tau)
    # print(z)
    # print(A)
    # tau=200
    mass=math.pow(10, mass)

    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta=2
    tnfw = TNFW(mass, c, z, tau, eta)
    
    profile=tnfw
    # R = np.linspace(0.01, 1.5, 75)
    dSigma=np.squeeze(tnfw.projected_excess(r))/1000000


    
    # dSigma=nfw.projected_excess(R)
    halo_table = np.genfromtxt(f'{bin}(Mh70).txt', delimiter='\t', usecols=(0, 1), dtype=float)
    halo_r=halo_table[:,0]/1000
    
    halo_ds=halo_table[:,1]
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)*A

    summed_halo=np.add(dSigma,halo_dSigma)/1000000

    return profile, summed_halo

def subhalo_profile(r,mass,tau,A):
# def subhalo_profile(r,mass, A):
    # print(mass)
    # print(tau)
    # print(z)
    # print(A)
    # tau=200
    mass=math.pow(10, mass)

    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta=2
    tnfw = TNFW(mass, c, z, tau, eta)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    dSigma=np.squeeze(tnfw.projected_excess(r))/1000000


    
    # dSigma=nfw.projected_excess(R)
    halo_table = np.genfromtxt(f'{bin}(Mh70).txt', delimiter='\t', usecols=(0, 1), dtype=float)
    halo_r=halo_table[:,0]/1000
    
    halo_ds=halo_table[:,1]
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)*A

    summed_halo=np.add(dSigma,halo_dSigma)/1000000

    return summed_halo



# Read the CSV file
# df = pd.read_csv(f'D:/GitHub/summer-research/output-roman(correct)/roman_esd_ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
df = pd.read_csv(f'D:/roman_esd_ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
# df=pd.read_csv(f'D:/GitHub/summer-research/output/{bin}.txt',
#                 delim_whitespace = True, 
#                 names = ['rp','ds','ds_err'], 
#                 comment = '#')

# Save the "ds" and "rp" columns as variables
ds = df['ds']
rp = df['rp']
ds_err=df['ds_err']

init=[12, 1, 1]

param_bounds=([10, 0,-np.inf], [np.inf, 100, np.inf])

popt, pcov = curve_fit(subhalo_profile, rp, ds, p0=init, bounds=param_bounds, sigma=ds_err, absolute_sigma=True,
                       method='dogbox')

# print(pcov)
# print(popt)
# Extract the optimized parameters

# lens_mass, lens_tau , lens_z, param_A = popt
lens_mass, lens_tau, param_A= popt
lens_z=z

profile, fit = summed_profile(rp, lens_mass, lens_tau, param_A)

deltaC=profile.delta_c
rho_crit=profile.critical_density
Rs=profile.rs[0]
Rt=lens_tau*Rs
M0=deltaC*rho_crit*16*math.pi*Rs**3
print("M0(rho, Rs): ",math.log10(M0))
t=lens_tau
denominator=(t**2/((t**2+1)**2))*((t**2-1)*math.log(t)+t*math.pi-(t**2+1))
Mo=math.pow(10,lens_mass)/denominator
print("M0(tau): ",math.log10(Mo))
print('lense mass: ',lens_mass)

r_full,ds_halo, ds_sub=make_profile(math.pow(10,lens_mass), lens_z, lens_tau, param_A,  B=1, distbin=bin, plot=False)
# r_full,ds_full=make_profile(1e12, 0.35, 35, 0.6, distbin=bin, plot=False)
# plt.plot(rp, ds, 'bo', label='Isaac Data')
# plt.plot(rp, fit, 'r-', label='interp Curve')
plt.plot(r_full, ds_halo+ds_sub, 'g-', label='fitted Curve')
plt.errorbar(rp, ds, ds_err, fmt='o',label='dsigma Data')
plt.plot(r_full, ds_halo, '--', label='halo')
plt.plot(r_full, ds_sub, '--', label='subhalo')
plt.xlabel('R (Mpc)')
plt.ylabel('M/pc^2')
plt.grid()
plt.ylim(-40,120)
plt.title(f'{bin} lens log(mass): {lens_mass:.2f}, Z: {lens_z:.2}, tau: {lens_tau:.2f}, A: {param_A:.2}')
plt.legend()
plt.show()

