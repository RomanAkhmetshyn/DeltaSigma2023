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

chis=[]
taus=[]

for tau in np.arange(200,1000,50):
    print(tau)
    def subhalo_profile(r,mass,A):
    # def subhalo_profile(r,mass, A):
        # print(mass)
        
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
    ds_ar=ds.values
    rp = df['rp']
    ds_err=df['ds_err']
    
    init=[12, 1]
    
    param_bounds=([10, -np.inf], [np.inf, np.inf])
    
    popt, pcov = curve_fit(subhalo_profile, rp, ds, p0=init, bounds=param_bounds, sigma=ds_err, absolute_sigma=True,
                           method='dogbox')
    
    # print(pcov)
    # print(popt)
    # Extract the optimized parameters
    
    # lens_mass, lens_tau , lens_z, param_A = popt
    lens_mass,  param_A= popt
    lens_z=z
    
    fit = subhalo_profile(rp, lens_mass, param_A)
    
    
    residual=np.subtract(fit,ds) 
    residual_sq=[x**2 for x in residual] #get square of residuals
    chi2=np.sum(np.array(residual_sq)/np.array(ds)) #chi^2
    chis.append(chi2)
    taus.append(tau)
    

plt.plot(taus,chis)
plt.xlabel('tau')
plt.ylabel('chi^2')
plt.title(f'{bin}')
plt.grid()
plt.show()


