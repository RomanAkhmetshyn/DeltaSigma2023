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
import math
import pandas as pd

import random

def make_NFW_stars(mass, redshift, A, stellar_mass, distbin, plot):
    params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
    cosmology.addCosmology("737", params)
    cosmo = cosmology.setCosmology("737")
    

    lens_z = redshift

    
    
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=lens_z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)
    # halo_profile = profile_nfw.NFWProfile(M=mass, c=c, z=lens_z, mdef="200m")
    subhalo_profile = NFW(mass, c, lens_z)
    # eta=2
    # tnfw = TNFW(mass, c, lens_z, tau, eta)
    
    
    R = np.linspace(0.01, 1.5, 75)
    # dSigma=np.squeeze(tnfw.projected_excess(r))/1000000
    # dSigma=subhalo_profile.deltaSigma(R*1000)
    dSigma= np.squeeze(subhalo_profile.projected_excess(R))/1000000
    
    # R = np.linspace(0.01, 1.5, 75)
    # dSigma=np.squeeze(tnfw.projected_excess(R))/1000000
    
    starSigma=stellar_mass/(math.pi*R**2)/1000000
    
    # dSigma=nfw.projected_excess(R)
    halo_dSigma = np.genfromtxt(f'{distbin}(Mh70).txt', delimiter='\t', usecols=(1))*A
    summed_halo=np.add(dSigma,halo_dSigma[::-1])
    summed_halo=np.add(summed_halo,starSigma)/1000000
    # summed_halo=np.mean([dSigma, halo_dSigma[::-1]], axis=0)
    # summed_halo=np.subtract(dSigma,halo_dSigma[::-1])
    host_halo=halo_dSigma[::-1]/1000000
    subhalo=dSigma/1000000
    star_halo=starSigma/1000000
    
    
    if plot:
        plt.plot(R, summed_halo, label='combined')
        
        plt.plot(R,host_halo, label='halo offset', linestyle='--')
        # plt.plot(R,offset_halo/1000000, label='excess', linestyle='--')
        # plt.plot(R,offset_halo2/1000000, label='sigma', linestyle='--')
        # plt.plot(R,offset_halo3/1000000, label='sigma(<R)', linestyle='--')
        
        plt.plot(R,subhalo,label='subhalo tNFW',  linestyle='--')
        plt.plot(R,star_halo,label='stars',  linestyle='--')
        plt.ylim(-40,120)
        plt.title(f'lens mass: {mass:.2e}, Z: {lens_z}, A: {A:.2},stellar mass: {stellar_mass:.2e}')
        plt.legend()
        plt.grid()
        # print(summed_halo[39])
        # print(np.min(summed_halo))
    
    return R, host_halo, subhalo, star_halo

def make_NFW(mass, redshift, A, distbin, plot):
    params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
    cosmology.addCosmology("737", params)
    cosmo = cosmology.setCosmology("737")
    

    lens_z = redshift

    
    
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=lens_z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)
    # halo_profile = profile_nfw.NFWProfile(M=mass, c=c, z=lens_z, mdef="200m")
    subhalo_profile = profile_nfw.NFWProfile(M=mass, c=c, z=lens_z, mdef="200m")
   
    # tnfw = TNFW(mass, c, z, tau, eta)
    
    
    R = np.linspace(0.01, 1.5, 75)
    # dSigma=np.squeeze(tnfw.projected_excess(r))/1000000
    dSigma=subhalo_profile.deltaSigma(R*1000)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    # dSigma=np.squeeze(tnfw.projected_excess(R))/1000000*B
    
    # star=math.pow(10, B)/R
    
    # dSigma=nfw.projected_excess(R)
    halo_dSigma = np.genfromtxt(f'{distbin}(Mh70).txt', delimiter='\t', usecols=(1))*A
    summed_halo=np.add(dSigma,halo_dSigma[::-1])/1000000
    # summed_halo=np.add(summed_halo,star)
    # summed_halo=np.mean([dSigma, halo_dSigma[::-1]], axis=0)
    # summed_halo=np.subtract(dSigma,halo_dSigma[::-1])
    host_halo=halo_dSigma[::-1]/1000000
    subhalo=dSigma/1000000
    # star_halo=star/1000000
    
    
    if plot:
        plt.plot(R, summed_halo, label='combined')
        
        plt.plot(R,host_halo, label='halo offset', linestyle='--')
        # plt.plot(R,offset_halo/1000000, label='excess', linestyle='--')
        # plt.plot(R,offset_halo2/1000000, label='sigma', linestyle='--')
        # plt.plot(R,offset_halo3/1000000, label='sigma(<R)', linestyle='--')
        
        plt.plot(R,subhalo,label='subhalo NFW',  linestyle='--')
        # plt.plot(R,star_halo,label='stars',  linestyle='--')
        plt.ylim(-40,120)
        plt.title(f'lens mass: {mass:.2e}, Z: {lens_z}, A: {A:.2}')
        plt.legend()
        plt.grid()
        # print(summed_halo[39])
        # print(np.min(summed_halo))
    
    return R, host_halo, subhalo

def make_profile(mass, redshift, tau, A, B, distbin, plot):
    params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
    cosmology.addCosmology("737", params)
    cosmo = cosmology.setCosmology("737")
    
    z_array=[0.012, 0.05, 0.043, 0.034, 0.015, 0.008, 0.08, 0.017, 0.014]
    # lens_z = random.choice(z_array)
    lens_z = redshift
    mass_array=[1.5e12, 1.2e12, 5e10, 6.6e12, 1e11, 1.2e11, 1.1e11, 4e10, 0.9e11, 2e12]
    # mass = random.choice(mass_array)
    
    
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=lens_z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)
    # halo_profile = profile_nfw.NFWProfile(M=mass, c=c, z=lens_z, mdef="200m")
    # halo_profile = NFW(mass,c,lens_z)
    
    # scale_radius = halo_profile.getParameterArray()[1]/1000
    # trunc_array=np.linspace(0.01, 0.36, 10+1)
    # trunc_radius=random.choice(trunc_array)
    # trunc_radius=0.175
    # tau=trunc_radius/scale_radius
    # mass=math.pow(10,mass)
    # print(mass)
    eta=2
    
    
    tnfw = TNFW(mass, c, lens_z, tau, eta)
    
    
    R = np.linspace(0.01, 1.5, 75)
    dSigma=np.squeeze(tnfw.projected_excess(R))/1000000*B
    
    # offset_halo=np.squeeze(halo_profile.offset_projected_excess(R, 0.6))/1000000
    # offset_halo2=np.squeeze(halo_profile.offset_projected(R, 0.6))/1000000
    # offset_halo3=np.squeeze(halo_profile.offset_projected_cumulative(R, 0.6))/1000000
    # dSigma=nfw.projected_excess(R)
    halo_dSigma = np.genfromtxt(f'{distbin}(Mh70).txt', delimiter='\t', usecols=(1))*A
    summed_halo=np.add(dSigma,halo_dSigma[::-1])/1000000
    # summed_halo=np.mean([dSigma, halo_dSigma[::-1]], axis=0)
    # summed_halo=np.subtract(dSigma,halo_dSigma[::-1])
    host_halo=halo_dSigma[::-1]/1000000
    subhalo=dSigma/1000000
    
    if plot:
        plt.plot(R, summed_halo, label='combined')
        
        plt.plot(R,host_halo, label='halo', linestyle='--')
        # plt.plot(R,offset_halo/1000000, label='excess', linestyle='--')
        # plt.plot(R,offset_halo2/1000000, label='sigma', linestyle='--')
        # plt.plot(R,offset_halo3/1000000, label='sigma(<R)', linestyle='--')
        
        plt.plot(R,subhalo,label='subhalo',  linestyle='--')
        plt.ylim(-40,120)
        plt.title(f'lens mass: {mass:.2e}, Z: {lens_z}, Rt/Rs: {tau:.2}, A: {A:.2}, B:{B:.2}')
        plt.legend()
        plt.grid()
        # print(summed_halo[39])
        # print(np.min(summed_halo))
    
    return R, host_halo, subhalo

if __name__ == '__main__':
    # rp,host,sub=make_profile(math.pow(10,12.74), 0.23, math.pow(10,3.48), 0.81 , float(1), '0609', True)
    
    
    # fig, ax = plt.subplots()
    # taus=[0.2, 0.4, 0.8, 1.2, 2, 3, 5, 10]
    # df = pd.DataFrame()
    # df['Rp']=np.linspace(0.01, 1.5, 75)
    # for tau in taus:

    #     rp,host,sub=make_profile(1e12, 0.31, float(tau), 1 , float(1), '0306', False)
        
       
    #     ax.plot(rp,host+sub)
    #     # plt.ylim(-10,100)
    #     # plt.xlim(0,0.4)
    #     column_name = f'column_{tau}'
    #     df[column_name] = host+sub
    
    # concentration_model="duffy08"
    # c=concentration.concentration(
    #         M=1e12, mdef="200m", z=0.31, model=concentration_model
    #         )
    # subhalo_profile = NFW(1e12, c, 0.31)
    # # eta=2
    # # tnfw = TNFW(mass, c, lens_z, tau, eta)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    # # dSigma=np.squeeze(tnfw.projected_excess(r))/1000000
    # # dSigma=subhalo_profile.deltaSigma(R*1000)
    # dSigma= np.squeeze(subhalo_profile.projected_excess(R))/1000000
    

    
    # # dSigma=nfw.projected_excess(R)
    # halo_dSigma = np.genfromtxt('0306(Mh70).txt', delimiter='\t', usecols=(1))
    # summed_halo=np.add(dSigma,halo_dSigma[::-1])/1000000
    # ax.plot(R, summed_halo, label='nfw', color='red')
    # ax.legend()
    # plt.grid()
    # plt.show()
    # df['nfw'] = summed_halo
    # csv_filename = 'tauandprofile.csv'
    # df.to_csv(csv_filename, index=False)
    
    
    
    # fig, ax = plt.subplots()
    # for mass in np.arange(11.5,12.2,0.1):

    #     rp,host,sub=make_profile(math.pow(10,mass), 0.31, float(100), 0.81 , float(1), '0306', False)
        
       
    #     ax.plot(rp,host+sub,label=f'{mass}')
    #     plt.ylim(-10,100)
    #     # plt.xlim(0,0.4)
        

    # ax.legend()
    # plt.grid()
    
    rp,host,sub, star=make_NFW_stars(1e12, 0.31, float(1), 1e12, '0306', True)