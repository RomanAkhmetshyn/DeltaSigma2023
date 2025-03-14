# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:17:52 2024

@author: romix
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from colossus.cosmology import cosmology
from colossus.halo import concentration, profile_nfw
from NFW_funcs import quick_MK_profile
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import trange
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def random_points(halo_mass,
                      halo_z,
                      mass_per_point,
                      c,
                      concentration_model="duffy08",
                      mdef="200m",
                      cdf_resolution=1000):
    """
    

    Parameters
    ----------
    double halo_mass : double
        Mass of the host halo in M_sun.
    double halo_z : double
        Host halo redshift.
    double mass_per_point : double
        Mass asigned for every random point in M-C.
    str concentration_model : str, optional
        Concentration model for colossus calculation. The default is "duffy08".
    str mdef : str, optional
        Mass model for colossus calculation. The default is "200m".
    int cdf_resolution : int, optional
        Number of points for interpolation of probability function for the profile.

    Returns
    -------
    random_radii_x : numpy array
        X coords of M-C point.
    random_radii_y : numpy array
        Y coords of M-C point.

    """

    halo_profile = profile_nfw.NFWProfile(M=halo_mass, c=c, z=halo_z, mdef=mdef) #build host halo NFW

    scale_radius = halo_profile.getParameterArray()[1] #get NFW profile parameter R_scale
    virial_radius = scale_radius * c #R_vir= concentration * R_scale
    #
    # Determine CDF of projected (2D) NFW enclosed mass
    #CDF - cumulative distribution function
    #
    interp_radii = np.linspace(0, virial_radius*4, cdf_resolution) #distance for cdf
    

    # Temporarily ignore division by zero and overflow warnings
    with np.errstate(divide="ignore", over="ignore"):
        interp_delta_sigmas = halo_profile.deltaSigma(interp_radii)
        interp_surface_densities = halo_profile.surfaceDensity(interp_radii)
        # interp_enclosed_masses = halo_profile.enclosedMass(interp_radii)
    # Correct delta sigmas and surface densities at r=0 to be zero
    interp_delta_sigmas[0] = 0.0
    
    #!!!
    plt.plot(interp_radii[1:], interp_delta_sigmas[1:], linewidth=1, c='blue', alpha =0.5,
             linestyle='--')
    
    
    #!!!
    interp_surface_densities[0] = 0.0
    interp_2d_encl_masses = (
        np.pi * interp_radii**2 * (interp_delta_sigmas + interp_surface_densities)
    )
    # interp_2d_encl_masses = interp_enclosed_masses


    n_points = round(interp_2d_encl_masses[-1] / (mass_per_point)) 

    #
    # Make 1D interpolator for this halo
    #

    interp_normed_2d_encl_masses = interp1d(
        interp_2d_encl_masses / interp_2d_encl_masses[-1], 
        interp_radii,
        assume_sorted=True,
    )


    #
    # Generate random points for this halo + offset combination
    rng = np.random.default_rng()

    random_cdf_yvals = rng.uniform(0, 1, size=n_points)
    # print("Begin interpolation")

    random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)

    random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
    random_radii_x = random_radii * np.cos(random_azimuths) 
    random_radii_y = random_radii * np.sin(random_azimuths) 

    
    return random_radii_x, random_radii_y

#setting global cosmology. Keep everything H=100, unless your data is different
params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

bins=['0609'] #input distance bin, i.e. distance of lens galaxy from cluster center in Mpc
# bins=[ '0103'] #input distance bin, i.e. distance of lens galaxy from cluster center in Mpc

for bin in bins:

    if bin=='0609':
        lowlim=0.6
        highlim=0.9
    elif bin=='0306':
        lowlim=0.3
        highlim=0.6
    elif bin=='0103':
        lowlim=0.1
        highlim=0.3 
        

    lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits") #RedMaPPer catalog -
    #Combined by myself with host halo masses and redshifts - email me if you want it
    dist_file=Table.read(f'C:/catalogs/{bin}_members_dists.fits')
    
    
    #filter lenses that are in a distance bin. You can also filter by membership probability and redshift
    data_mask = (
            (lenses["R"] >= lowlim)
            & (lenses["R"] < highlim)
            # & (lenses["PMem"] > 0.8)
            # & (lenses["zspec"] > -1)
            & (lenses["PMem"] > 0.8)
        )
    lenses = lenses[data_mask] #updated table of lenses
    cdf_resolution=1000 #interpolation resulotion, i.e. number of points for probability distribution func.
    mass_per_point = 1098372.008822474*10000 #arbitrary number, Mass/mpp = number of points for M-C
    
    # start_bin=0.01 * 1.429 #first ring lens-centric disatnce Mpc
    start_bin = 0.002 * 1.429
    # end_bin=1.5 * 1.429 #final ring lens-centric distance
    end_bin=2.5 * 1.429 #final ring lens-centric distance
    ring_incr=0.02 * 1.429 #distance between rings
    ring_num=round((end_bin-start_bin)/ring_incr) #number of rings
    ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000 #Mpc*1000=kpc, radii of all rings in kps
    # threshold=ring_incr/2*100 #the same small width for each ring.
    threshold=ring_incr/4*100 #the same small width for each ring.
    # threshold = 0.5
    
    mdef="200m" #cosmological mass definition 
    
    DeltaSigmas=np.zeros((1, len(ring_radii))) #np array for all delta sigma measurments
    debug_start = time.time() #timing the whole script
    
        
    halo_dict={} # a dictionary for each host halo, so we don't calculate same thing repeatedly
    lenses = lenses[:1000]
    dist_file = dist_file[:1000]
    

    for s in trange(len(lenses)): #iterate through each lens
        sat = lenses[s]
        
        if sat['ID'] not in halo_dict: #check if M-C was calculated for this ID (host halo ID)
            halo_dict={} #empty the dictionary
            # c=concentration.concentration(
            #     M=sat['M_halo'], mdef="200m", z=sat['Z_halo'], model="duffy08"
            # ) #calculate concentration using colossus
            
            # c = 4.67*np.power((sat['M_halo']/(1e14*1.429)), -0.11)
            
            if 0.08<sat['Z_halo']<=0.35:
                C0 = 5.119
                gamma = 0.205
                M = sat['M_halo']
                M0 = np.power(10, 14.083)
            else:
                C0 = 4.875
                gamma = 0.221
                M = sat['M_halo']
                M0 = np.power(10, 13.750)
                
            c = C0 * np.power(M/1e12,-gamma) * (1 + np.power(M/M0, 0.4))
            
            random_radii_x, random_radii_y = random_points(sat['M_halo'],
                                                              #here I multiplied by 1.429 cuz I calculated
                                                              #masses for H=70 cosmology
                                                              sat['Z_halo'],
                                                              mass_per_point,
                                                              c,
                                                              "duffy08",
                                                              "200m",
                                                              cdf_resolution)
            
            halo_dict[sat['ID']]=[random_radii_x, random_radii_y] #add halo to the dictionary
            
        else:
            
            random_radii_x, random_radii_y = halo_dict[sat['ID']] #next lenses will use first M-C coordinates
            
        
        sat_x = dist_file[s]['R0'] * 1000 * 1.429 * 0 #Mpc*1000 convert coords to kpc
        # sat_x = sat['R'] * 1000 * 1.429 #Mpc*1000 convert coords to kp
        sat_y = 0
        
        S=[np.pi*((r+threshold)**2-(r-threshold)**2) for r in ring_radii] #area of rings
        # Calculate the distances for all random points at once
        distances = np.sqrt((random_radii_x - sat_x)**2 + (random_radii_y - sat_y)**2)
        
        # Create an empty array to store the counts for each ring
        ring_counts = np.zeros(len(ring_radii), dtype=np.int64) #counts in the rings
        circle_counts = np.zeros(len(ring_radii), dtype=np.int64) #counts in enclosed circles
        
        # Iterate over each ring radius and count the points within each ring
        for i in range(len(ring_radii)):
            mask = np.logical_and(ring_radii[i] - threshold <= distances, 
                                  distances <= ring_radii[i] + threshold) #mask points that are within ring
            
            
            ring_counts[i] = np.sum(mask)*mass_per_point/S[i] #get surface density in rings
            
        for i in range(len(ring_radii)): #the same but iterate each circle
            # mask = np.logical_and(0 <= distances, distances <= ring_radii[i] - threshold) #!!!
            mask = np.logical_and(0 <= distances, distances <= ring_radii[i]) #!!!
            
            # circle_counts[i] = np.sum(mask)*mass_per_point/(np.pi*(ring_radii[i]- threshold)**2) #!!!
            circle_counts[i] = np.sum(mask)*mass_per_point/(np.pi*(ring_radii[i])**2) #!!!
    
        
        sums = np.array([DeltalessR - DeltaR for DeltalessR, DeltaR in zip(circle_counts, ring_counts)])
    
    
        DeltaSigmas=np.add(DeltaSigmas,np.array(sums))


    t=time.time() - debug_start
    print(
        f"Finished calculating {s} sat after",
        t,
    )
    avgDsigma=DeltaSigmas/len(lenses) #average Delta Sigma of all all lenses
    table=np.column_stack((ring_radii,avgDsigma[0]))
    # np.savetxt(f'new-test/{bin}C_Xu.txt', table, delimiter='\t', fmt='%f') #save average delta sigma
    # plt.vlines(np.mean(dist_file['R0']) * 1000 * 1.429, -2e8, 2e8, color='r')
    # print(np.mean(lenses['R'])*1.429 * 1000)
    # plt.vlines(np.mean(lenses['R']) * 1000 * 1.429, -2e8, 2e8, color='r')
    plt.plot(ring_radii[1:],avgDsigma[0,1:], color='r', linewidth=1.8, linestyle='-',
             label =f'{len(lenses)} - avg dsigma 0 offset')
    plt.legend()
    plt.xlim(0,4000)
    plt.show()
    