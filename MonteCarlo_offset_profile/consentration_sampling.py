# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:27:14 2024

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
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

#setting global cosmology. Keep everything H=100, unless your data is different
params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%%

start_bin=0.01 * 1.429 #first ring lens-centric disatnce Mpc
end_bin=1.5 * 1.429 #final ring lens-centric distance
ring_incr=0.02 #distance between rings
ring_num=round((end_bin-start_bin)/ring_incr) #number of rings
ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000 #Mpc*1000=kpc, radii of all rings in kps

bins=[ '0609'] #input distance bin, i.e. distance of lens galaxy from cluster center in Mpc

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

def halo(rp, A, c):

    # print(A, c)
    num=0 #iterator for number of lenses calculated, can be removed
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
    

    threshold=ring_incr/2*100 #the same small width for each ring.
    
    mdef="200m" #cosmological mass definition 
    
    DeltaSigmas=np.zeros((1, len(ring_radii))) #np array for all delta sigma measurments
    debug_start = time.time() #timing the whole script
    
    # with open(f'new-test/{bin}(H70).txt', 'a+') as f: #text file which will contain all deltaSigma measurments
    #     np.savetxt(f, [ring_radii[::-1]], fmt='%f', newline='\n')
        
    halo_dict={} # a dictionary for each host halo, so we don't calculate same thing repeatedly
    
    for sat in lenses[:200]: #iterate through each lens
        # t1 = time.time()
        
        if sat['ID'] not in halo_dict: #check if M-C was calculated for this ID (host halo ID)
            halo_dict={} #empty the dictionary
            # c=concentration.concentration(
            #     M=sat['M_halo'], mdef="200m", z=sat['Z_halo'], model="duffy08"
            # ) #calculate concentration using colossus
            
            # c = np.power(4.67*(sat['M_halo']/(1e14*1.429)), -0.11)
            
            random_radii_x, random_radii_y = quick_MK_profile(sat['M_halo'],
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
            

        sat_x = dist_file[num]['R0'] * 1000 * 1.429 #Mpc*1000 convert coords to kpc
        sat_y = 0
        
        S=[np.pi*((r+threshold)**2-(r-threshold)**2) for r in ring_radii] #area if rings
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
            mask = np.logical_and(0 <= distances, distances <= ring_radii[i] - threshold)
            
            circle_counts[i] = np.sum(mask)*mass_per_point/(np.pi*(ring_radii[i]- threshold)**2)
    

        sums = np.array([DeltalessR - DeltaR for DeltalessR, DeltaR in zip(circle_counts, ring_counts)])

    
        DeltaSigmas=np.add(DeltaSigmas,np.array(sums))
        num+=1

    t=time.time() - debug_start
    print(
        f"Finished calculating {num} sat after",
        t,
    )
    avgDsigma=DeltaSigmas/len(lenses) #average Delta Sigma of all all lenses
    table=np.column_stack((ring_radii,avgDsigma[0]))
    # np.savetxt(f'new-test/{bin}(H70)(1).txt', table, delimiter='\t', fmt='%f') #save average delta sigma
    
    f = interp1d(ring_radii, avgDsigma[0], kind='cubic')
    halo_dSigma=f(rp*1000)
    return halo_dSigma/A
    
    #%%
    


data_path = 'C:/catalogs'        
df = pd.read_csv(data_path+f'/roman_esd_70ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
ds = (df['ds']).values
ds = ds[-9:]
rp = (df['rp']).values
rp = rp[-9:]
ds_err=(df['ds_err']).values
ds_err = ds_err[-9:]

init=np.array([18000, 2])


popt, pcov = curve_fit(halo, rp, ds, p0=init, sigma=ds_err, absolute_sigma=True,
                       method='dogbox')

A, c= popt

fit = halo(rp, A, c)


#%%


fig, axes = plt.subplots(nrows=1, ncols=1)

# Plot the second graph on the right subplot
axes.errorbar(rp, ds, ds_err, fmt='o', markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
axes.plot(rp,fit)
axes.set_xlabel('kpc')
axes.set_ylabel('avg DeltaSigma')


plt.show()