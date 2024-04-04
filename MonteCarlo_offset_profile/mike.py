# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:57:14 2023

@author: Roman Akhmetshyn
e-mail: romix_aa@ukr.net

A code that uses Monte-Carlo method to calculate offset host halo profile. 
The Monte-Carlo part is all in NFW_funcs.quick_MK_profile()
The rest is just modelling rings and circles and calculating their area.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from colossus.cosmology import cosmology
from NFW_funcs import quick_MK_profile

#setting global cosmology. Keep everything H=100, unless your data is different
params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%%
# bins=['0609', '0306', '0103'] #input distance bin, i.e. distance of lens galaxy from cluster center in Mpc
bins=[ '0306'] #input distance bin, i.e. distance of lens galaxy from cluster center in Mpc

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
    
    start_bin=0.01 #first ring lens-centric disatnce Mpc
    end_bin=1.5 #final ring lens-centric distance
    ring_incr=0.02 #distance between rings
    ring_num=round((end_bin-start_bin)/ring_incr) #number of rings
    ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000 #Mpc*1000=kpc, radii of all rings in kps
    threshold=ring_incr/2*100 #the same small width for each ring.
    
    mdef="200m" #cosmological mass definition 
    
    DeltaSigmas=np.zeros((1, len(ring_radii))) #np array for all delta sigma measurments
    debug_start = time.time() #timing the whole script
    
    with open(f'new-test/{bin}big05c(Mh70).txt', 'a+') as f: #text file which will contain all deltaSigma measurments
        np.savetxt(f, [ring_radii[::-1]], fmt='%f', newline='\n')
        
    halo_dict={} # a dictionary for each host halo, so we don't calculate same thing repeatedly
    
    for sat in lenses: #iterate through each lens
        # t1 = time.time()
        
        if sat['ID'] not in halo_dict: #check if M-C was calculated for this ID (host halo ID)
            halo_dict={} #empty the dictionary
            random_radii_x, random_radii_y = quick_MK_profile(sat['M_halo']*1.429,
                                                              #here I multiplied by 1.429 cuz I calculated
                                                              #masses for H=70 cosmology
                                                              sat['Z_halo'],
                                                              mass_per_point, 
                                                              "duffy08",
                                                              "200m",
                                                              cdf_resolution)
            
            halo_dict[sat['ID']]=[random_radii_x, random_radii_y] #add halo to the dictionary
            
        else:
            
            random_radii_x, random_radii_y = halo_dict[sat['ID']] #next lenses will use first M-C coordinates
            
        
        # print(time.time()-t1)
        
        
        # offset_sigmas=np.zeros((5,len(ring_radii)))
        # for offset_num, sat_x in enumerate(offsets[k]):
        
        sat_x = dist_file[num]['R0'] * 1000 #Mpc*1000 convert coords to kpc
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
    
        
        
        # sums=[]
    
        # for i in range(len(ring_radii)-1,-1,-1): #iterate through each radii
        #     DeltalessR=circle_counts[i]
        #     DeltaR=ring_counts[i]
        #     sums.append(DeltalessR-DeltaR) #Delta Sigmas for each ring-circle pair
        # sums=np.array(sums)
        
        sums = np.array([DeltalessR - DeltaR for DeltalessR, DeltaR in zip(circle_counts, ring_counts)])
            # data2 = {'Ring Radii': ring_radii[::-1], 'SigmaDelta(R)': sums} #a really useless variable in here
            # offset_sigmas[offset_num]=np.array(sums)
        
        # offset_weights=list(weights[k])
        # weighted_average_dsigma=np.average(offset_sigmas,axis=0,weights=offset_weights)
        with open(f'new-test/{bin}big05c(Mh70).txt', 'a+') as f: #each row is DeltaSigma of single lense
            np.savetxt(f, [sums], fmt='%f', newline='\n')
    
    
        DeltaSigmas=np.add(DeltaSigmas,np.array(sums))
        num+=1
        
        # fig, axes = plt.subplots(nrows=2, ncols=1)
    
        # # Plot the first graph on the left subplot
        # axes[0].plot(data['Ring Radii'],data['Delta(R)'])
        # axes[0].set_xlabel('kpc')
        # axes[0].set_ylabel('sigma(R)')
    
        
        # # Plot the second graph on the right subplot
        # axes[1].plot(data2['Ring Radii'],data2['SigmaDelta(R)'])
        # axes[1].set_xlabel('kpc')
        # axes[1].set_ylabel('DeltaSigma')
    
        # fig.suptitle('mpp=%.2e, n=%.3e, t=%.2f, rings=%i'%(mass_per_point, n_points,  t, len(ring_radii)))
    
        # plt.show()
    
    t=time.time() - debug_start
    print(
        f"Finished calculating {num} sat after",
        t,
    )
    avgDsigma=DeltaSigmas/len(lenses) #average Delta Sigma of all all lenses
    table=np.column_stack((ring_radii,avgDsigma[0]))
    np.savetxt(f'new-test/{bin}05c(Mh70).txt', table, delimiter='\t', fmt='%f') #save average delta sigma
    
    fig, axes = plt.subplots(nrows=1, ncols=1)
    
    # Plot the second graph on the right subplot
    axes.plot(ring_radii,avgDsigma[0])
    axes.set_xlabel('kpc')
    axes.set_ylabel('avg DeltaSigma')
    
    
    plt.show()
        
    #old plots for visualizations :
            
    # with plt.rc_context({"axes.grid": False}):
    #     fig, ax = plt.subplots(dpi=100)
    #     img = ax.hexbin(random_radii_x, random_radii_y, gridsize=100, bins="log")
    #     ax.plot(0, 0, "r+")
    
    #     for radius in ring_radii:
    #         circle = plt.Circle((sat_x, sat_y), radius, edgecolor='red', facecolor='none')
    #         ax.add_patch(circle)
        
    #     # ax.set_ylim(-radius, radius)
    #     # ax.set_xlim(-radius, radius)
    #     fig.colorbar(img)
    #     ax.set_aspect("equal")
    #     # ax.set_title(f"{len(multi_lenses)} halos")
    #     plt.show()
                
        


