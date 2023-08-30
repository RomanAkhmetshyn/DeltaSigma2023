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
from NFW_funcs import quick_MK_profile #COMMENT THIS LINE IF YOU DON'T WANT TO RUN CYTHON

'''UNCOMMENT FOLLOWING FUNCTION IF YOU DON'T WANT TO RUN ON CYTHON'''

# def quick_MK_profile(halo_mass,
#                      halo_z,
#                      mass_per_point,
#                      concentration_model="duffy08",
#                      mdef="200m",
#                      cdf_resolution=1000):
#     """
    

#     Parameters
#     ----------
#     double halo_mass : double
#         Mass of the host halo in M_sun.
#     double halo_z : double
#         Host halo redshift.
#     double mass_per_point : double
#         Mass asigned for every random point in M-C.
#     str concentration_model : str, optional
#         Concentration model for colossus calculation. The default is "duffy08".
#     str mdef : str, optional
#         Mass model for colossus calculation. The default is "200m".
#     int cdf_resolution : int, optional
#         Number of points for interpolation of probability function for the profile.

#     Returns
#     -------
#     random_radii_x : numpy array
#         X coords of M-C point.
#     random_radii_y : numpy array
#         Y coords of M-C point.

#     """
#     from colossus.halo import concentration, profile_nfw
#     from scipy.interpolate import interp1d
#     c=concentration.concentration(
#         M=halo_mass, mdef="200m", z=halo_z, model=concentration_model
#     ) #calculate concentration using colossus
#     halo_profile = profile_nfw.NFWProfile(M=halo_mass, c=c, z=halo_z, mdef=mdef) #build host halo NFW

#     scale_radius = halo_profile.getParameterArray()[1] #get NFW profile parameter R_scale
#     virial_radius = scale_radius * c #R_vir= concentration * R_scale
#     #
#     # Determine CDF of projected (2D) NFW enclosed mass
#     #CDF - cumulative distribution function
#     #
#     interp_radii = np.linspace(0, virial_radius, cdf_resolution) #distance for cdf
    

#     # Temporarily ignore division by zero and overflow warnings
#     with np.errstate(divide="ignore", over="ignore"):
#         interp_delta_sigmas = halo_profile.deltaSigma(interp_radii)
#         interp_surface_densities = halo_profile.surfaceDensity(interp_radii)
#     # Correct delta sigmas and surface densities at r=0 to be zero
#     interp_delta_sigmas[0] = 0.0
#     interp_surface_densities[0] = 0.0
#     interp_2d_encl_masses = (
#         np.pi * interp_radii**2 * (interp_delta_sigmas + interp_surface_densities)
#     )



#     n_points = round(interp_2d_encl_masses[-1:][0] / (mass_per_point))
#     print("For each offset, will generate", n_points, "points for this halo")
#     #
#     # Make 1D interpolator for this halo
#     #

#     interp_normed_2d_encl_masses = interp1d(
#         interp_2d_encl_masses / interp_2d_encl_masses[-1:][0],
#         interp_radii,
#         assume_sorted=True,
#     )

    

#     #
#     # Generate random points for this halo + offset combination
#     #
#     rng = np.random.default_rng()
#     offset=0
#     offset_angle = rng.uniform(0, 2 * np.pi)
#     offset_x = offset * np.cos(offset_angle)
#     offset_y = offset * np.sin(offset_angle)
#     #
#     random_cdf_yvals = rng.uniform(0, 1, size=n_points)
#     # print("Begin interpolation")

#     random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)

#     random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
#     random_radii_x = random_radii * np.cos(random_azimuths) + offset_x
#     random_radii_y = random_radii * np.sin(random_azimuths) + offset_y
#     # print("Begin extending list")

#     #if return_xy:

#     #else:
#     # random_r=np.array([ np.sqrt(random_radii_x**2 + random_radii_y**2)])
#     # random_theta=np.array([ np.arctan2(random_radii_y, random_radii_x)])
#     # print("Finished extending list after", time.time() - debug_start4)
#     # print()
    
#     return random_radii_x, random_radii_y

#setting global cosmology. Keep everything H=100, unless your data is different
params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%%
bin='0306' #input distance bin, i.e. distance of lens galaxy from cluster center in Mpc

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
lenses = Table.read("./data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits") #RedMaPPer catalog -
#Combined by myself with host halo masses and redshifts - email me if you want it

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

with open(f'{bin}big(Mh70).txt', 'a+') as f: #text file which will contain all deltaSigma measurments
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
    num+=1
    sat_x = sat['R'] * 1000 #Mpc*1000 convert coords to kpc
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

    
    
    sums=[]

    for i in range(len(ring_radii)-1,-1,-1): #iterate through each radii
        DeltalessR=circle_counts[i]
        DeltaR=ring_counts[i]
        sums.append(DeltalessR-DeltaR) #Delta Sigmas for each ring-circle pair
    data2 = {'Ring Radii': ring_radii[::-1], 'SigmaDelta(R)': sums} #a really useless variable in here
    with open(f'{bin}big(Mh70).txt', 'a+') as f: #each row is DeltaSigma of single lense
        np.savetxt(f, [sums], fmt='%f', newline='\n')
    # t=time.time() - debug_start
    # print(num)
    DeltaSigmas=np.add(DeltaSigmas,np.array(sums))
    
    
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
table=np.column_stack((np.array(data2['Ring Radii']),avgDsigma[0]))
np.savetxt(f'{bin}(Mh70).txt', table, delimiter='\t', fmt='%f') #save average delta sigma

fig, axes = plt.subplots(nrows=1, ncols=1)

# Plot the second graph on the right subplot
axes.plot(data2['Ring Radii'],avgDsigma[0])
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
                
        


