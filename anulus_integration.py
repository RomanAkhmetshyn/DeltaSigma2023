# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:57:14 2023

@author: Admin
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from colossus.cosmology import cosmology
from colossus.halo import concentration, profile_nfw

from scipy.interpolate import interp1d

from NFW_funcs import quick_MK_profile


params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%%
num=0
lenses = Table.read("./data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits")
data_mask = (
        (lenses["R"] >= 0.3)
        & (lenses["R"] < 0.6)
        & (lenses["PMem"] > 0.8)
    )
lenses = lenses[data_mask]
cdf_resolution=1000
mass_per_point = 1098372.008822474*10000
start_bin=0.01
end_bin=1.5
ring_incr=0.02
ring_num=round((end_bin-start_bin)/ring_incr)
ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000 #Mpc*1000
threshold=ring_incr/2*100

mdef="200m"
# print(lenses['ID'])
DeltaSigmas=np.zeros((1, len(ring_radii)))
debug_start = time.time()
with open('0306big.txt', 'a+') as f:
    np.savetxt(f, [ring_radii[::-1]], fmt='%f', newline='\n')
halo_dict={}
for sat in lenses:
    # t1 = time.time()
    
    if sat['ID'] not in halo_dict:
        halo_dict={}
        random_radii_x, random_radii_y = quick_MK_profile(sat['M_halo'],
                                                          sat['Z_halo'],
                                                          mass_per_point, 
                                                          "duffy08",
                                                          "200m",
                                                          cdf_resolution)
        
        halo_dict[sat['ID']]=[random_radii_x, random_radii_y]
        
    else:
        
        random_radii_x, random_radii_y = halo_dict[sat['ID']]
        
    
    # print(time.time()-t1)
    num+=1
    sat_x = sat['R'] * 1000 #Mpc*1000
    sat_y = 0
    
    S=[np.pi*((r+threshold)**2-(r-threshold)**2) for r in ring_radii]
    # Calculate the distances for all points at once
    distances = np.sqrt((random_radii_x - sat_x)**2 + (random_radii_y - sat_y)**2)
    
    # Create an empty array to store the counts for each ring
    ring_counts = np.zeros(len(ring_radii), dtype=np.int64)
    circle_counts = np.zeros(len(ring_radii), dtype=np.int64)
    
    # Iterate over each ring radius and count the points within each ring
    for i in range(len(ring_radii)):
        mask = np.logical_and(ring_radii[i] - threshold <= distances, distances <= ring_radii[i] + threshold)
        # ring_counts[i] = np.sum(mask)
        
        ring_counts[i] = np.sum(mask)*mass_per_point/S[i]
        
    for i in range(len(ring_radii)):
        mask = np.logical_and(0 <= distances, distances <= ring_radii[i] - threshold)
        # ring_counts[i] = np.sum(mask)
        circle_counts[i] = np.sum(mask)*mass_per_point/(np.pi*(ring_radii[i]- threshold)**2)

    # Create a DataFrame using the ring radii and ring counts
    # data = {'Ring Radii': ring_radii, 'Delta(R)': ring_counts}
    # df = pd.DataFrame(data)
    
    sums=[]

    for i in range(len(ring_radii)-1,-1,-1):
        DeltalessR=circle_counts[i]
        DeltaR=ring_counts[i]
        sums.append(DeltalessR-DeltaR)
    data2 = {'Ring Radii': ring_radii[::-1], 'SigmaDelta(R)': sums}
    with open('0306big.txt', 'a+') as f:
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
avgDsigma=DeltaSigmas/len(lenses)
table=np.column_stack((np.array(data2['Ring Radii']),avgDsigma[0]))
np.savetxt('0306.txt', table, delimiter='\t', fmt='%f')

fig, axes = plt.subplots(nrows=1, ncols=1)

# Plot the second graph on the right subplot
axes.plot(data2['Ring Radii'],avgDsigma[0])
axes.set_xlabel('kpc')
axes.set_ylabel('avg DeltaSigma')


plt.show()
    

        
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
                
        


