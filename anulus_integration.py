# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:57:14 2023

@author: Admin
"""

import time
import warnings
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from colossus.cosmology import cosmology
from colossus.halo import concentration, profile_nfw
from scipy.integrate import simpson
from scipy.interpolate import interp1d

import NFW_funcs

import seaborn as sns
import pandas as pd

import NFW_funcs
from scipy import interpolate

params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%%

lenses = Table.read("./data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits")
data_mask = (
        (lenses["R"] >= 0.6)
        & (lenses["R"] < 0.9)
        & (lenses["zspec"] > -1.0)
    )
lenses = lenses[data_mask]
cdf_resolution=1000
mass_per_point = 1098372.008822474*5000
start_bin=0.01
end_bin=1.5
ring_incr=0.02
ring_num=round((end_bin-start_bin)/ring_incr)
ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000
threshold=ring_incr/2*100

mdef="200m"

DeltaSigmas=np.empty((0, len(ring_radii)))

for sat in lenses[0:10]:
    debug_start = time.time()
    c=concentration.concentration(
        M=sat["M_halo"], mdef="200m", z=sat["Z_halo"], model="duffy08"
    )
    halo_profile = profile_nfw.NFWProfile(M=sat["M_halo"], c=c, z=sat["Z_halo"], mdef=mdef)

    central_density, scale_radius = halo_profile.getParameterArray()
    virial_radius = scale_radius * c
    #
    # Determine CDF of projected (2D) NFW enclosed mass
    #
    interp_radii = np.linspace(0, virial_radius, cdf_resolution)
    
    # debug_start = time.time()
    # Temporarily ignore division by zero and overflow warnings
    with np.errstate(divide="ignore", over="ignore"):
        interp_delta_sigmas = halo_profile.deltaSigma(interp_radii)
        interp_surface_densities = halo_profile.surfaceDensity(interp_radii)
    # Correct delta sigmas and surface densities at r=0 to be zero
    interp_delta_sigmas[0] = 0.0
    interp_surface_densities[0] = 0.0
    interp_2d_encl_masses = (
        np.pi * interp_radii**2 * (interp_delta_sigmas + interp_surface_densities)
    )

    # print(
    #     "Finished calculating enclosed mass with colossus after",
    #     time.time() - debug_start,
    # )
    #
    # Determine number of points to generate for this halo
    #

    n_points = round(interp_2d_encl_masses[-1:][0] / (mass_per_point))
    print("For each offset, will generate", n_points, "points for this halo")
    #
    # Make 1D interpolator for this halo
    #
    # print("Begin creating 2D NFW CDF interpolator")
    # debug_start2 = time.time()
    interp_normed_2d_encl_masses = interp1d(
        interp_2d_encl_masses / interp_2d_encl_masses[-1:][0],
        interp_radii,
        assume_sorted=True,
    )

    # print(
    #     "Finished creating 2D NFW CDF interpolator after",
    #     time.time() - debug_start2,
    # )
    # print()

    #
    # Generate random points for this halo + offset combination
    #
    rng = np.random.default_rng()
    offset=0
    offset_angle = rng.uniform(0, 2 * np.pi)
    offset_x = offset * np.cos(offset_angle)
    offset_y = offset * np.sin(offset_angle)
    #
    random_cdf_yvals = rng.uniform(0, 1, size=n_points)
    # print("Begin interpolation")
    # debug_start3 = time.time()
    random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)
    # print("Finished interpolation after", time.time() - debug_start3)
    random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
    random_radii_x = random_radii * np.cos(random_azimuths) + offset_x
    random_radii_y = random_radii * np.sin(random_azimuths) + offset_y
    # print("Begin extending list")
    # debug_start4 = time.time()
    #if return_xy:

    #else:
    random_r=np.array([ np.sqrt(random_radii_x**2 + random_radii_y**2)])
    random_theta=np.array([ np.arctan2(random_radii_y, random_radii_x)])
    # print("Finished extending list after", time.time() - debug_start4)
    print()

    
    sat_x = sat['R'] * 1000
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
    data = {'Ring Radii': ring_radii, 'Delta(R)': ring_counts}
    # df = pd.DataFrame(data)
    
    sums=[]

    for i in range(len(ring_radii)-1,-1,-1):
        DeltalessR=circle_counts[i]
        DeltaR=ring_counts[i]
        sums.append(DeltalessR-DeltaR)
    data2 = {'Ring Radii': ring_radii[::-1], 'SigmaDelta(R)': sums}
    t=time.time() - debug_start
    print(
        "Finished calculating 1 sat after",
        t,
    )
    DeltaSigmas=np.append(DeltaSigmas, [np.array(sums)],axis=0)
    
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


avgDsigma=np.mean(DeltaSigmas,axis=0)

fig, axes = plt.subplots(nrows=1, ncols=1)

# Plot the second graph on the right subplot
axes.plot(data2['Ring Radii'],avgDsigma)
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
                
        


