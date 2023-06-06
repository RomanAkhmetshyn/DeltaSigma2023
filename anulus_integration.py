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

data = Table.read("./data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits")
data_mask = (
        (data["R"] >= 0.6)
        & (data["R"] < 0.9)
        & (data["ZSPEC"] > -1.0)
    )
data = data[data_mask]
cdf_resolution=1000
mass_per_point = 1098372.008822474

mdef="200m"

for sat in data[0:1]:
    
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
    
    debug_start = time.time()
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

    print(
        "Finished calculating enclosed mass with colossus after",
        time.time() - debug_start,
    )
    #
    # Determine number of points to generate for this halo
    #

    n_points = round(interp_2d_encl_masses[-1:][0] / (mass_per_point))
    print("For each offset, will generate", n_points, "points for this halo")
    #
    # Make 1D interpolator for this halo
    #
    print("Begin creating 2D NFW CDF interpolator")
    debug_start2 = time.time()
    interp_normed_2d_encl_masses = interp1d(
        interp_2d_encl_masses / interp_2d_encl_masses[-1:][0],
        interp_radii,
        assume_sorted=True,
    )

    print(
        "Finished creating 2D NFW CDF interpolator after",
        time.time() - debug_start2,
    )
    print()

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
    print("Begin interpolation")
    debug_start3 = time.time()
    random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)
    print("Finished interpolation after", time.time() - debug_start3)
    random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
    random_radii_x = random_radii * np.cos(random_azimuths) + offset_x
    random_radii_y = random_radii * np.sin(random_azimuths) + offset_y
    print("Begin extending list")
    debug_start4 = time.time()
    #if return_xy:

    #else:
    random_r=np.array([ np.sqrt(random_radii_x**2 + random_radii_y**2)])
    random_theta=np.array([ np.arctan2(random_radii_y, random_radii_x)])
    print("Finished extending list after", time.time() - debug_start4)
    print()





    # sat_x = sat['R'] * 100
    # sat_y = 0
    # ring_radii = np.linspace(0.1, 1.5, 14+1) * 10
    
    # ring_counts = []
    
    # for radius in ring_radii:
    #     count = 0
    #     for x, y in zip(random_radii_x, random_radii_y):
    #         distance = np.sqrt((x - sat_x)**2 + (y - sat_y)**2)
    #         if radius - 0.5 <= distance <= radius + 0.5:
    #             count += 1
    #     ring_counts.append(count)
    
    # # Create a DataFrame using the ring radii and ring counts
    # data = {'Ring Radii': ring_radii, 'Point Counts': ring_counts}
    # df = pd.DataFrame(data)
    
    sat_x = sat['R'] * 100
    sat_y = 0
    ring_radii = np.linspace(0.1, 1.5, 140+1) * 100
    threshold=0.5
    S=[np.pi*((r+threshold)**2-(r-threshold)**2) for r in ring_radii]
    # Calculate the distances for all points at once
    distances = np.sqrt((random_radii_x - sat_x)**2 + (random_radii_y - sat_y)**2)
    
    # Create an empty array to store the counts for each ring
    ring_counts = np.zeros(len(ring_radii), dtype=int)
    circle_counts = np.zeros(len(ring_radii), dtype=int)
    
    # Iterate over each ring radius and count the points within each ring
    for i in range(len(ring_radii)):
        mask = np.logical_and(ring_radii[i] - threshold <= distances, distances <= ring_radii[i] + threshold)
        # ring_counts[i] = np.sum(mask)
        ring_counts[i] = np.sum(mask)*mass_per_point/S[i]
        
    for i in range(len(ring_radii)):
        mask = np.logical_and(0 <= distances, distances <= ring_radii[i] - threshold)
        # ring_counts[i] = np.sum(mask)
        circle_counts[i] = np.sum(mask)*mass_per_point/S[i]

    # Create a DataFrame using the ring radii and ring counts
    data = {'Ring Radii': ring_radii, 'Delta(R)': ring_counts}
    df = pd.DataFrame(data)
    
    sums=[]

    for i in range(len(ring_radii)-1,-1,-1):
        DeltalessR=circle_counts[i]
        DeltaR=ring_counts[i]
        sums.append(DeltalessR-DeltaR)

        
    

        
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
                
        
    # with plt.rc_context({"axes.grid": False}):
    #     fig, ax = plt.subplots(dpi=100)
    #     img = ax.hexbin(random_radii_x, random_radii_y, gridsize=100, bins="log")
    #     ax.plot(0, 0, "r+")
    #     fig.colorbar(img)
    #     ax.scatter(x_coords, y_coords, s=1, c='yellow',zorder=5)
    #     ax.set_aspect("equal")
    #     ax.set_title("1 halos")
    #     plt.show()
    

    
    # distances = np.sqrt((random_radii_x - sat_x)**2 + (random_radii_y - sat_y)**2)
    # ring_radii = np.linspace(0.1, 1.5, 14+1)
    # tolerance = 0.05  # Example tolerance threshold
    # ring_counts = []
    
    # for i in range(len(ring_radii)):
    #     # Calculate the lower and upper boundaries of the tolerance range
    #     lower_bound = ring_radii[i] - tolerance
    #     upper_bound = ring_radii[i] + tolerance
    
    #     # Count the number of points within the tolerance range for the current ring
    #     count_within_tolerance = np.sum(
    #         (distances >= lower_bound) & (distances < upper_bound)
    #     )
        
    #     # Add the count to the ring_counts list
    #     ring_counts.append(count_within_tolerance)


