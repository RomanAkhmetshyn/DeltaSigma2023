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

import NFW_funcs
from scipy import interpolate

params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%% Load Tinker catalog as lenses


# lenses = Table.read("D:/GitHub/summer-research/data/dsigma_measurements/output/example_esd_ShapePipe_clusterDist0.6_randomsTrue_1806950.csv")
# lenses = Table.read("D:/GitHub/summer-research/data/dr8_redmapper_v6.3.1_members_masked.fits")
# lenses = Table.read("D:/dr8_run_redmapper_v6.3.1_lgt20_catalog_members.fit")
lenses = Table.read("./data/Tinker_SDSS.fits")
# lenses_mask = (
#         (lenses["R"] >= 0.6)
#         & (lenses["R"] < 0.9)
#         & (lenses["ZSPEC"] > -1.0)
#     )
# lenses = lenses[lenses_mask]

halo_mass_bin = 12.5
halo_mass_bins_end = {12.5: 13.0, 13.0: 13.5, 13.5: 14.0, 14.0: np.inf}
lenses_mask = (
    (lenses["M_halo"] >= 10**halo_mass_bin)
    & (lenses["M_halo"] < 10 ** halo_mass_bins_end[halo_mass_bin])
    & (lenses["id"] == lenses["igrp"])
)
lenses = lenses[lenses_mask]



lense=Table(lenses[0])
halo_c = concentration.concentration(
    M=lense["M_halo"], mdef="200m", z=lense["z"], model="duffy08"
)
halo_density, halo_scale_radius = profile_nfw.NFWProfile.nativeParameters(
    M=lense["M_halo"], c=halo_c, z=lense["z"], mdef="200m"
)
halo_virial_radius = halo_scale_radius * halo_c

sats = Table.read("D:/GitHub/summer-research/data/dr8_redmapper_v6.3.1_members_masked.fits")
sats_mask = (
        (sats["R"] >= 0.6)
        & (sats["R"] < 0.9)
        & (sats["ZSPEC"] > -1.0)
    )
sats = sats[sats_mask]

offsets=sats["R"]

# plt.hist(offsets, bins=30, edgecolor='black')


# plt.show()
multi_offsets =  np.empty([1,1], dtype=float)

multi_offsets=np.vstack([multi_offsets, [0]])

# print(multi_offsets)
multi_offsets=np.delete(multi_offsets, 0,0)

# multi_mass_per_point = 1098372.008822474
multi_mass_per_point = 1098372.008822474*10
time_start = time.time()
multi_rvals, multi_thetavals = NFW_funcs.sample_nfw(
    masses=np.array(lense["M_halo"].data).astype('<f8'),
    redshifts=np.array(lense["z"].data).astype('<f8'),
    mass_per_point=multi_mass_per_point,
    offsets=multi_offsets,
    seeds=None,
    cdf_resolution=1000,
    return_xy=False,
    verbose=True,
)

multi_xvals = multi_rvals * np.cos(multi_thetavals)
multi_yvals = multi_rvals * np.sin(multi_thetavals)

radius=np.amax(multi_rvals)

#single example
# x_coords=[]
# y_coords=[]
# for x,y in zip(multi_xvals,multi_yvals):
#     distance = np.linalg.norm(np.array((x,y)) - np.array((offsets[0]*radius,0)))
#     if radius - 0.5 <= distance <= radius + 0.5:
#         x_coords.append(x)
#         y_coords.append(y)


#Multiple example
for offset in offsets:
    print(offset)
    distances = []
    x_coords=[]
    y_coords=[]
    for x,y in zip(multi_xvals,multi_yvals):
        distance = np.linalg.norm(np.array((x,y)) - np.array((offset*radius,0)))
        if radius - 0.5 <= distance <= radius + 0.5:
            x_coords.append(x)
            y_coords.append(y)
            
            # for x,y in zip(x_coords, y_coords):
            #     distance = np.linalg.norm(np.array((x,y)) - np.array((0,0)))
            #     distances.append(distance)
                
    # hist, edges = np.histogram(distances,bins=100)

    # Plot the histogram as a distribution
    # plt.bar(edges[:-1], hist, width=np.diff(edges), align='edge', color='blue', alpha=0.7)
    # plt.xlabel("Distance from Plane Center (0,0)")
    # plt.ylabel("Density")
    # plt.title(str(offset))
    # plt.show()
        


# with plt.rc_context({"axes.grid": False}):
#     fig, ax = plt.subplots(dpi=100)
#     img = ax.hexbin(multi_xvals, multi_yvals, gridsize=100, bins="log")
#     ax.plot(0, 0, "r+")
#     ax.plot(offsets*radius,np.zeros_like(offsets), marker='.', markersize=1, color='red')
#     # for center_x, center_y in zip(offsets*radius, np.zeros_like(offsets)):
#     #     circle = plt.Circle((center_x, center_y), np.amax(multi_rvals), edgecolor='red', facecolor='none')
#     #     ax.add_patch(circle)
#     circle = plt.Circle((offsets[0]*radius, 0), np.amax(multi_rvals), edgecolor='red', facecolor='none')
#     ax.add_patch(circle)
#     ax.scatter(x_coords, y_coords, s=1, c='yellow',zorder=5)
#     ax.set_ylim(-radius, radius)
#     ax.set_xlim(-radius, radius)
#     fig.colorbar(img)
#     ax.set_aspect("equal")
#     # ax.set_title(f"{len(multi_lenses)} halos")
#     plt.show()
    


# Create a histogram of distances

# hist, edges = np.histogram(distances,bins=100)

# # Plot the histogram as a distribution
# plt.bar(edges[:-1], hist, width=np.diff(edges), align='edge', color='blue', alpha=0.7)
# plt.xlabel("Distance from Plane Center (0,0)")
# plt.ylabel("Density")
# plt.title("Density Distribution of Overlapping Points")
# plt.show()





