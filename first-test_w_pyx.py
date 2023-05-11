# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:49:02 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:28:07 2023

@author: Isaac (copy of his original code)
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

params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%% Load Tinker catalog as lenses

halo_mass_bin = 12.5
halo_mass_bins_end = {12.5: 13.0, 13.0: 13.5, 13.5: 14.0, 14.0: np.inf}
#
lenses = Table.read("./data/Tinker_SDSS.fits")
lenses_mask = (
    (lenses["M_halo"] >= 10**halo_mass_bin)
    & (lenses["M_halo"] < 10 ** halo_mass_bins_end[halo_mass_bin])
    & (lenses["id"] == lenses["igrp"])
)
lenses = lenses[lenses_mask]

# print(lenses)

#%% NFW profile params

test_lens = lenses[0]  # log(M_halo) = 12.96

test_c = concentration.concentration(
    M=test_lens["M_halo"], mdef="200m", z=test_lens["z"], model="duffy08"
)
test_density, test_scale_radius = profile_nfw.NFWProfile.nativeParameters(
    M=test_lens["M_halo"], c=test_c, z=test_lens["z"], mdef="200m"
)
test_virial_radius = test_scale_radius * test_c
# print(test_density, test_scale_radius, test_virial_radius)

#%% Sigma(R) of multiple offset halos

multi_lenses = lenses[10:20]
multi_offsets =  np.empty([1,1], dtype=float)

for lens in multi_lenses:
    tmp_c = concentration.concentration(
        M=lens["M_halo"], mdef="200m", z=lens["z"], model="duffy08"
    )
    tmp_density, tmp_scale_radius = profile_nfw.NFWProfile.nativeParameters(
        M=lens["M_halo"], c=tmp_c, z=lens["z"], mdef="200m"
    )
    tmp_virial_radius = tmp_scale_radius * tmp_c
    multi_offsets=np.vstack([multi_offsets, [0.4 * tmp_virial_radius]])

# print(multi_offsets)
multi_offsets=np.delete(multi_offsets, 0,0)

#%%

multi_mass_per_point = 1098372.008822474
time_start = time.time()
multi_rvals, multi_thetavals = NFW_funcs.sample_nfw(
    masses=np.array(multi_lenses["M_halo"].data).astype('<f8'),
    redshifts=np.array(multi_lenses["z"].data).astype('<f8'),
    mass_per_point=multi_mass_per_point,
    offsets=multi_offsets,
    seeds=None,
    cdf_resolution=1000,
    return_xy=False,
    verbose=True,
)



print(time.time() - time_start, multi_rvals.shape)
multi_xvals = multi_rvals * np.cos(multi_thetavals)
multi_yvals = multi_rvals * np.sin(multi_thetavals)
print(np.mean(multi_xvals), np.mean(multi_yvals))

#%%

with plt.rc_context({"axes.grid": False}):
    fig, ax = plt.subplots(dpi=100)
    img = ax.hexbin(multi_xvals, multi_yvals, gridsize=100, bins="log")
    ax.plot(0, 0, "r+")
    fig.colorbar(img)
    ax.set_aspect("equal")
    ax.set_title(f"{len(multi_lenses)} halos")
    plt.show()
    
#%%

radius_bins = np.linspace(0, 500, 40)  # kpc
radius_bin_centers = 0.5 * (radius_bins[1:] + radius_bins[:-1])  # just for plotting
#
multi_sigmas = (
    np.histogram(multi_rvals, bins=radius_bins, density=False)[0]
    * multi_mass_per_point
    / (np.pi * (radius_bins[1:] ** 2 - radius_bins[:-1] ** 2))
)  # average surface density within annulus
#
fig, ax = plt.subplots(dpi=100)
ax.plot(radius_bin_centers, multi_sigmas, "o-")
ax.set_xlabel(r"$R$ [kpc h$^{-1}$]")
ax.set_ylabel(r"$\Sigma(R)$ [$\rm M_\odot\, kpc^{-2}\, h$]")
plt.show()

#%% DeltaSigma of multiple offset halos

radius_bins = np.linspace(0, 500, 1000)  # kpc

binned_masses = np.histogram(multi_rvals, bins=radius_bins)[0] * multi_mass_per_point
# Average surface density of annulus
multi_sigmas = binned_masses / (np.pi * (radius_bins[1:] ** 2 - radius_bins[:-1] ** 2))
# <https://bdiemer.bitbucket.io/colossus/halo_profile_nfw.html#halo.profile_nfw.NFWProfile.deltaSigma>
integral_vals = []
integrand_vals = 2 * np.pi * radius_bins[1:] * multi_sigmas
integrand_vals = np.insert(integrand_vals, 0, 0.0)
for i in range(multi_sigmas.size):
    integral_vals.append(simpson(y=integrand_vals[: i + 2], x=radius_bins[: i + 2]))
integral_vals = np.array(integral_vals)
multi_dsigmas = integral_vals / (np.pi * radius_bins[1:] ** 2) - multi_sigmas

fig, ax = plt.subplots(dpi=100)
ax.plot(radius_bins[1:], multi_dsigmas, "s", ms=0.5)
ax.set_xlabel(r"$R$ [kpc h$^{-1}$]")
ax.set_ylabel(r"$\Delta\Sigma(R)$ [$\rm M_\odot\, kpc^{-2}\, h$]")
plt.show()