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

#%% Sigma(R) of multiple offset halos

multi_lenses = lenses[10:11]
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