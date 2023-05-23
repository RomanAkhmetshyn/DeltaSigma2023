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

import pickle

params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

#%% Load Tinker catalog as lenses


# lenses = Table.read("D:/GitHub/summer-research/data/dsigma_measurements/output/example_esd_ShapePipe_clusterDist0.6_randomsTrue_1806950.csv")
lenses = Table.read("D:/GitHub/summer-research/data/dr8_redmapper_v6.3.1_members_masked.fits")
# lenses = Table.read("D:/dr8_run_redmapper_v6.3.1_lgt20_catalog_members.fit")
# lenses = Table.read("./data/Tinker_SDSS.fits")
lenses_mask = (
        (lenses["R"] >= 0.6)
        & (lenses["R"] < 0.9)
        & (lenses["ZSPEC"] > -1.0)
    )
lenses = lenses[lenses_mask]
# print(lenses["R"])

# print(lenses.columns)

offsets=lenses["R"]

plt.hist(offsets, bins=30, edgecolor='black')


plt.show()


