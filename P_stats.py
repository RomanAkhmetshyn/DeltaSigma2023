# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:50:17 2023

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

#%%

lenses = Table.read("./data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits")
data_mask = (
        (lenses["R"] >= 0.6)
        & (lenses["R"] <= 0.9)
        & (lenses["zspec"] > -1.0)
    )
lenses = lenses[data_mask]

import matplotlib.pyplot as plt

# Sample data (distance and probability)
distance = lenses['R']
probability = lenses['PMem']

# Plotting the histogram
plt.hist2d(distance, probability, bins=100, cmap='bone_r')

# Adding labels and title
plt.xlabel('Distance (Mpc)')
plt.ylabel('Pmem')
plt.title('Distance(0.6-0.9) vs Probability (zspec>-1) ')

# Adding a colorbar
plt.colorbar()

# Displaying the plot
plt.show()
