# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:11:39 2024

@author: romix
"""

import emcee
import numpy as np
import math
import pandas as pd
from profiley.nfw import TNFW
import matplotlib.pyplot as plt
from colossus.halo import concentration
from colossus.cosmology import cosmology
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
from multiprocessing import Pool
import matplotlib as mpl
import os

mpl.rcParams['figure.dpi'] = 300

profile_path='C:/Users/romix/Documents/GitHub/DeltaSigma2023/MonteCarlo_offset_profile/new-test'

df1 = pd.read_csv(profile_path+f'/roman_esd_ShapePipe_redmapper_clusterDist0.1_randomsTrue_1.csv')
df2 = pd.read_csv(profile_path+f'/roman_esd_ShapePipe_redmapper_clusterDist0.3_randomsTrue_1.csv')
df3 = pd.read_csv(profile_path+f'/roman_esd_ShapePipe_redmapper_clusterDist0.6_randomsTrue_1.csv')

ds1 = df1['ds'].values
rp1 = df1['rp']
ds_err1 = df1['ds_err']

ds2 = df2['ds'].values
rp2 = df2['rp']
ds_err2 = df2['ds_err']

ds3 = df3['ds'].values
rp3 = df3['rp']
ds_err3 = df3['ds_err']

fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Plot for df1
axs[0].errorbar(rp1, ds1, ds_err1, fmt='o', markeredgewidth=2, alpha=0.6)
axs[0].set_title('0.1-0.3 Mpc')
axs[0].set_xlabel('R (Mpc)', fontsize=16)
axs[0].set_ylabel(r'M$_{sol}$/pc$^2$', fontsize=16)

# Plot for df2
axs[1].errorbar(rp2, ds2, ds_err2, fmt='o', markeredgewidth=2, alpha=0.6)
axs[1].set_title('0.3-0.6 Mpc')
axs[1].set_xlabel('R (Mpc)', fontsize=16)
axs[1].set_ylabel(r'M$_{sol}$/pc$^2$', fontsize=16)

# Plot for df3
axs[2].errorbar(rp3, ds3, ds_err3, fmt='o', markeredgewidth=2, alpha=0.6)
axs[2].set_title('0.6-0.9 Mpc')
axs[2].set_xlabel('R (Mpc)', fontsize=16)
axs[2].set_ylabel(r'M$_{sol}$/pc$^2$', fontsize=16)

plt.tight_layout()

plt.show()