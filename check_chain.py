# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:42:41 2023

@author: Admin
"""

import emcee
import numpy as np
import math
import pandas as pd
from profiley.nfw import TNFW, NFW
import matplotlib.pyplot as plt
from colossus.halo import concentration, profile_nfw
from colossus.cosmology import cosmology
from astropy.table import Table
from scipy.interpolate import interp1d
from subhalo_profile import make_profile
import corner

bin='0609'

if bin=='0609':
    lowlim=0.6
    highlim=0.9
elif bin=='0306':
    lowlim=0.3
    highlim=0.6
elif bin=='0103':
    lowlim=0.1
    highlim=0.3 
    
chainfile = f'{bin}sampler_chain(binning).txt'
full_chain= np.genfromtxt(chainfile, delimiter=' ', dtype=float, skip_header=1)

burnin=1000
ndim=3
steps=5000
walkers=100
bins=100
new_shape = (walkers, steps, ndim)
nonflat_samples = np.empty(new_shape)


# Reshape the original array into the 3D shape using loops
for i in range(new_shape[0]):
    nonflat_samples[i] = full_chain[i::new_shape[0]]
    
burnedsamples = np.empty((walkers, steps-burnin, ndim))
for i in range(burnedsamples.shape[0]):
    burnedsamples[i] = nonflat_samples[i, burnin:, :]

# Flatten the modified 3D array
samples = burnedsamples.reshape(-1, 3)
best_fit_params = np.median(samples, axis=0)

# Extract the parameter names (you can customize them as needed)
param_names = ['log(Mass)', 'log(Tau)', 'A']

# Plot the corner plot to visualize the parameter space
fig = corner.corner(samples, labels=param_names, truths=best_fit_params, quantiles=[0.16, 0.5, 0.84], show_titles=True)
# plt.savefig('corner0103.png',dpi=800)
plt.show()

# Plot the trace plots to see the evolution of the walkers
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = [f"{i}" for i in param_names]
for i in range(ndim):
    ax = axes[i]
    ax.plot(nonflat_samples[:, :, i].T, "k", alpha=0.3)
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step")
plt.show()

#%%

plt.hist(samples[:,1], bins=bins, edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
plt.close()


#%%
sorted_chainprob=np.genfromtxt(f'{bin}sorted_chain_logprob.txt', delimiter=',', dtype=float)

num_bins=bins
param_index=1
min_mass = np.min(sorted_chainprob[:, param_index])
max_mass = np.max(sorted_chainprob[:, param_index])
mass_bins = np.linspace(min_mass, max_mass, num_bins)

# Use np.digitize to assign each sample to a bin
mass_bin_indices = np.digitize(sorted_chainprob[:, param_index], mass_bins)

bin_info_dict = {}

# Iterate through each sample and update the bin information
for i in range(len(sorted_chainprob)):
    bin_index = mass_bin_indices[i] - 1  # -1 because bin indices start from 1
    log_prb = sorted_chainprob[i, 3]  # Assuming the log_probability is in the fourth column
    # tau = sorted_chainprob[i, 1]       # Assuming Tau is in the second column
    mass = sorted_chainprob[i, 0]       # Assuming Tau is in the second column
    a = sorted_chainprob[i, 2]         # Assuming A is in the third column
    
    if bin_index not in bin_info_dict:
        bin_info_dict[bin_index] = {
           'mass_values': [],
           'tau_values': [],
           'a_values': [],
           'log_prob_values': [],
           'min_log_prob': log_prb,
           # 'min_log_prob_tau': tau,
           'min_log_prob_mass': mass,
           'min_log_prob_a': a
       }
    else:
        # bin_info_dict[bin_index]['min_log_prob'] = max(bin_info_dict[bin_index]['min_log_prob'], log_prb)
        if log_prb > bin_info_dict[bin_index]['min_log_prob']:
            bin_info_dict[bin_index]['min_log_prob'] = log_prb
            # bin_info_dict[bin_index]['min_log_prob_tau'] = tau
            bin_info_dict[bin_index]['min_log_prob_mass'] = mass
            bin_info_dict[bin_index]['min_log_prob_a'] = a
    
    # bin_info_dict[bin_index]['mass_values'].append(sorted_chainprob[i, 0])
    # bin_info_dict[bin_index]['tau_values'].append(tau)
    bin_info_dict[bin_index]['tau_values'].append(sorted_chainprob[i, param_index])
    bin_info_dict[bin_index]['mass_values'].append(mass)
    bin_info_dict[bin_index]['a_values'].append(a)
    bin_info_dict[bin_index]['log_prob_values'].append(log_prb)
    
average_mass_per_bin = []
for bin_index, bin_info in bin_info_dict.items():
    # average_mass = np.mean(bin_info['mass_values'])
    average_mass = np.mean(bin_info['tau_values'])
    average_mass_per_bin.append(average_mass)

# Extract minimum log prob values for plot
min_log_prob_values = [bin_info['min_log_prob'] for bin_info in bin_info_dict.values()]
min_log_prob_values=np.exp(min_log_prob_values)
# Create a plot


plt.plot(average_mass_per_bin, min_log_prob_values, marker='.', linestyle='', markersize=5)
plt.xlabel('Average Tau per Bin')
plt.ylabel('Probability')
plt.title(f'{bin}')

# plt.ylim(-70,-45)
# plt.xlim(11.8,13.0)
plt.grid(True)
plt.show()
plt.close()

#%%
