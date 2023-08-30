# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:18:03 2023

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

lenses = Table.read("./data/redmapper_mnc_allz.fits")
data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
       
    )
lenses = lenses[data_mask]

z=np.mean(lenses['z_any'])


def subhalo_profile(r,mass,tau,A):
    # print(mass)
    # print(tau)
    # print(z)
    # print(A)
    tau=math.pow(10,tau)
    mass=math.pow(10, mass)
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta=2
    tnfw = TNFW(mass, c, z, tau, eta)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    dSigma=np.squeeze(tnfw.projected_excess(r))/1000000


    
    # dSigma=nfw.projected_excess(R)
    halo_table = np.genfromtxt(f'{bin}(Mh70).txt', delimiter='\t', usecols=(0, 1), dtype=float)
    halo_r=halo_table[:,0]/1000
    
    halo_ds=halo_table[:,1]
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)*A

    summed_halo=np.add(dSigma,halo_dSigma)/1000000

    return summed_halo



    
def log_likelihood(params, r, y, y_err):
    
    mass, tau, A = params
    model_prediction = subhalo_profile(r, mass, tau, A)
    sigma2 = y_err**2
    return -0.5 * np.sum((y - model_prediction)**2 / sigma2 + np.log(sigma2))

def log_prior(params):
    mass, tau, A = params
    if  A > 0:
        return 0.0  
    return -np.inf 

def log_probability(params, r, y, y_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, r, y, y_err)

# Read the CSV file
# df = pd.read_csv(f'D:/GitHub/summer-research/output-roman(correct)/roman_esd_ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
df = pd.read_csv(f'D:/roman_esd_ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
# df=pd.read_csv(f'D:/GitHub/summer-research/output/{bin}.txt',
#                 delim_whitespace = True, 
#                 names = ['rp','ds','ds_err'], 
#                 comment = '#')

# Save the "ds" and "rp" columns as variables
ds = df['ds']
rp = df['rp']
ds_err=df['ds_err']

ndim = 3
nwalkers = 100

mass_state=np.random.uniform(12, 13.2, size=nwalkers)
tau_state=np.random.uniform(-1, 5, size=nwalkers)
A_state=np.random.uniform(0.5, 1, size=nwalkers)

initial_positions = np.vstack((mass_state, tau_state, A_state)).T

# def log_ratio_of_proposal_probabilities(proposed_params, current_params):
#     # Calculate the proposal probabilities for the proposed and current parameters
    
#     # Calculate the Metropolis ratio
#     log_metropolis_ratio = log_likelihood(proposed_params[0], rp, ds, ds_err) + log_prior(proposed_params[0]) \
#                            - log_likelihood(current_params[0], rp, ds, ds_err) - log_prior(current_params[0])
    
#     return log_metropolis_ratio

def proposal_function(p0, random):
    
    new_p0 = p0 + random.normal(0, 0.1, size=p0.shape)
    # log_metropolis_ratio = log_ratio_of_proposal_probabilities(new_p0, p0)
    
    return new_p0, 0.0
MH=[emcee.moves.MHMove(proposal_function)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(rp, ds, ds_err), moves=MH)

nsteps =5000
sampler.run_mcmc(initial_positions, nsteps, progress=True)


samples = sampler.get_chain(flat=True)

best_fit_params = np.median(samples, axis=0)
param_uncertainties = np.std(samples, axis=0)

lens_mass, lens_tau, param_A =best_fit_params

print("Best-fit parameters:")
print("Mass:", lens_mass)
print("Tau:", lens_tau)
print("A:", param_A)
lens_z=z

print("Parameter uncertainties:")
print("Mass uncertainty:", param_uncertainties[0])
print("Tau uncertainty:", param_uncertainties[1])
print("A uncertainty:", param_uncertainties[2])

# fit = subhalo_profile(rp, lens_mass, lens_tau, param_A)

r_full,ds_host,ds_sub=make_profile(math.pow(10,lens_mass), lens_z, math.pow(10,lens_tau), param_A, B=1 ,distbin=bin, plot=False)
# r_full,ds_full=make_profile(1e12, 0.35, 35, 0.6, distbin=bin, plot=False)
# plt.plot(rp, ds, 'bo', label='Isaac Data')
# plt.plot(rp, fit, 'r-', label='interp Curve')
plt.plot(r_full,ds_host, label='halo', linestyle='--', color='orange')
plt.plot(r_full,ds_sub,label='subhalo',  linestyle='--', color='green')
plt.plot(r_full, ds_host+ds_sub, 'k-', label='fitted Curve')
plt.errorbar(rp, ds, ds_err, fmt='o',label='dsigma Data', color='tab:blue')
plt.xlabel('R (Mpc)')
plt.ylabel('M/pc^2')
plt.grid()
# plt.ylim(-40,120)
plt.title(f'{bin} lens log(mass): {lens_mass:.2f}, Z: {lens_z:.2}, Rt/Rs: {lens_tau:.2f}, A: {param_A:.2}')
plt.legend()
plt.show()
plt.close()

import corner


# Get the samples (chain)
samples = sampler.get_chain(discard=1000, flat=True)  # Discard the first 100 steps as burn-in

# Extract the parameter names (you can customize them as needed)
param_names = ['log(Mass)', 'log(Tau)', 'A']

# Plot the corner plot to visualize the parameter space
fig = corner.corner(samples, labels=param_names, truths=best_fit_params, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.show()

# Plot the trace plots to see the evolution of the walkers
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = [f"{i}" for i in param_names]
for i in range(ndim):
    ax = axes[i]
    ax.plot(sampler.chain[:, :, i].T, "k", alpha=0.3)
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step")
plt.show()

chain_file_path = f"{bin}sampler_chain(binning).txt"
np.savetxt(chain_file_path, sampler.get_chain( flat=True).reshape(-1, ndim), header=" ".join(param_names), fmt='%f')

# # # Save samples to a text file
# samples_file_path = f"{bin}samples(limit tau10s).txt"
# np.savetxt(samples_file_path, samples, header=" ".join(param_names), fmt='%f')


log_prob=sampler.lnprobability
np.savetxt(f'{bin}chain_logprob.txt', log_prob, delimiter=',',fmt='%f')
np.save(f'{bin}chain_logprob.npy', log_prob)
# chain=sampler.get_chain()
# for step, log_prob_values  in enumerate(log_prob):
#     print(f"Step {step}: Log Probability values = {log_prob_values}")
flat_prob=sampler.flatlnprobability
np.savetxt(f'{bin}chain_flatlogprob.txt', flat_prob, delimiter=',',fmt='%f')
#%%
full_chain=sampler.get_chain( flat=True).reshape(-1, ndim)
first_column = full_chain[:, 0]

# Calculate statistics
mean = np.mean(first_column)
median = np.median(first_column)
std_dev = np.std(first_column)
min_val = np.min(first_column)
max_val = np.max(first_column)
from scipy import stats
mode = stats.mode(first_column)[0][0]
print(f"Mode: {mode}")
# Plot a histogram
plt.hist(first_column, bins=650, edgecolor='black')
plt.title('Histogram of First Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
plt.close()

# Print statistics
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Minimum Value: {min_val}")
print(f"Maximum Value: {max_val}")

chainprob = np.concatenate((full_chain, flat_prob.reshape(-1, 1)), axis=1)
mass_sort_ind = np.argsort(chainprob[:, 0])

# Sort the array based on the sorted indices
sorted_chainprob = chainprob[mass_sort_ind]
np.savetxt(f'{bin}sorted_chain_logprob.txt', sorted_chainprob, delimiter=',',fmt='%f')
#%%
# Define the mass range
num_bins=650
min_mass = np.min(sorted_chainprob[:, 0])
max_mass = np.max(sorted_chainprob[:, 0])
mass_bins = np.linspace(min_mass, max_mass, num_bins)

# Use np.digitize to assign each sample to a bin
mass_bin_indices = np.digitize(sorted_chainprob[:, 0], mass_bins)

# min_log_probs_dict = {}

# # Iterate through each sample and update the minimum log_probability for its bin
# for i in range(len(sorted_chainprob)):
#     bin_index = mass_bin_indices[i] - 1  # -1 because bin indices start from 1
#     log_prb = sorted_chainprob[i, 3]  # Assuming the log_probability is in the fourth column
    
#     # Check if the bin index is already in the dictionary
#     if bin_index not in min_log_probs_dict:
#         min_log_probs_dict[bin_index] = log_prb
#     else:
#         min_log_probs_dict[bin_index] = min(min_log_probs_dict[bin_index], log_prb)

# # Now min_log_probs_dict contains the minimum log_probability for each bin
# for bin_index, min_log_prob in min_log_probs_dict.items():
#     print(f"Bin {bin_index + 1}: Minimum Log Probability = {min_log_prob}")

# Create a dictionary to store information for each bin
bin_info_dict = {}

# Iterate through each sample and update the bin information
for i in range(len(sorted_chainprob)):
    bin_index = mass_bin_indices[i] - 1  # -1 because bin indices start from 1
    log_prb = sorted_chainprob[i, 3]  # Assuming the log_probability is in the fourth column
    tau = sorted_chainprob[i, 1]       # Assuming Tau is in the second column
    a = sorted_chainprob[i, 2]         # Assuming A is in the third column
    
    if bin_index not in bin_info_dict:
        bin_info_dict[bin_index] = {
           'mass_values': [],
           'tau_values': [],
           'a_values': [],
           'log_prob_values': [],
           'min_log_prob': log_prb,
           'min_log_prob_tau': tau,
           'min_log_prob_a': a
       }
    else:
        # bin_info_dict[bin_index]['min_log_prob'] = max(bin_info_dict[bin_index]['min_log_prob'], log_prb)
        if log_prb > bin_info_dict[bin_index]['min_log_prob']:
            bin_info_dict[bin_index]['min_log_prob'] = log_prb
            bin_info_dict[bin_index]['min_log_prob_tau'] = tau
            bin_info_dict[bin_index]['min_log_prob_a'] = a
    
    bin_info_dict[bin_index]['mass_values'].append(sorted_chainprob[i, 0])
    bin_info_dict[bin_index]['tau_values'].append(tau)
    bin_info_dict[bin_index]['a_values'].append(a)
    bin_info_dict[bin_index]['log_prob_values'].append(log_prb)
    
#%%

average_mass_per_bin = []
for bin_index, bin_info in bin_info_dict.items():
    average_mass = np.mean(bin_info['mass_values'])
    average_mass_per_bin.append(average_mass)

# Extract minimum log prob values for plot
min_log_prob_values = [bin_info['min_log_prob'] for bin_info in bin_info_dict.values()]

# Create a plot
plt.plot(average_mass_per_bin, min_log_prob_values, marker='.', linestyle='', markersize=5)
plt.xlabel('Average Mass per Bin')
plt.ylabel('Minimum Log Probability')
plt.title('Minimum Log Probability vs Average Mass per Bin')
plt.ylim(-70,-45)
plt.xlim(11.8,13.0)
plt.grid(True)
plt.show()
plt.close()

#%%

