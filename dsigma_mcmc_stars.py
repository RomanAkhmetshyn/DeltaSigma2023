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
from subhalo_profile import make_profile, make_NFW_stars

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


def subhalo_profile(r,mass,A, stellar_mass):
    # print(mass)
    # print(tau)
    # print(z)
    # print(A)
    stellar_mass=math.pow(10, stellar_mass)
    mass=math.pow(10, mass)
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)
    # halo_profile = profile_nfw.NFWProfile(M=mass, c=c, z=z, mdef="200m")
    halo_profile=NFW(mass, c, z)
    # eta=2
    # tnfw = TNFW(mass, c, z, tau, eta)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    # dSigma=np.squeeze(tnfw.projected_excess(r))/1000000
    # dSigma=halo_profile.deltaSigma(r*1000)
    dSigma= np.squeeze(halo_profile.projected_excess(r))/1000000
    
    starSigma=stellar_mass/(math.pi*r**2)/1000000
    # dSigma=nfw.projected_excess(R)
    halo_table = np.genfromtxt(f'{bin}(Mh70).txt', delimiter='\t', usecols=(0, 1), dtype=float)
    halo_r=halo_table[:,0]/1000
    
    halo_ds=halo_table[:,1]
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)*A

    summed_halo=np.add(dSigma,halo_dSigma)
    summed_halo=np.add(summed_halo,starSigma)/1000000

    return summed_halo



    
def Gaussian(params, r, y, y_err):
    mass, A, stellar_mass= params
    model_prediction = subhalo_profile(r, mass, A, stellar_mass)
    sigma2 = y_err**2
    return -0.5 * np.sum((y - model_prediction)**2 / sigma2 + np.log(sigma2))

def log_prior(params):
    mass, A, stellar_mass = params
    if mass > 10 and A > 0:
        return 0.0  
    return -np.inf 

def log_probability(params, r, y, y_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + Gaussian(params, r, y, y_err)

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
A_state=np.random.uniform(0.5, 1, size=nwalkers)
stellar_mass_state=np.random.uniform(9, 10.5, size=nwalkers)

initial_positions = np.vstack((mass_state, A_state, stellar_mass_state)).T

def proposal_function(p0, random):
    new_p0 = p0 + random.normal(0, 0.1, size=p0.shape)
    return new_p0, 0.0  # The second value is the log probability, set to 0 for symmetric proposal

MH=[emcee.moves.MHMove(proposal_function)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(rp, ds, ds_err), moves=MH)

nsteps = 500
sampler.run_mcmc(initial_positions, nsteps, progress=True)


samples = sampler.get_chain(flat=True)

best_fit_params = np.median(samples, axis=0)
param_uncertainties = np.std(samples, axis=0)

lens_mass, param_A, lens_star =best_fit_params

print("Best-fit parameters:")
print("Mass:", lens_mass)

print("A:", param_A)
lens_z=z

print("Parameter uncertainties:")
print("Mass uncertainty:", param_uncertainties[0])
# print("Tau uncertainty:", param_uncertainties[1])
print("A uncertainty:", param_uncertainties[1])

fit = subhalo_profile(rp, lens_mass, param_A, lens_star)
residual=np.subtract(fit,ds) 
residual_sq=[x**2 for x in residual] #get square of residuals
chi2=np.sum(np.array(residual_sq)/np.array(ds)) #chi^2

r_full,ds_host,ds_sub, star=make_NFW_stars(math.pow(10,lens_mass), lens_z, param_A, math.pow(10,lens_star) ,distbin=bin, plot=False)
# r_full,ds_full=make_profile(1e12, 0.35, 35, 0.6, distbin=bin, plot=False)
# plt.plot(rp, ds, 'bo', label='Isaac Data')
# plt.plot(rp, fit, 'r-', label='interp Curve')
plt.plot(r_full,ds_host, label='halo', linestyle='--', color='orange')
plt.plot(r_full,ds_sub,label='subhalo',  linestyle='--', color='green')
plt.plot(r_full,star,label='star',  linestyle='--', color='pink')
plt.plot(r_full, ds_host+ds_sub, 'k-', label='fitted Curve')
plt.errorbar(rp, ds, ds_err, fmt='o',label='dsigma Data', color='tab:blue')
plt.xlabel('R (Mpc)')
plt.ylabel('M/pc^2')
plt.grid()
# plt.ylim(-40,120)
plt.title(f'{bin} lens log(mass): {lens_mass:.2f}, Z: {lens_z:.2}, A: {param_A:.2}, stellar mass: {lens_star:.2f}')
plt.legend()
plt.show()
plt.close()

import corner


# Get the samples (chain)
samples = sampler.get_chain(discard=100, flat=True)  # Discard the first 100 steps as burn-in

# Extract the parameter names (you can customize them as needed)
param_names = ['log(Mass)', 'A', 'log(StMass)']

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

# chain_file_path = f"{bin}sampler_chain_stellar.txt"
# np.savetxt(chain_file_path, sampler.chain.reshape(-1, ndim), header=" ".join(param_names), fmt='%f')

# Save samples to a text file
# samples_file_path = f"{bin}samples.txt"
# np.savetxt(samples_file_path, samples, header=" ".join(param_names), fmt='%f')
