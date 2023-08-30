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

def subhalo_profile(r,mass,taus,As):
    # print(mass)
    # print(tau)
    # print(z)
    # print(A)
    
    mass=math.pow(10, mass)
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta=2
    
    
    
    # R = np.linspace(0.01, 1.5, 75)
    
    summed_halos = []
    for i in range(len(taus)):
        tau = math.pow(10, taus[i])
        tnfw = TNFW(mass, c, z, tau, eta)
        dSigma=np.squeeze(tnfw.projected_excess(r))/1000000


        bin=bin_names[i]
        # dSigma=nfw.projected_excess(R)
        halo_table = np.genfromtxt(f'{bin}(Mh70).txt', delimiter='\t', usecols=(0, 1), dtype=float)
        halo_r=halo_table[:,0]/1000
        
        halo_ds=halo_table[:,1]
        
        f = interp1d(halo_r, halo_ds, kind='cubic')
        halo_dSigma=f(r)*As[i]

        summed_halo=np.add(dSigma,halo_dSigma)/1000000
        summed_halos.append(summed_halo)
    
    return summed_halos



    
def log_likelihood(params, r, ys, y_errs):
    mass, tau1, tau2, tau3, A1, A2, A3 = params
    taus=[tau1,tau2,tau3]
    As=[A1,A2,A3]
    model_prediction = subhalo_profile(r, mass, taus, As)
    log_lk=0
    for i in range(len(ys)):
        sigma2 = y_errs[i]**2
        log_lk+=-0.5 * np.sum((ys[i] - model_prediction[i])**2 / sigma2 + np.log(sigma2))
    return log_lk

def log_prior(params):
    mass, tau1, tau2, tau3, A1, A2, A3 = params
    if  A1> 0 and A2> 0 and A3 > 0:
        return 0.0  
    return -np.inf 

def log_probability(params, r, ys, y_errs):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, r, ys, y_errs)

def proposal_function(p0, random):
    

    new_p0 = np.copy(p0)
    for i in range(len(bin_names)):
        new_p0[:, i + 1] += random.normal(0, 0.1, size=nwalkers)  # Update tau
        new_p0[:, i + 4] += random.normal(0, 0.1, size=nwalkers)  # Update A

    return new_p0, 0.0

bin_names = ['0609', '0306', '0103']
ds = []
ds_err=[]
lenses = Table.read("./data/redmapper_mnc_allz.fits")
data_mask = (
        (lenses["R"] >= 0.1)
        & (lenses["R"] < 0.9)
        & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
       
    )
lenses = lenses[data_mask]

z=np.mean(lenses['z_any'])

for bin in bin_names:
    if bin=='0609':
        lowlim=0.6
        highlim=0.9
    elif bin=='0306':
        lowlim=0.3
        highlim=0.6
    elif bin=='0103':
        lowlim=0.1
        highlim=0.3 
    df = pd.read_csv(f'D:/roman_esd_ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
    ds.append(df['ds'])
    rp = df['rp']
    ds_err.append(df['ds_err'])
    
    
nwalkers=100
ndim=7
mass_state = np.random.uniform(12, 13.2, size=nwalkers)
tau_states = np.random.uniform(-1, 5, size=(nwalkers, len(bin_names)))
A_states = np.random.uniform(0.5, 1, size=(nwalkers, len(bin_names)))

initial_positions = np.hstack((mass_state[:, np.newaxis], tau_states, A_states))

MH = [emcee.moves.MHMove(proposal_function)]

# Modify sampling section to use the new likelihood and parameter structure
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(rp, ds, ds_err), moves=MH)

nsteps = 500
sampler.run_mcmc(initial_positions, nsteps, progress=True)
#%%
samples = sampler.get_chain(discard=80, flat=True)  # Discard the first 100 steps as burn-in

best_fit_params = np.median(samples, axis=0)
param_uncertainties = np.std(samples, axis=0)

lens_mass, tau1, tau2, tau3, A1, A2, A3 =best_fit_params

import corner


# Get the samples (chain)


# Extract the parameter names (you can customize them as needed)
param_names = ['log(Mass)', 'log(Tau1)','log(Tau2)' ,'log(Tau3)','A1','A2','A3']

# Plot the corner plot to visualize the parameter space
fig = corner.corner(samples, labels=param_names, truths=best_fit_params, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('bigcorner.png', dpi=800)
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