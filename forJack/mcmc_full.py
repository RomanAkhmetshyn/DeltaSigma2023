# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:11:09 2024

@author: romix
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

np.random.seed(42)
mpl.rcParams['figure.dpi'] = 300

astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.049)
params = {"flat": True, "H0": 70, "Om0": 0.3,
          "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")


def chisquare(y, yfit, err):

    if err.shape == (len(y), len(y)):

        inv_cov = np.linalg.inv(np.matrix(err))
        chi2 = 0
        for i in range(len(y)):
            for j in range(len(y)):
                chi2 = chi2 + (y[i]-yfit[i])*inv_cov[i, j]*(y[j]-yfit[j])
        return chi2

    elif err.shape == (len(y),):
        return sum(((y-yfit)**2.)/(err**2.))
    else:
        raise IOError('error in err or cov_mat input shape')


def subhalo_profile(r, mass, A, scale, show=False):

    mass = math.pow(10, mass)

    concentration_model = "duffy08"
    c = concentration.concentration(
        M=mass, mdef="200m", z=avg_z, model=concentration_model
    )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta = 2
    tnfw = TNFW(mass, c, avg_z, 100, eta, cosmo=astropy_cosmo)

    stellarSigma = M_stellar/(np.pi*r**2)/1000000/1000000

    interpolated_model = np.zeros_like(models[0])
    for i in range(models.shape[1]):
        interp_func = interp1d(
            scales, models[:, i], kind='cubic', fill_value="extrapolate")
        interpolated_model[i] = interp_func(scale)

    halo_r = np.genfromtxt(
        files[0], delimiter='\t', usecols=(0), dtype=float)

    halo_r = halo_r/1000

    halo_ds = interpolated_model
    # halo2_ds=halo_table2[:,1]

    # halo_ds = A*halo1_ds

    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma = f(r) * A

    sat_term = np.squeeze(tnfw.projected_excess(r))/1000000/1000000

    summed_halo = np.add(halo_dSigma, sat_term)

    summed_halo = np.add(summed_halo, stellarSigma)

    if show:
        stellarSigma = M_stellar/(np.pi*halo_r**2)/1000000/1000000
        sat_term = np.squeeze(tnfw.projected_excess(halo_r))/1000000/1000000
        return halo_r, halo_ds * A, stellarSigma, sat_term

    return summed_halo


def log_likelihood(params, r, y, y_err):

    mass, A, scale = params
    model_prediction = subhalo_profile(r, mass, A, scale)

    chi = chisquare(y, model_prediction, y_err)
    return -0.5 * chi


def log_prior(params):
    mass, A, scale = params
    if any(val < 0 for val in [mass, A, scale]) or not (0.0008 <= scale <= 0.085) or mass < 11:
        return -np.inf
    # return -np.inf
    return 0.0


def log_probability(params, r, y, y_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, r, y, y_err)


def proposal_function(p0, random):
    std_devs = np.array([0.05, 0.01, 0.01])
    new_p0 = p0 + random.normal(0, std_devs, size=p0.shape)
    # log_metropolis_ratio = log_ratio_of_proposal_probabilities(new_p0, p0)

    return new_p0, 0.0


bin = '0103'
index = '_rayleigh'

if bin == '0609':
    lowlim = 0.6
    highlim = 0.9

    M_stellar = math.pow(10, 10.94)

elif bin == '0306':
    lowlim = 0.3
    highlim = 0.6

    M_stellar = math.pow(10, 10.92)

elif bin == '0103':
    lowlim = 0.1
    highlim = 0.3

    M_stellar = math.pow(10, 10.86)

data_path = 'C:/scp'
df = pd.read_csv(
    data_path+f'/roman_esd_70ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')

file_pattern = f"{bin}_*_rayleigh.txt"
files = sorted(glob.glob(file_pattern))  #

scales = []  # To store scale values
models = []  # To store the second column of data

for file in files:
    scale = float(file.split('_')[1])
    scales.append(scale)

    data = np.loadtxt(file, usecols=1)
    models.append(data)

scales = np.array(scales)
models = np.array(models)/1000000


# cov=np.loadtxt(f'C:/scp/roman_esd_70ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1_covmat.txt', delimiter=' ', dtype='float64')
cov = np.loadtxt(f'C:/scp/{lowlim}trimmed1.csv', delimiter=',', dtype='float64')
factor = (100-len(cov[0])-2)/(100-1)
factor = 1/factor
cov = cov*factor

if bin == '0609':
    ds = (df['ds']).values
    ds = ds[1:]
    # ds=np.concatenate((ds[1:5], ds[-4:]))
    rp = (df['rp']).values
    rp = rp[1:]
    # rp=np.concatenate((rp[1:5], rp[-4:]))
    ds_err = df['ds_err']
    ds_err = ds_err[1:]

elif bin == '0306':
    ds = (df['ds']).values
    ds = ds[1:]
    # ds=np.concatenate((ds[1:4], ds[8:]))
    rp = (df['rp']).values
    rp = rp[1:]
    # rp=np.concatenate((rp[1:4], rp[8:]))
    ds_err = df['ds_err']
    ds_err = ds_err[1:]
    # ds_err=np.concatenate((ds_err[1:4], ds_err[8:]))*math.sqrt(factor)

elif bin == '0103':
    ds = (df['ds']).values
    ds = ds[1:]
    # ds=np.concatenate((ds[1:2], ds[5:]))
    rp = (df['rp']).values
    rp = rp[1:]
    # rp=np.concatenate((rp[1:2], rp[5:]))
    ds_err = df['ds_err']
    ds_err = ds_err[1:]
    # ds_err=np.concatenate((ds_err[1:2], ds_err[5:]))*math.sqrt(factor)

lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits")
data_mask = (
    (lenses["R"] >= lowlim)
    & (lenses["R"] < highlim)
    & (lenses["PMem"] > 0.8)
)
lenses = lenses[data_mask]

sumz = 0
for i in range(len(lenses)):
    if lenses[i]['zspec'] > -1:
        sumz += lenses[i]['zspec']
    else:
        sumz += lenses[i]['Z_halo']

avg_z = sumz/len(lenses)
del sumz, i

ndim = 3
nwalkers = 50

mass_state = np.random.uniform(11, 13, size=nwalkers)

A_state = np.random.uniform(0, 1, size=nwalkers)

scale_state = np.random.uniform(0.001, 0.02, size=nwalkers)

initial_positions = np.vstack((mass_state, A_state, scale_state)).T

param_names = ['log(M)', 'A', 'scale']

if __name__ == "__main__":
    with Pool(processes=10) as pool:
        MH = [emcee.moves.MHMove(proposal_function)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(rp, ds, cov), moves=MH, pool=pool)

        nsteps = 10000
        sampler.run_mcmc(initial_positions, nsteps, progress=True)

    samples = sampler.get_chain(discard=5000, flat=True)
    np.save(f'{bin}_samples{index}.txt', samples)

    best_fit_params = np.median(samples, axis=0)
    print(best_fit_params)

    best_mass, best_A, best_scale = best_fit_params

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels = [f"{i}" for i in param_names]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(sampler.chain[:, :, i].T, "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step")
    plt.show()

    R, halo, stellar, sathalo = subhalo_profile(
        rp, best_mass, best_A, best_scale, show=True)
    # plt.plot(R, subhalo+stellar+sathalo)
    # plt.plot(R, halo, linestyle='--')
    # plt.plot(R, sathalo, linestyle='--')
    # plt.plot(R, stellar, linestyle='--')
    # plt.errorbar(rp, ds, ds_err, fmt='o')
    # plt.ylim(-20, 120)
    # plt.xlim(0, 2.4)
    # plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(
        18, 12), gridspec_kw={'height_ratios': [2, 1]})
    np.savetxt(f'./results/{bin}_halo{index}.txt', halo)
    np.savetxt(f'./results/{bin}_sub{index}.txt', sathalo)
    np.savetxt(f'./results/{bin}_star{index}.txt', stellar)
    np.savetxt(f'./results/R{index}.txt', R)
    # Plot for the first subplot
    axs[0].plot(R, sathalo, label='satellite')
    axs[0].plot(R, halo, label='halo', linestyle='--')
    axs[0].plot(R, stellar, label='stellar', c='purple')
    axs[0].plot(R, sathalo+halo+stellar, label='combined fit')
    axs[0].errorbar(df['rp'], df['ds'], df['ds_err'], fmt='o',
                    markerfacecolor='none', markeredgecolor='k', markeredgewidth=2, alpha=0.3)
    axs[0].errorbar(rp, ds, ds_err, fmt='o', label='dsigma Data',
                    markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
    axs[0].set_xlabel('R (Mpc)', fontsize=16)
    axs[0].set_ylabel('M/pc^2', fontsize=16)
    axs[0].grid()
    axs[0].set_ylim(-20, 120)
    axs[0].legend()
    # Compute chi-square and median chi-square
    chi = chisquare(ds, subhalo_profile(rp, best_mass, best_A, best_scale), cov)
    # Create title
    title = f'{bin} z: {avg_z:.3f}  \n log(M): {best_mass:.3f}, A: {best_A:.3f}, Rayleigh scale: {best_scale:.4f} \n chi$^2$: {chi:.3f}'
    # title += ('\n' + r'med log(M): {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, A: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$,chi$^2$: {:.3f}'.format(
    # mass_tiles[1], q1[0], q1[1], A_tiles[1], q3[0], q3[1], med_chi))
    axs[0].set_title(title, fontsize=16)

    # Plot for the second subplot
    residual = ds - subhalo_profile(rp, best_mass, best_A, best_scale)
    np.savetxt(f'./results/{bin}_res{index}.txt', residual)
    axs[1].scatter(rp, residual)
    axs[1].axhline(y=0, color='black', linestyle='--',
                   linewidth=1)  # Zero line for reference
    for x, err, res in zip(rp, ds_err, residual):
        axs[1].plot([x, x], [res - err, res + err], color='blue', alpha=0.5)
    axs[1].set_ylabel('residuals', fontsize=16)
    # axs[1].set_title(f'sum of residuals = {}')  # You can uncomment this line if you want to include a title
    plt.tight_layout()  # Adjust spacing between subplots
    plt.legend()
    plt.show()

    import corner
    param_names = ['log(Mass)', 'A', 'scale']

    # Plot the corner plot to visualize the parameter space
    fig = corner.corner(samples, labels=param_names, truths=best_fit_params, quantiles=[
                        0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f')
    plt.show()
