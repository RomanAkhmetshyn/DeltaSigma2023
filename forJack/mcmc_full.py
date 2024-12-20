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


def subhalo_profile(r, A, scale, show=False):

    stellarSigma = M_stellar/(np.pi*r**2)/1000000/1000000

    interpolated_model = np.zeros_like(models[0])
    for i in range(models.shape[1]):
        interp_func = interp1d(
            scales, models[:, i], kind='cubic', fill_value="extrapolate")
        interpolated_model[i] = interp_func(scale)

    halo_r = np.genfromtxt(
        files[0], delimiter='\t', usecols=(0), dtype=float)

    halo_r = halo_r/1000

    halo1_ds = interpolated_model
    # halo2_ds=halo_table2[:,1]

    halo_ds = A*halo1_ds

    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma = f(r)

    summed_halo = np.add(halo_dSigma, stellarSigma)

    if show:
        stellarSigma = M_stellar/(np.pi*halo_r**2)/1000000/1000000
        return halo_r, interpolated_model, stellarSigma

    return summed_halo


def log_likelihood(params, r, y, y_err):

    A, scale = params
    model_prediction = subhalo_profile(r, A, scale)

    chi = chisquare(y, model_prediction, y_err)
    return -0.5 * chi


def log_prior(params):
    A, scale = params
    if any(val < 0 for val in [A, scale]) or not (0.001 <= scale <= 0.01):
        return -np.inf
    # return -np.inf
    return 0.0


def log_probability(params, r, y, y_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, r, y, y_err)


def proposal_function(p0, random):

    new_p0 = p0 + random.normal(0, 0.005, size=p0.shape)
    # log_metropolis_ratio = log_ratio_of_proposal_probabilities(new_p0, p0)

    return new_p0, 0.0


bin = '0609'
index = '_rayleigh'

if bin == '0609':
    lowlim = 0.6
    highlim = 0.9

    M_stellar = math.pow(10, 10.94)

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

ndim = 2
nwalkers = 50


A_state = np.random.uniform(0, 1, size=nwalkers)

scale_state = np.random.uniform(0.001, 0.004, size=nwalkers)

initial_positions = np.vstack((A_state, scale_state)).T

param_names = ['A', 'scale']

if __name__ == "__main__":
    with Pool(processes=10) as pool:
        MH = [emcee.moves.MHMove(proposal_function)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(rp, ds, cov), moves=MH, pool=pool)

        nsteps = 1000
        sampler.run_mcmc(initial_positions, nsteps, progress=True)

    samples = sampler.get_chain(discard=0, flat=True)

    best_fit_params = np.median(samples, axis=0)
    print(best_fit_params)

    best_A, best_scale = best_fit_params

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels = [f"{i}" for i in param_names]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(sampler.chain[:, :, i].T, "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step")
    plt.show()

    R, subhalo, stellar = subhalo_profile(rp, best_A, best_scale, show=True)
    plt.plot(R, subhalo+stellar)
    plt.plot(R, subhalo)
    plt.plot(R, stellar)
    plt.errorbar(rp, ds, ds_err, fmt='o')
    plt.ylim(-20, 120)
    plt.show()
