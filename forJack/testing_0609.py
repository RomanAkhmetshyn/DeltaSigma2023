# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:08:33 2025

@author: romix
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from matplotlib.gridspec import GridSpec
from profiley.nfw import TNFW
from colossus.halo import concentration
from colossus.cosmology import cosmology
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from natsort import natsorted, ns
import glob

np.random.seed(42)
mpl.rcParams['figure.dpi'] = 300

astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.049)
params = {"flat": True, "H0": 70, "Om0": 0.3,
          "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")


def chisquare(y, yfit, err, v=False):
    """
    Compute chi2 between data and fitted curve

    Parameters
    ----------
    y: array
        y values of data
    err: array (1D or 2D)
        either error bars (if independent data points)
        or covariance matrix
    yfit: array
        fitted values of data
    v: bool
        verbose output

    Returns
    -------
    chi2: float
    """

    if err.shape == (len(y), len(y)):
        # use full covariance
        if v:
            print('cov_mat chi2')
        inv_cov = np.linalg.inv(np.matrix(err))
        chi2 = 0
        for i in range(len(y)):
            for j in range(len(y)):
                chi2 = chi2 + (y[i]-yfit[i])*inv_cov[i, j]*(y[j]-yfit[j])
        return chi2

    elif err.shape == (len(y),):
        if v:
            print('diagonal chi2')
        return sum(((y-yfit)**2.)/(err**2.))
    else:
        raise IOError('error in err or cov_mat input shape')


def subhalo_profile(r, mass, A, scale, M_stellar, avg_z, show=False):

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


scales = []  # To store scale values
models = []  # To store the second column of data

file_pattern = f"0609_*_rayleigh.txt"
files = natsorted(glob.glob(file_pattern))

for file in files:
    scale = float(file.split('_')[1])
    scales.append(scale)

    data = np.loadtxt(file, usecols=1)
    models.append(data)

data_path = 'C:/scp'
df = pd.read_csv(
    data_path + f'/roman_esd_70ShapePipe_redmapper_clusterDist0.6_randomsTrue_1.csv')
rp = df['rp'].values
ds = df['ds'].values
ds_err = df['ds_err'].values


# cov = np.loadtxt(f'C:/scp/{lowlim}trimmed.csv', delimiter=',', dtype='float64')
# dont worry it does nothin
cov = np.loadtxt(
    f'C:/scp/roman_esd_70ShapePipe_redmapper_clusterDist0.6_randomsTrue_1_covmat.txt')
factor = (100 - len(cov[0]) - 2) / (100 - 1)
factor = 1 / factor
cov = cov * factor
cov = cov[1:, 1:]

ds_err = ds_err * math.sqrt(factor)

scales = np.array(scales)
models = np.array(models)/1000000
r = np.genfromtxt(f'./results/R_rayleigh.txt')

R, halo, stellar, sat = subhalo_profile(
    r, 12.536, 1.424, 334.219, np.power(10, 10.86), 0.229, True)
model = subhalo_profile(rp, 12.536, 1.424, 334.219,
                        np.power(10, 10.86), 0.229, False)
chisq = chisquare(ds, model, ds_err)

plt.plot(R, halo+sat+stellar, label=f'left peak: {chisq:.2f}')

R, halo, stellar, sat = subhalo_profile(
    r, 12.486, 1.544, 395.913, np.power(10, 10.86), 0.229, True)
model = subhalo_profile(rp, 12.486, 1.544, 395.913,
                        np.power(10, 10.86), 0.229, False)
chisq = chisquare(ds, model, ds_err)

plt.plot(R, halo+sat+stellar, label=f'right peak: {chisq:.2f}')

R, halo, stellar, sat = subhalo_profile(
    r, 12.478, 1.513, 369, np.power(10, 10.86), 0.229, True)
model = subhalo_profile(rp, 12.478, 1.513, 369,
                        np.power(10, 10.86), 0.229, False)
chisq = chisquare(ds, model, ds_err)

plt.plot(R, halo+sat+stellar, label=f'dip: {chisq:.2f}')

plt.errorbar(rp, ds, ds_err, fmt='.', label='observed lensing signal', capsize=4, ecolor='k',
             markerfacecolor='none', markeredgecolor='k', markeredgewidth=2, zorder=8)

plt.ylim(-5, 60)
plt.xlim(0, 3.5)
plt.legend()
plt.show()
