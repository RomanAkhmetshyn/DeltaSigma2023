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


# Sample data path
data_path = 'C:/scp'

# Define custom figure and GridSpec layout
fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('#FBFEF9')  # Background color of the entire figure
gs = GridSpec(9, 1, figure=fig, height_ratios=[
              0.5, 3, 1, 0.5, 3, 1, 0.5, 3, 1], hspace=0.0)

bins = ['0103', '0306', '0609']
labels = ['[0.1 - 0.3]', '[0.3 - 0.6]', '[0.6 - 0.9]']
stellar = [np.power(10, 10.94),
           np.power(10, 10.92),
           np.power(10, 10.86)]
avg_z = [0.3479985,
         0.3122,
         0.229]
index = '_rayleigh'

for i, bin in enumerate(bins):
    lowlim = bin[0] + '.' + bin[1]
    df = pd.read_csv(
        data_path + f'/roman_esd_70ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
    rp = df['rp'].values
    ds = df['ds'].values
    ds_err = df['ds_err'].values

    scales = []  # To store scale values
    models = []  # To store the second column of data

    file_pattern = f"{bin}_*_rayleigh.txt"
    files = natsorted(glob.glob(file_pattern))

    for file in files:
        scale = float(file.split('_')[1])
        scales.append(scale)

        data = np.loadtxt(file, usecols=1)
        models.append(data)

    scales = np.array(scales)
    models = np.array(models)/1000000

    # cov = np.loadtxt(f'C:/scp/{lowlim}trimmed.csv', delimiter=',', dtype='float64')
    # dont worry it does nothin
    cov = np.loadtxt(
        f'C:/scp/roman_esd_70ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1_covmat.txt')
    factor = (100 - len(cov[0]) - 2) / (100 - 1)
    factor = 1 / factor
    cov = cov * factor
    cov = cov[1:, 1:]

    ds_err = ds_err * math.sqrt(factor)

    if bin == '0609':
        # ds=np.concatenate((ds[1:5], ds[-4:]))
        # rp=np.concatenate((rp[1:5], rp[-4:]))
        # ds_err=np.concatenate((ds_err[1:5], ds_err[-4:]))*math.sqrt(factor)

        ds = ds[1:]
        rp = rp[1:]
        ds_err = ds_err[1:]
    elif bin == '0306':
        # ds=np.concatenate((ds[1:4], ds[8:]))
        # rp=np.concatenate((rp[1:4], rp[8:]))
        # ds_err=np.concatenate((ds_err[1:4], ds_err[8:]))*math.sqrt(factor)

        ds = ds[1:]
        rp = rp[1:]
        ds_err = ds_err[1:]
    elif bin == '0103':
        # ds=np.concatenate((ds[1:2], ds[5:]))
        # rp=np.concatenate((rp[1:2], rp[5:]))
        # ds_err=np.concatenate((ds_err[1:2], ds_err[5:]))*math.sqrt(factor)

        ds = ds[1:]
        rp = rp[1:]
        ds_err = ds_err[1:]

    halo = np.genfromtxt(f'./results/{bin}_halo{index}.txt')
    # halo2 = np.genfromtxt(f'{bin}_halo2{index}.txt')
    sub = np.genfromtxt(f'./results/{bin}_sub{index}.txt')
    star = np.genfromtxt(f'./results/{bin}_star{index}.txt')
    R = np.genfromtxt(f'./results/R{index}.txt')
    residuals = np.genfromtxt(f'./results/{bin}_res{index}.txt')

    samples = np.load(f'./results/{bin}_samples{index}.txt.npy')
    random_indices = np.random.choice(samples.shape[0], 1000, replace=False)
    lower_bounds = np.percentile(samples, 16, axis=0)
    upper_bounds = np.percentile(samples, 84, axis=0)
    best_fit_params = np.median(samples, axis=0)

    # Main plot
    ax_main = fig.add_subplot(gs[i*3+1])
    ax_main.plot(R, sub, label='subhalo signal',
                 linestyle='--', c='#1E3E48', linewidth=2)
    ax_main.plot(R, halo, label='offset halo 1 signal',
                 linestyle='--', c='#69995D', linewidth=2)
    # ax_main.plot(R, halo2, label='offset halo 2 signal', linestyle='-.', c='lime', linewidth=2)
    ax_main.plot(R, star, label='stellar signal',
                 linestyle='--', c='#7E1946', linewidth=2)
    ax_main.plot(R, star + halo + sub,
                 label='combined profile', linewidth=2, c='red', zorder=6)
    # ax_main.plot(R, star + halo + sub + halo2, label='combined profile', linewidth=2, c='red')
    ax_main.errorbar(df['rp'], df['ds'], df['ds_err'], fmt='.', capsize=4, ecolor='k',
                     markerfacecolor='none', markeredgecolor='k', markeredgewidth=2, alpha=0.4, zorder=7)
    ax_main.errorbar(rp, ds, ds_err, fmt='.', label='observed lensing signal', capsize=4, ecolor='k',
                     markerfacecolor='none', markeredgecolor='k', markeredgewidth=2, zorder=8)

    lower_curve = sum(subhalo_profile(
        df['rp'], *lower_bounds, stellar[i], show=True, avg_z=avg_z[i])[1:])
    upper_curve = sum(subhalo_profile(
        df['rp'], *upper_bounds, stellar[i], show=True, avg_z=avg_z[i])[1:])

    ax_main.fill_between(R, lower_curve, upper_curve, color='red', alpha=0.2)
    ax_main.set_facecolor('#FBFEF9')         # Background color of the plot area
    # mass, A, scale = [12.527, 1.525, 393.934]

    # Generate the curve using the subhalo_profile function
    # R, s_halo, s_stellar, s_sathalo = subhalo_profile(
    #     df['rp'], mass, A, scale, stellar[i], show=True, avg_z=avg_z[i])

    # Plot the random curve with faint red color
    # ax_main.plot(R, s_halo+s_stellar+s_sathalo,
    #              color='red', alpha=0.5, linewidth=0.8)

    # print(chisquare(ds, subhalo_profile(
    #     df['rp'], mass, A, scale, stellar[i], show=False, avg_z=avg_z[i]), cov)/len(ds))

    # for idx in random_indices:
    #     mass, A, scale = samples[idx]

    #     # Generate the curve using the subhalo_profile function
    #     R, s_halo, s_stellar, s_sathalo = subhalo_profile(
    #         df['rp'], mass, A, scale, stellar[i], show=True, avg_z=avg_z[i])

    #     # Plot the random curve with faint red color
    #     ax_main.plot(R, s_halo+s_stellar+s_sathalo,
    #                  color='red', alpha=0.05, linewidth=0.8)

    # ax_main.grid()
    ax_main.set_ylim(-8, 70)
    ax_main.set_xlim(0, 2.12)
    ax_main.tick_params(axis='x', labelsize=16)
    ax_main.tick_params(axis='y', labelsize=16)
    ax_main.set_ylabel(
        r'$\Delta \Sigma (R) \, [M_\odot / \mathrm{pc}^2]$', fontsize=16, labelpad=0)
    ax_main.set_title(labels[i], loc='center', x=0.92,
                      y=0.82, fontsize=18, color='black')
    # ax_main.legend()

    print(chisquare(ds, ds - residuals, cov)/(len(ds)-3))
    # Residuals plot
    ax_residuals = fig.add_subplot(gs[i*3 + 2])
    ax_residuals.errorbar(rp, residuals/ds_err, ds_err/ds_err, fmt='.', ecolor='blue',
                          markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
    ax_residuals.axhline(0, color='k', linewidth=1, linestyle='--')
    # ax_residuals.grid()
    ax_residuals.set_ylim(-3, 3)
    ax_residuals.set_ylabel('Z-score', fontsize=12)
    ax_residuals.set_xticklabels([])
    ax_residuals.set_facecolor('#FBFEF9')
    if i == len(bins) - 1:
        ax_residuals.set_xlabel(r'$R (\mathrm{Mpc})$', fontsize=18)

    ax_main.spines['top'].set_linewidth(1.5)
    ax_main.spines['bottom'].set_linewidth(1.5)
    ax_main.spines['left'].set_linewidth(1.5)
    ax_main.spines['right'].set_linewidth(1.5)

    ax_residuals.spines['top'].set_linewidth(1.5)
    ax_residuals.spines['bottom'].set_linewidth(1.5)
    ax_residuals.spines['left'].set_linewidth(1.5)
    ax_residuals.spines['right'].set_linewidth(1.5)

    # ax_main.grid(which='both', linewidth=1.5)
    # ax_residuals.grid(which='both', linewidth=1.5)
    ax_main.set_xticklabels(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], fontsize=16)
    ax_main.get_xaxis().set_visible(False)
ax_residuals.set_xticklabels(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], fontsize=16)
# plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
# plt.savefig(f'final_fit{index}.pdf', bbox_inches='tight', dpi=900)
plt.show()
