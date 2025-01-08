import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from matplotlib.gridspec import GridSpec


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
fig = plt.figure(figsize=(9, 16))
gs = GridSpec(6, 1, figure=fig, height_ratios=[3, 1, 3, 1, 3, 1])

bins = ['0103', '0306', '0609']
labels = ['[0.1 - 0.3]', '[0.3 - 0.6]', '[0.6 - 0.9]']

index = '_rayleigh'

for i, bin in enumerate(bins):
    lowlim = bin[0] + '.' + bin[1]
    df = pd.read_csv(
        data_path + f'/roman_esd_70ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')
    rp = df['rp'].values
    ds = df['ds'].values
    ds_err = df['ds_err'].values

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

    # Main plot
    ax_main = fig.add_subplot(gs[i*2])
    ax_main.plot(R, sub, label='subhalo signal',
                 linestyle='--', c='blue', linewidth=2)
    ax_main.plot(R, halo, label='offset halo 1 signal',
                 linestyle='--', c='green', linewidth=2)
    # ax_main.plot(R, halo2, label='offset halo 2 signal', linestyle='-.', c='lime', linewidth=2)
    ax_main.plot(R, star, label='stellar signal',
                 linestyle='--', c='purple', linewidth=2)
    ax_main.plot(R, star + halo + sub,
                 label='combined profile', linewidth=2, c='red')
    # ax_main.plot(R, star + halo + sub + halo2, label='combined profile', linewidth=2, c='red')
    ax_main.errorbar(df['rp'], df['ds'], df['ds_err'], fmt='.', capsize=4, ecolor='k',
                     markerfacecolor='none', markeredgecolor='k', markeredgewidth=2, alpha=0.4)
    ax_main.errorbar(rp, ds, ds_err, fmt='.', label='observed lensing signal', capsize=4, ecolor='k',
                     markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
    ax_main.grid()
    ax_main.set_ylim(-30, 90)
    ax_main.set_xlim(0, 2.12)
    ax_main.tick_params(axis='x', labelsize=16)
    ax_main.tick_params(axis='y', labelsize=16)
    ax_main.set_ylabel(
        r'$\Delta \Sigma (R) \, [M_\odot / \mathrm{pc}^2]$', fontsize=18, labelpad=0)
    ax_main.set_title(labels[i], loc='center', x=0.5,
                      y=0.85, fontsize=18, color='black')
    ax_main.legend()

    print(chisquare(ds, ds - residuals, cov)/len(ds))

    # Residuals plot
    ax_residuals = fig.add_subplot(gs[i*2 + 1])
    ax_residuals.errorbar(rp, residuals, ds_err, fmt='.', ecolor='blue',
                          markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
    ax_residuals.axhline(0, color='k', linewidth=1, linestyle='--')
    ax_residuals.grid()
    ax_residuals.set_ylim(-20, 20)
    ax_residuals.set_ylabel('Residuals', fontsize=16)
    ax_residuals.set_xticklabels([])

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

    ax_main.grid(which='both', linewidth=1.5)
    ax_residuals.grid(which='both', linewidth=1.5)


plt.tight_layout()
# plt.savefig(f'final_fit{index}.png', bbox_inches='tight', dpi=300)
plt.show()
