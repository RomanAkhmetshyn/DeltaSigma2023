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

os.environ["OMP_NUM_THREADS"] = "1"
mpl.rcParams['figure.dpi'] = 100

astropy_cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Ob0= 0.049)
params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

# from subhalo_profile import make_profile
profile_path='C:/Users/romix/Documents/GitHub/DeltaSigma2023/MonteCarlo_offset_profile/new-test'

# def chisquare(expected, observed, expected_err):
#     # chi2=np.sum((expected - observed)**2 / expected_err**2 + np.log(expected_err**2))
#     chi2=np.sum((expected - observed)**2 / expected_err**2)
    
#     return chi2

def chisquare(y, yfit, err, v = False):
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
    
    
    if err.shape == (len(y),len(y)):
        #use full covariance
        if v:
            print('cov_mat chi2')
        inv_cov = np.linalg.inv( np.matrix(err) )
        chi2 = 0
        for i in range(len(y)):
            for j in range(len(y)):
                chi2 = chi2 + (y[i]-yfit[i])*inv_cov[i,j]*(y[j]-yfit[j])
        return chi2
        
    elif err.shape == (len(y),):
        if v:
            print('diagonal chi2')
        return sum(((y-yfit)**2.)/(err**2.))
    else:
        raise IOError('error in err or cov_mat input shape')

def build_profile(mass, A, B, z, distbin):
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta=2
    tau=100
    tnfw = TNFW(mass, c, z, tau, eta, cosmo=astropy_cosmo)
    
    
    R = np.linspace(0.01, 1.5, 75)
    sat_term=np.squeeze(tnfw.projected_excess(R))/1000000/1000000
    
    halo1_term=np.genfromtxt(profile_path+f'/{distbin}R0(Mh70).txt', delimiter='\t', usecols=( 1), dtype=float)/1000000
    halo2_term=np.genfromtxt(profile_path+f'/{distbin}R1(Mh70).txt', delimiter='\t', usecols=( 1), dtype=float)/1000000
    
    return R, sat_term, B*halo1_term*A, B*halo2_term*(1-A)
    

def combined_profile():
    pass

def subhalo_profile(r,mass,A, B, bin, redshift):

    tau=100
    mass=math.pow(10, mass)
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=redshift, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta=2
    tnfw = TNFW(mass, c, redshift, tau, eta)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    dSigma=np.squeeze(tnfw.projected_excess(r))/1000000


    
    # dSigma=nfw.projected_excess(R)
    
    halo_table1 = np.genfromtxt(profile_path+f'/{bin}R0(Mh70).txt', delimiter='\t', usecols=(0, 1), dtype=float)
    halo_table2 = np.genfromtxt(profile_path+f'/{bin}R1(Mh70).txt', delimiter='\t', usecols=(0, 1), dtype=float)
    
    halo_r=halo_table1[:,0]/1000
    
    halo1_ds=halo_table1[:,1]
    halo2_ds=halo_table2[:,1]
    
    halo_ds=B*(A*halo1_ds+(1-A)*halo2_ds)
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)

    summed_halo=np.add(dSigma,halo_dSigma)/1000000

    return summed_halo


def log_likelihood(params, r, y1, y2, y3, y_err1, y_err2, y_err3):
    
    mass1, mass2, mass3, A1, A2, A3, B = params
    model_prediction1 = subhalo_profile(r, mass1, A1, B, '0103', z1)
    model_prediction2 = subhalo_profile(r, mass2, A2, B, '0306', z2)
    model_prediction3 = subhalo_profile(r, mass3, A3, B, '0609', z3)
    
    # chi1= -0.5 * np.sum((y1 - model_prediction1)**2 / y_err1**2 + np.log(y_err1**2))
    # chi2= -0.5 * np.sum((y2 - model_prediction2)**2 / y_err2**2 + np.log(y_err2**2))
    # chi3= -0.5 * np.sum((y3 - model_prediction3)**2 / y_err3**2 + np.log(y_err3**2))
    
    chi1=chisquare(y1, model_prediction1, y_err1)
    chi2=chisquare(y2, model_prediction2, y_err2)
    chi3=chisquare(y3, model_prediction3, y_err3)
    
    return -0.5*(chi1+chi2+chi3)

def log_prior(params):
    mass1, mass2, mass3, A1, A2, A3, B = params
    if any(val < 0 for val in params) or not (11 <= mass1 <= 13.5 and 11 <= mass2 <= 13.5 and 11 <= mass3 <= 13.5):
        return -np.inf 
    return 0.0


def log_probability(params, r, y1, y2, y3, y_err1, y_err2, y_err3):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, r, y1, y2, y3, y_err1, y_err2, y_err3)

# bin='0609'

# if bin=='0609':
#     lowlim=0.6
#     highlim=0.9
# elif bin=='0306':
#     lowlim=0.3
#     highlim=0.6
# elif bin=='0103':
#     lowlim=0.1
#     highlim=0.3 
    
# lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits")
# data_mask = (
#         (lenses["R"] >= lowlim)
#         & (lenses["R"] < highlim)
#         & (lenses["PMem"] > 0.8)
#         # & (lenses["zspec"] > -1)
       
#     )
# lenses = lenses[data_mask]

# sumz=0
# for i in range(len(lenses)):
#     if lenses[i]['zspec']>-1:
#         sumz+=lenses[i]['zspec']
#     else:
#         sumz+=lenses[i]['Z_halo']
   
# avg_z=sumz/len(lenses)
# del sumz, i
z1=0.3479985437898726
z2=0.3122043843287107
z3=0.22904300349155315
        
df1 = pd.read_csv(profile_path+'/roman_esd_ShapePipe_redmapper_clusterDist0.1_randomsTrue_1.csv')
df2 = pd.read_csv(profile_path+'/roman_esd_ShapePipe_redmapper_clusterDist0.3_randomsTrue_1.csv')
df3 = pd.read_csv(profile_path+'/roman_esd_ShapePipe_redmapper_clusterDist0.6_randomsTrue_1.csv')
cov1=np.loadtxt('C:/scp/roman_esd_ShapePipe_redmapper_clusterDist0.1_randomsTrue_1_covmat.txt', delimiter=' ', dtype='float64')
cov2=np.loadtxt('C:/scp/roman_esd_ShapePipe_redmapper_clusterDist0.3_randomsTrue_1_covmat.txt', delimiter=' ', dtype='float64')
cov3=np.loadtxt('C:/scp/roman_esd_ShapePipe_redmapper_clusterDist0.6_randomsTrue_1_covmat.txt', delimiter=' ', dtype='float64')



# Save the "ds" and "rp" columns as variables
# ds1 = (df1['ds'][1:]).values
# ds2 = (df2['ds'][1:]).values
# ds3 = (df3['ds'][1:]).values
# rp = df1['rp'][1:]
# ds_err1=df1['ds_err'][1:]
# ds_err2=df2['ds_err'][1:]
# ds_err3=df3['ds_err'][1:]
# cov1=cov1[1:, 1:]
# cov2=cov2[1:, 1:]
# cov3=cov3[1:, 1:]

ds1 = df1['ds']
ds2 = df2['ds']
ds3 = df3['ds']
rp = df1['rp']
ds_err1=df1['ds_err']
ds_err2=df2['ds_err']
ds_err3=df3['ds_err']

ndim = 7
nwalkers = 50

mass1_state=np.random.uniform(11, 13, size=nwalkers)
mass2_state=np.random.uniform(11, 13, size=nwalkers)
mass3_state=np.random.uniform(11, 13, size=nwalkers)
# tau_state=np.random.uniform(-1, 3, size=nwalkers)
A1_state=np.random.uniform(0, 1, size=nwalkers)
A2_state=np.random.uniform(0, 1, size=nwalkers)
A3_state=np.random.uniform(0, 1, size=nwalkers)
B_state=np.random.uniform(0.5, 0.8, size=nwalkers)

initial_positions = np.vstack((mass1_state, 
                               mass2_state,
                               mass3_state, 
                               A1_state, 
                               A2_state, 
                               A3_state, 
                               B_state)).T


def proposal_function(p0, random):
    
    new_p0 = p0 + random.normal(0, 0.1, size=p0.shape)
    # log_metropolis_ratio = log_ratio_of_proposal_probabilities(new_p0, p0)
    
    return new_p0, 0.0

if __name__ == "__main__":
    with Pool(processes=12) as pool:
        MH = [emcee.moves.MHMove(proposal_function)]
        nsteps = 10000
    
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            # args=(rp, ds1, ds2, ds3, ds_err1, ds_err2, ds_err3),
            args=(rp, ds1, ds2, ds3, cov1, cov2, cov3),
            moves=MH,
            pool=pool  # Use the pool for parallelization
        )
    
        sampler.run_mcmc(initial_positions, nsteps, progress=True)
    
    
    samples = sampler.get_chain(discard=5000, flat=True)
    
    non_flat_samples=sampler.get_chain()
    
    min_logprob_idx=np.unravel_index(np.argmax(sampler.lnprobability), sampler.lnprobability.shape)
    best_fit_params=non_flat_samples[min_logprob_idx[1], min_logprob_idx[0], :]
    
    med_params = np.median(samples, axis=0)
    
    mass1_perc = np.percentile(samples[:, 0], [16, 50, 84])
    mass2_perc = np.percentile(samples[:, 1], [16, 50, 84])
    mass3_perc = np.percentile(samples[:, 2], [16, 50, 84])
    A1_perc = np.percentile(samples[:, 3], [16, 50, 84])
    A2_perc = np.percentile(samples[:, 4], [16, 50, 84])
    A3_perc = np.percentile(samples[:, 5], [16, 50, 84])
    B_perc = np.percentile(samples[:, 6], [16, 50, 84])
    
    q_mass1 = np.diff(mass1_perc)
    q_mass2 = np.diff(mass2_perc)
    q_mass3 = np.diff(mass3_perc)
    q_A1 = np.diff(A1_perc)
    q_A2 = np.diff(A2_perc)
    q_A3 = np.diff(A3_perc)
    q_B = np.diff(B_perc)

    
    # best_fit_params = np.median(samples, axis=0)
    # param_uncertainties = np.std(samples, axis=0)
    
    lens_mass1, lens_mass2, lens_mass3, param_A1, param_A2, param_A3, param_B =best_fit_params
    
    import corner
    
    
    # Get the samples (chain)
    # samples = sampler.get_chain(discard=1000, flat=True)  # Discard the first 100 steps as burn-in
    
    # Extract the parameter names (you can customize them as needed)
    param_names = ['log(Mass) 1', 'log(Mass) 2', 'log(Mass) 3', 'A 1', 'A 2', 'A 3', 'B']
    
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
    
    #%%
    plt.subplot(2, 1, 1)
    R, sat_term, halo1_term, halo2_term=build_profile(math.pow(10,lens_mass1), param_A1, param_B, z1, '0103')
    
    
    chi=chisquare(ds1, subhalo_profile(rp, lens_mass1, param_A1, param_B, '0103', z1), cov1)
    
    plt.plot(R,sat_term, label='satellite')
    plt.plot(R,halo1_term, label='halo R0', linestyle='--')
    plt.plot(R,halo2_term, label='halo R1' , linestyle='--')
    plt.plot(R, sat_term+halo1_term+halo2_term, label='combined fit')
    plt.errorbar(rp, ds1, ds_err1, fmt='o',label='dsigma Data', markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
    plt.xlabel('R (Mpc)')
    plt.ylabel('M/pc^2')
    plt.grid()
    # plt.ylim(-40,120)
    med_chi=chisquare(ds1, subhalo_profile(rp, mass1_perc[1], A1_perc[1], B_perc[1], '0103', z1), cov1)
    title=f'0103 z: {z1:.3f}  \n best logprob log(M): {lens_mass1:.3f}, A: {param_A1:.3f}, B: {param_B:.3f}, chi$^2$: {chi:.3f}'
    title = (
        title + '\n'
        r'med log(M): {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, A: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, B: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$,chi$^2$: {:.3f}'.format(
        mass1_perc[1], q_mass1[0], q_mass1[1], A1_perc[1], q_A1[0], q_A1[1], B_perc[1], q_B[0], q_B[1], med_chi)
    )
    plt.title(title)
    # plt.legend()
    
    plt.subplot(2, 1, 2)
    residual=ds1-subhalo_profile(rp, lens_mass1, param_A1, param_B, '0103', z1)
    # print(np.sum(residual))
    plt.scatter(rp, residual)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Zero line for reference
    for x, err in zip(rp, ds_err1):
        plt.plot([x, x], [0 - err , 0 + err], color='blue', alpha=0.5)
    plt.ylabel('data-model')
    # plt.title(f'sum of residuals = {}')
    plt.tight_layout()
    plt.show()
    
    #%%
    plt.subplot(2, 1, 1)
    R, sat_term, halo1_term, halo2_term=build_profile(math.pow(10,lens_mass2), param_A2, param_B, z2, '0306')
    
    
    chi=chisquare(ds2, subhalo_profile(rp, lens_mass2, param_A2, param_B, '0306', z2), cov2)
    
    plt.plot(R,sat_term, label='satellite')
    plt.plot(R,halo1_term, label='halo R0', linestyle='--')
    plt.plot(R,halo2_term, label='halo R1' , linestyle='--')
    plt.plot(R, sat_term+halo1_term+halo2_term, label='combined fit')
    plt.errorbar(rp, ds2, ds_err2, fmt='o',label='dsigma Data', markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
    plt.xlabel('R (Mpc)')
    plt.ylabel('M/pc^2')
    plt.grid()
    # plt.ylim(-40,120)
    med_chi=chisquare(ds2, subhalo_profile(rp, mass2_perc[1], A2_perc[1], B_perc[1], '0306', z2), cov2)
    title=f'0306 z: {z2:.3f}  \n best logprob log(M): {lens_mass2:.3f}, A: {param_A2:.3f}, B: {param_B:.3f}, chi$^2$: {chi:.3f}'
    title = (
        title + '\n'
        r'med log(M): {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, A: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, B: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$,chi$^2$: {:.3f}'.format(
        mass2_perc[1], q_mass2[0], q_mass2[1], A2_perc[1], q_A2[0], q_A2[1], B_perc[1], q_B[0], q_B[1], med_chi)
    )
    plt.title(title)
    # plt.legend()
    
    plt.subplot(2, 1, 2)
    residual=ds2-subhalo_profile(rp, lens_mass2, param_A2, param_B, '0306', z2)
    # print(np.sum(residual))
    plt.scatter(rp, residual)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Zero line for reference
    for x, err in zip(rp, ds_err1):
        plt.plot([x, x], [0 - err , 0 + err], color='blue', alpha=0.5)
    plt.ylabel('data-model')
    # plt.title(f'sum of residuals = {}')
    plt.tight_layout()
    plt.show()
    
    #%%
    plt.subplot(2, 1, 1)
    R, sat_term, halo1_term, halo2_term=build_profile(math.pow(10,lens_mass3), param_A3, param_B, z3, '0609')
    
    
    chi=chisquare(ds3, subhalo_profile(rp, lens_mass3, param_A3, param_B, '0609', z3), cov3)
    
    plt.plot(R,sat_term, label='satellite')
    plt.plot(R,halo1_term, label='halo R0', linestyle='--')
    plt.plot(R,halo2_term, label='halo R1' , linestyle='--')
    plt.plot(R, sat_term+halo1_term+halo2_term, label='combined fit')
    plt.errorbar(rp, ds3, ds_err3, fmt='o',label='dsigma Data', markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
    plt.xlabel('R (Mpc)')
    plt.ylabel('M/pc^2')
    plt.grid()
    # plt.ylim(-40,120)
    med_chi=chisquare(ds3, subhalo_profile(rp, mass3_perc[1], A3_perc[1], B_perc[1], '0609', z3), cov3)
    title=f'0609 z: {z3:.3f}  \n best logprob log(M): {lens_mass3:.3f}, A: {param_A3:.3f}, B: {param_B:.3f}, chi$^2$: {chi:.3f}'
    title = (
        title + '\n'
        r'med log(M): {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, A: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, B: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$,chi$^2$: {:.3f}'.format(
        mass3_perc[1], q_mass3[0], q_mass3[1], A3_perc[1], q_A3[0], q_A3[1], B_perc[1], q_B[0], q_B[1], med_chi)
    )
    plt.title(title)
    # plt.legend()
    
    plt.subplot(2, 1, 2)
    residual=ds3-subhalo_profile(rp, lens_mass3, param_A3, param_B, '0609', z3)
    # print(np.sum(residual))
    plt.scatter(rp, residual)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Zero line for reference
    for x, err in zip(rp, ds_err1):
        plt.plot([x, x], [0 - err , 0 + err], color='blue', alpha=0.5)
    plt.ylabel('data-model')
    # plt.title(f'sum of residuals = {}')
    plt.tight_layout()
    plt.show()