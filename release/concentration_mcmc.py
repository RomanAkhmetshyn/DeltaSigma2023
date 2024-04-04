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

astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0= 0.049)
params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

# from subhalo_profile import make_profile
profile_path='C:/Users/romix/Documents/GitHub/DeltaSigma2023/MonteCarlo_offset_profile/new-test'

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

def build_profile(mass, A, z, distbin):
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )
    # c=4.67*(mass/math.pow(10, 14))**(-0.11)

    eta=2
    tnfw = TNFW(mass, c, avg_z, 100, eta, cosmo=astropy_cosmo)
    
    
    
    
    R = np.linspace(0.01, 1.5, 75)
    sat_term=np.squeeze(tnfw.projected_excess(R))/1000000/1000000
    
    stellarSigma = M_stellar/(np.pi*R**2)/1000000/1000000
    
    halo1_term=np.genfromtxt(profile_path+f'/{distbin}R0(Mh70).txt', delimiter='\t', usecols=( 1), dtype=float)/1000000
    # halo2_term=np.genfromtxt(profile_path+f'/{distbin}R1(Mh70).txt', delimiter='\t', usecols=( 1), dtype=float)/1000000
    
    return R, halo1_term*A
    

def subhalo_profile(r,c,A):


    halo_r =
    
    halo1_ds =
    # halo2_ds=halo_table2[:,1]
    
    halo_ds=A*halo1_ds
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)

    summed_halo = halo_dSigma/1000000
    

    return summed_halo


def log_likelihood(params, r, y, y_err):
    
    mass, A= params
    model_prediction = subhalo_profile(r, mass, A)
    
    # y=np.concatenate((y[1:2], y[-9:]))
    # model_prediction=np.concatenate((model_prediction[1:2], model_prediction[-9:]))
    # y_err=np.concatenate((y_err[1:2,1:2], y_err[-9:, -9:]))
    
    chi=chisquare(y, model_prediction, y_err)
    return -0.5 * chi

def log_prior(params):
    mass, A = params
    if any(val < 0 for val in [mass, A]) or not (11 <= mass <= 13) or A>1:
        return -np.inf  
    # return -np.inf 
    return 0.0

def log_probability(params, r, y, y_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, r, y, y_err)

bin='0609'

if bin=='0609':
    lowlim=0.6
    highlim=0.9
    
    M_stellar = math.pow(10, 10.94)
    
elif bin=='0306':
    lowlim=0.3
    highlim=0.6
    
    M_stellar = math.pow(10, 10.92)
    
elif bin=='0103':
    lowlim=0.1
    highlim=0.3 
    
    M_stellar = math.pow(10, 10.86)
    
lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits")
data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
       
    )
lenses = lenses[data_mask]

sumz=0
for i in range(len(lenses)):
    if lenses[i]['zspec']>-1:
        sumz+=lenses[i]['zspec']
    else:
        sumz+=lenses[i]['Z_halo']
   
avg_z=sumz/len(lenses)
del sumz, i
        
df = pd.read_csv(profile_path+f'/roman_esd_ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1.csv')


# cov=np.loadtxt(f'C:/scp/roman_esd_ShapePipe_redmapper_clusterDist{lowlim}_randomsTrue_1_covmat.txt', delimiter=' ', dtype='float64')
cov=np.loadtxt(f'C:/scp/{lowlim}trimmed.txt', delimiter=' ', dtype='float64')
factor=(100-len(cov[0])-2)/(100-1)
factor=1/factor
cov=cov*factor

# y_err=np.zeros((10,10))
# y_err[0]=np.concatenate((cov[1,1:2], cov[1,-9:]))
# onecol=cov[-9:,1]
# rest=cov[-9:,-9:]
# rest=np.hstack((onecol.reshape(-1, 1), rest))
# y_err=np.vstack((y_err[0], rest))
# cov=y_err

# Save the "ds" and "rp" columns as variabl
# cov=cov[1:, 1:]


# covmat = np.loadtxt('<file>')

# #15x15 matrix

# mask = np.array([False, True, False, False, True, True, ....])

# dvec_new = dvec[mask]
# covmat_new = np.array([line[mask] for line in covmat[mask]])

# ds = (df['ds'][1:]).values
# rp = df['rp'][1:]
# ds_err=df['ds_err'][1:]
if bin=='0609':
    ds = (df['ds']).values
    ds=np.concatenate((ds[1:5], ds[-4:]))
    rp = df['rp']
    rp=np.concatenate((rp[1:5], rp[-4:]))
    ds_err=df['ds_err']
    ds_err=np.concatenate((ds_err[1:5], ds_err[-4:]))*math.sqrt(factor)
elif bin=='0306':
    ds = (df['ds']).values
    ds=np.concatenate((ds[1:3], ds[-6:]))
    rp = df['rp']
    rp=np.concatenate((rp[1:3], rp[-6:]))
    ds_err=df['ds_err']
    ds_err=np.concatenate((ds_err[1:3], ds_err[-6:]))*math.sqrt(factor)
elif bin=='0103':
    ds = (df['ds']).values
    ds=np.concatenate((ds[1:2], ds[-9:]))
    rp = df['rp']
    rp=np.concatenate((rp[1:2], rp[-9:]))
    ds_err=df['ds_err']
    ds_err=np.concatenate((ds_err[1:2], ds_err[-9:]))*math.sqrt(factor)

# ds = (df['ds']).values
# rp = df['rp']
# ds_err=df['ds_err']*math.sqrt(factor)

ndim = 2
nwalkers = 50

mass_state=np.random.uniform(11, 12.5, size=nwalkers)
# tau_state=np.random.uniform(-1, 2, size=nwalkers)
A_state=np.random.uniform(0, 1, size=nwalkers)
# B_state=np.random.uniform(0, 1, size=nwalkers)

initial_positions = np.vstack((mass_state, A_state)).T


def proposal_function(p0, random):
    
    new_p0 = p0 + random.normal(0, 0.1, size=p0.shape)
    # log_metropolis_ratio = log_ratio_of_proposal_probabilities(new_p0, p0)
    
    return new_p0, 0.0

if __name__ == "__main__":
    with Pool(processes=12) as pool:
        MH=[emcee.moves.MHMove(proposal_function)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(rp, ds, cov), moves=MH, pool=pool)
        
        nsteps =1000
        sampler.run_mcmc(initial_positions, nsteps, progress=True)


    samples = sampler.get_chain(discard=500, flat=True)
    
    non_flat_samples=sampler.get_chain()
    min_logprob_idx=np.unravel_index(np.argmax(sampler.lnprobability), sampler.lnprobability.shape)
    best_fit_params=non_flat_samples[min_logprob_idx[1], min_logprob_idx[0], :]
    med_params = np.median(samples, axis=0)
    
    mass_tiles=np.percentile(samples[:, 0], [16, 50, 84])

    A_tiles=np.percentile(samples[:, 1], [16, 50, 84])
    # B_tiles=np.percentile(samples[:, 3], [16, 50, 84])
    q1 = np.diff(mass_tiles)

    q3 = np.diff(A_tiles)
    # q4 = np.diff(B_tiles)
    
    # best_fit_params = np.median(samples, axis=0)
    # param_uncertainties = np.std(samples, axis=0)
    
    lens_mass, param_A=best_fit_params
    
    import corner
    
    
    # Get the samples (chain)
    # samples = sampler.get_chain(discard=1000, flat=True)  # Discard the first 100 steps as burn-in
    
    # Extract the parameter names (you can customize them as needed)
    param_names = ['log(Mass)', 'A']
    
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
    
    
    plt.subplot(2, 1, 1)
    R, sat_term, halo1_term, star_term = build_profile(math.pow(10,lens_mass), param_A, avg_z, bin)
    plt.plot(R, sat_term, label='satellite')
    plt.plot(R, halo1_term, label='halo R0', linestyle='--')
    plt.plot(R, star_term, label='stellar', c='purple')
    # plt.plot(R,halo2_term, label='halo R1' , linestyle='--')
    plt.plot(R, sat_term+halo1_term+star_term, label='combined fit')
    plt.errorbar(df['rp'], df['ds'], df['ds_err'], fmt='o',label='dsigma Data', markerfacecolor='none', markeredgecolor='k', markeredgewidth=2, alpha=0.3)
    plt.errorbar(rp, ds, ds_err, fmt='o',label='dsigma Data', markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)

    plt.xlabel('R (Mpc)')
    plt.ylabel('M/pc^2')
    plt.grid()
    plt.ylim(-20,120)
    chi=chisquare(ds, subhalo_profile(rp, lens_mass,param_A), cov)
    med_chi=chisquare(ds, subhalo_profile(rp, mass_tiles[1], A_tiles[1]), cov)
    title=f'{bin} z: {avg_z:.3f}  \n best logprob log(M): {lens_mass:.3f}, A: {param_A:.3f}, chi$^2$: {chi:.3f}'
    title = (
        title + '\n'
        r'med log(M): {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$, A: {:.3f}$_{{ -{:.2f} }}^{{ +{:.2f} }}$,chi$^2$: {:.3f}'.format(
        mass_tiles[1], q1[0], q1[1], A_tiles[1], q3[0], q3[1], med_chi)
    )
    # plt.yscale('log')
    # plt.xscale('log')
    plt.title(title)
    # plt.legend()
    
    plt.subplot(2, 1, 2)
    residual=ds-subhalo_profile(rp, lens_mass,param_A)
    # print(np.sum(residual))
    plt.scatter(rp, residual)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Zero line for reference
    for x, err, res in zip(rp, ds_err, residual):
        plt.plot([x, x], [res - err , res + err], color='blue', alpha=0.5)
    plt.ylabel('reiduals')
    # plt.title(f'sum of residuals = {}')
    plt.tight_layout()
    plt.show()
   