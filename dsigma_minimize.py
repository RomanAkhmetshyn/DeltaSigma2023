
from scipy.optimize import minimize
import pandas as pd
from profiley.nfw import TNFW, NFW
import matplotlib.pyplot as plt
from colossus.halo import concentration, profile_nfw
from colossus.cosmology import cosmology
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit
from subhalo_profile import make_profile
import math

def subhalo_profile(r,mass,tau,A):
    # print(mass)
    # print(tau)
    # print(z)
    # print(A)
    
    mass=math.pow(10, mass)
    concentration_model="duffy08"
    c=concentration.concentration(
            M=mass, mdef="200m", z=z, model=concentration_model
            )

    eta=2
    tnfw = TNFW(mass, c, z, tau, eta)
    
    
    # R = np.linspace(0.01, 1.5, 75)
    dSigma=np.squeeze(tnfw.projected_excess(r))/1000000


    
    # dSigma=nfw.projected_excess(R)
    halo_table = np.genfromtxt(f'{bin}.txt', delimiter='\t', usecols=(0, 1), dtype=float)
    halo_r=halo_table[:,0]/1000
    
    halo_ds=halo_table[:,1]
    
    f = interp1d(halo_r, halo_ds, kind='cubic')
    halo_dSigma=f(r)*A

    summed_halo=np.add(dSigma,halo_dSigma)/1000000

    return summed_halo

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

# Define the model function

# Define the objective function (sum of squared errors)
def objective_function(params):
    predicted = subhalo_profile(rp, *params)
    error = (ds - predicted)/ds_err
    return np.sum(error**2)

# Initial guess for the model parameters
initial_params = [12, 5.83, 1]
bounds=[(1,None),(0,None),(None,None)]

# Minimize the objective function to fit the model
result = minimize(objective_function, initial_params, bounds=bounds)

# Extract the optimized parameters
lens_mass, lens_tau , param_A = result.x

lens_z=z
fit = subhalo_profile(rp, lens_mass, lens_tau, param_A)


r_full,ds_halo,ds_sub=make_profile(math.pow(10,lens_mass), lens_z, lens_tau, param_A, B=1 ,distbin=bin, plot=False)
# r_full,ds_full=make_profile(1e12, 0.35, 35, 0.6, distbin=bin, plot=False)
# plt.plot(rp, ds, 'bo', label='Isaac Data')
plt.plot(rp, fit, 'r-', label='interp Curve')
plt.plot(r_full, ds_halo+ds_sub, 'g-', label='fitted Curve')
plt.errorbar(rp, ds, ds_err, fmt='o',label='dsigma Data')
plt.xlabel('R (Mpc)')
plt.ylabel('M/pc^2')
plt.grid()
plt.ylim(-40,120)
plt.title(f'{bin} lens log(mass): {lens_mass:.2f}, Z: {lens_z:.2}, Rt/Rs: {lens_tau:.2f}, A: {param_A:.2}')
plt.legend()
plt.show()
