# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:35:03 2024

@author: romix
"""

import time
import warnings
from numbers import Number
from colossus.halo import concentration, profile_nfw
from scipy.interpolate import interp1d
import numpy as np
from colossus.cosmology import cosmology
params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

from colossus.halo import concentration, profile_nfw
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.table import Table



halo_mass=1e15
halo_z=0.3
mdef='200m'
cdf_resolution=1000
mass_per_point = 1098372.008822474*10000

c=concentration.concentration(
    M=halo_mass, mdef="200m", z=halo_z, model='duffy08'
) #calculate concentration using colossus
halo_profile = profile_nfw.NFWProfile(M=halo_mass, c=c, z=halo_z, mdef=mdef) #build host halo NFW
# halo_profile = profile_einasto.EinastoProfile(M=halo_mass, c=c, z=halo_z, mdef=mdef) #build host halo NFW

scale_radius = halo_profile.getParameterArray()[1] #get NFW profile parameter R_scale
virial_radius = scale_radius * c #R_vir= concentration * R_scale
#
# Determine CDF of projected (2D) NFW enclosed mass
#CDF - cumulative distribution function
#
interp_radii = np.linspace(0, virial_radius*4, cdf_resolution) #distance for cdf
# interp_radii = np.logspace(-4, np.log10(virial_radius*4), cdf_resolution) #distance for cdf


# Temporarily ignore division by zero and overflow warnings
with np.errstate(divide="ignore", over="ignore"):
    interp_delta_sigmas = halo_profile.deltaSigma(interp_radii)
    interp_surface_densities = halo_profile.surfaceDensity(interp_radii)
    interp_enclosed_masses = halo_profile.enclosedMass(interp_radii)
# Correct delta sigmas and surface densities at r=0 to be zero
interp_delta_sigmas[0] = 0.0
interp_surface_densities[0] = 0.0
interp_2d_encl_masses = (
    np.pi * interp_radii**2 * (interp_delta_sigmas + interp_surface_densities)
)
interp_2d_encl_masses = interp_enclosed_masses
# plt.plot(interp_radii, interp_delta_sigmas, label='delta sigmas')
# plt.plot(interp_radii, interp_surface_densities, label='sigma R')
# plt.plot(interp_radii, interp_enclosed_masses, label='<R (colusus)')
# plt.plot(interp_radii, interp_2d_encl_masses, label='<R')
# plt.yscale('log')
# plt.legend()
# plt.show()


n_points = round(interp_2d_encl_masses[-1] / (mass_per_point)) 
print("For each offset, will generate", n_points, "points for this halo")
#
# Make 1D interpolator for this halo
#

interp_normed_2d_encl_masses = interp1d(
    interp_2d_encl_masses / interp_2d_encl_masses[-1],
    interp_radii,
    assume_sorted=True,
)



#
# Generate random points for this halo + offset combination
#
rng = np.random.default_rng()

# offset=0
# offset_angle = rng.uniform(0, 2 * np.pi)
# offset_x = offset * np.cos(offset_angle)
# offset_y = offset * np.sin(offset_angle)
#
random_cdf_yvals = rng.uniform(0, 1, size=n_points)
# print("Begin interpolation")

random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)

plt.hist(random_radii, bins=2000)
# plt.yscale('log')
plt.xlim(-10,200)
plt.show()

random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
random_radii_x = random_radii * np.cos(random_azimuths) 
random_radii_y = random_radii * np.sin(random_azimuths)
# print("Begin extending list")

#if return_xy:

#else:
# random_r=np.array([ np.sqrt(random_radii_x**2 + random_radii_y**2)])
# random_theta=np.array([ np.arctan2(random_radii_y, random_radii_x)])
# print("Finished extending list after", time.time() - debug_start4)
# print()

plt.scatter( random_radii_x, random_radii_y, s=0.001)
plt.gca().set_aspect('equal')
plt.show()

# indices = np.where(np.abs(random_radii_x - 0) <= 10)[0]

# values = np.histogram(random_radii_y[indices], bins=200)

# plt.plot(values[1][:-1], values[0])
# plt.xlim(0, 2000)
# plt.show()

# plt.plot(interp_radii, halo_profile.surfaceDensity(interp_radii))
# plt.xlim(0, 2000)
# plt.show()

#%%

lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits")
#Combined by myself with host halo masses and redshifts - email me if you want it

lowlim=0.6
highlim=0.9
#filter lenses that are in a distance bin. You can also filter by membership probability and redshift
data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        # & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
        & (lenses["PMem"] > 0.8)
    )
lenses = lenses[data_mask] #updated table of lenses

print(np.mean(lenses['R']))
sat_x = np.mean(lenses['R']) * 1000*1.429 #Mpc*1000 convert coords to kpc
sat_y = 0

start_bin=0.001 * 1.429 #first ring lens-centric disatnce Mpc
end_bin=2.5 * 1.429 #final ring lens-centric distance
ring_incr=0.02 * 1.429 #distance between rings
ring_num=round((end_bin-start_bin)/ring_incr) #number of rings
ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000 #Mpc*1000=kpc, radii of all rings in kps
# threshold=ring_incr/2*1000 #the same small width for each ring.
threshold = 0.5

S=[np.pi*((r+threshold)**2-(r-threshold)**2) for r in ring_radii] #area if rings
# Calculate the distances for all random points at once
distances = np.sqrt((random_radii_x - sat_x)**2 + (random_radii_y - sat_y)**2)

# Create an empty array to store the counts for each ring
ring_counts = np.zeros(len(ring_radii), dtype=np.int64) #counts in the rings
circle_counts = np.zeros(len(ring_radii), dtype=np.int64) #counts in enclosed circles

# Iterate over each ring radius and count the points within each ring


for i in range(len(ring_radii)):
    ring_mask = np.logical_and(ring_radii[i] - threshold <= distances, 
                          distances <= ring_radii[i] + threshold) #mask points that are within ring
    
    # plt.scatter(random_radii_x, random_radii_y, s=0.001)
    # plt.scatter(random_radii_x[mask], random_radii_y[mask], s=0.001, c='red')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    
    ring_counts[i] = np.sum(ring_mask)*mass_per_point/S[i] #get surface density in rings
    

    circle_mask = np.logical_and(0 <= distances, distances <= ring_radii[i] - threshold)
    
    
    circle_counts[i] = np.sum(circle_mask)*mass_per_point/(np.pi*(ring_radii[i]- threshold)**2)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].scatter(random_radii_x, random_radii_y, s=0.001, alpha=0.4)
    axs[0].scatter(random_radii_x[circle_mask], random_radii_y[circle_mask], s=0.001, c='red')
    axs[0].scatter(random_radii_x[ring_mask], random_radii_y[ring_mask], s=0.001, c='k')
    axs[0].plot(0, 0, "b+")
    axs[0].plot(sat_x, sat_y, "k+")
    # axs[0].set_title('Scatter Plot')
    # axs[0].set_xlabel('X-axis')
    # axs[0].set_ylabel('Y-axis')
    axs[0].set_aspect('equal', adjustable='box')
    
    axs[1].plot(ring_radii, ring_counts, label=r'$\Sigma$(R)', ls='--')
    axs[1].plot(ring_radii, circle_counts, label=r'$\Sigma$(<R)', ls='--')
    axs[1].plot(ring_radii, circle_counts-ring_counts, label=r'Delta $\Sigma$')
    axs[1].axvline(sat_x, -0.5e8, 3e8, c='r')
    # axs[1].set_title('Cumulative Counts')
    axs[1].set_xlabel('R (kpc)')
    axs[1].set_ylabel('dsigma')
    axs[1].set_xlim([ring_radii[0], ring_radii[-1]])
    axs[1].legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()