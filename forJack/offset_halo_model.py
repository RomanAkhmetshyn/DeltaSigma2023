# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:17:52 2024

@author: romix
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from colossus.cosmology import cosmology
from colossus.halo import concentration, profile_nfw
from scipy.interpolate import interp1d
from tqdm import trange

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import cumulative_trapezoid


def random_points(halo_mass,
                  halo_z,
                  mass_per_point,
                  c,
                  concentration_model="duffy08",
                  mdef="200m",
                  cdf_resolution=1000,
                  trapz_mode=False):
    """

    This function calculates random points that follow NFW profile

    Parameters
    ----------
    double halo_mass : double
        Mass of the host halo in M_sun.
    double halo_z : double
        Host halo redshift.
    double mass_per_point : double
        Mass asigned for every random point in M-C.
    str concentration_model : str, optional
        Concentration model for colossus calculation. The default is "duffy08".
    str mdef : str, optional
        Mass model for colossus calculation. The default is "200m".
    int cdf_resolution : int, optional
        Number of points for interpolation of probability function for the profile.

    Returns
    -------
    random_radii_x : numpy array
        X coords of M-C point.
    random_radii_y : numpy array
        Y coords of M-C point.

    """

    halo_profile = profile_nfw.NFWProfile(
        M=halo_mass, c=c, z=halo_z, mdef=mdef)  # build host halo NFW

    # get NFW profile parameter R_scale
    scale_radius = halo_profile.getParameterArray()[1]
    virial_radius = scale_radius * c  # R_vir= concentration * R_scale
    #
    # Determine CDF of projected (2D) NFW enclosed mass
    # CDF - cumulative distribution function
    #
    interp_radii = np.linspace(
        0, virial_radius*4, cdf_resolution)  # distance for cdf

    # Temporarily ignore division by zero and overflow warnings
    with np.errstate(divide="ignore", over="ignore"):
        if not trapz_mode:
            interp_delta_sigmas = halo_profile.deltaSigma(interp_radii)
        interp_surface_densities = halo_profile.surfaceDensity(interp_radii)
        # interp_enclosed_masses = halo_profile.enclosedMass(interp_radii)
        
    rng = np.random.default_rng()
    if trapz_mode:
        interp_surface_densities[0] = 0.0
        cdf = cumulative_trapezoid(2*np.pi*interp_radii*interp_surface_densities, interp_radii, initial=0)
        inverse_cdf = interp1d(cdf/cdf[-1], interp_radii, bounds_error=False, fill_value="extrapolate")
        n_points = round(cdf[-1] / (mass_per_point))
        random_values = np.random.rand(n_points)
        random_radii = inverse_cdf(random_values)

    else:
        # Correct delta sigmas and surface densities at r=0 to be zero
        interp_delta_sigmas[0] = 0.0
        interp_surface_densities[0] = 0.0
        interp_2d_encl_masses = (
            np.pi * interp_radii**2 *
            (interp_delta_sigmas + interp_surface_densities)
        )
        # interp_2d_encl_masses = interp_enclosed_masses

        n_points = round(interp_2d_encl_masses[-1] / (mass_per_point))

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

        random_cdf_yvals = rng.uniform(0, 1, size=n_points)
        random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)

    random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
    random_radii_x = random_radii * np.cos(random_azimuths)
    random_radii_y = random_radii * np.sin(random_azimuths)

    return random_radii_x, random_radii_y


# setting global cosmology. Keep everything H=100, unless your data is different
params = {"flat": True, "H0": 70, "Om0": 0.3,
          "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

cosmo_dist = FlatLambdaCDM(H0=100, Om0=0.3, Ob0=0.049)  # cosmology for Rykoff

bins = ['0609', '0306', '0103']
# bins=[ '0103'] #input distance bin, i.e. distance of lens galaxy from cluster center in Mpc

for bin in bins:

    if bin == '0609':
        lowlim = 0.6
        highlim = 0.9
        scale = 0.018
    elif bin == '0306':
        lowlim = 0.3
        highlim = 0.6
        scale = 0.01
    elif bin == '0103':
        lowlim = 0.1
        highlim = 0.3
        scale = 0.008  # scale of rayleigh distr

    # RedMaPPer catalog -
    lenses = Table.read("members_n_clusters_masked.fits")
    # Combined by myself with host halo masses and redshifts - email me if you want it
    dist_file = Table.read(f'{bin}_members_dists.fits')
    cluster_file = Table.read(f'clusters_w_centers.fit')

    # filter lenses that are in a distance bin. You can also filter by membership probability and redshift
    data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        # & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
        & (lenses["PMem"] > 0.8)
    )
    lenses = lenses[data_mask]  # updated table of lenses
    # interpolation resulotion, i.e. number of points for probability distribution func.
    cdf_resolution = 1000
    # arbitrary number, Mass/mpp = number of points for M-C
    mass_per_point = 1098372.008822474*10000

    # start_bin=0.01 * 1.429 #first ring lens-centric disatnce Mpc
    start_bin = 0.002 * 1.429
    # end_bin=1.5 * 1.429 #final ring lens-centric distance
    end_bin = 2.5 * 1.429  # final ring lens-centric distance
    ring_incr = 0.02 * 1.429  # distance between rings
    ring_num = round((end_bin-start_bin)/ring_incr)  # number of rings
    # Mpc*1000=kpc, radii of all rings in kps
    ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000
    # threshold=ring_incr/2*100 #the same small width for each ring.
    threshold = ring_incr/4*100  # the same small width for each ring.
    # threshold = 0.5

    mdef = "200m"  # cosmological mass definition

    # np array for all delta sigma measurments
    DeltaSigmas = np.zeros((1, len(ring_radii)))
    debug_start = time.time()  # timing the whole script

    halo_dict = {}  # a dictionary for each host halo, so we don't calculate same thing repeatedly
    # lenses = lenses[:1000]
    # dist_file = dist_file[:1000]

    for s in trange(len(lenses)):  # iterate through each lens
        sat = lenses[s]

        # check if M-C was calculated for this ID (host halo ID)
        if sat['ID'] not in halo_dict:
            halo_dict = {}  # empty the dictionary
            c = concentration.concentration(
                M=sat['M_halo'], mdef="200m", z=sat['Z_halo'], model="duffy08"
            )  # calculate concentration using colossus

            time_rand = time.time()

            random_radii_x, random_radii_y = random_points(sat['M_halo'],
                                                           # here I multiplied by 1.429 cuz I calculated
                                                           # masses for H=70 cosmology
                                                           sat['Z_halo'],
                                                           mass_per_point,
                                                           c,
                                                           "duffy08",
                                                           "200m",
                                                           cdf_resolution)

            # add halo to the dictionary
            halo_dict[sat['ID']] = [random_radii_x, random_radii_y]

            print('calculated random points in: ', time.time() - time_rand)

        else:

            # next lenses will use first M-C coordinates
            random_radii_x, random_radii_y = halo_dict[sat['ID']]

        # time_coords = time.time()

        # get BCG cluster centers
        Ra0 = cluster_file['RA0deg'][cluster_file['ID'] == sat['ID']]
        Dec0 = cluster_file['DE0deg'][cluster_file['ID'] == sat['ID']]

        # get current satellite coords
        Ra_sat = sat['RAJ2000']
        De_sat = sat['DEJ2000']

        radii = np.random.rayleigh(scale, size=1)

        # Generate random angles uniformly between 0 and 2*pi
        angles = np.random.uniform(0, 2 * np.pi, size=1)

        Ra_random = Ra0 + radii * np.cos(angles)  # Random RA values around BCG
        # Random Dec values around BCG
        Dec_random = Dec0 + radii * np.sin(angles)

        # Create SkyCoord objects for coordinates of cluster center and satellite galaxy
        center = SkyCoord(ra=Ra_random, dec=Dec_random,
                          frame='icrs', unit="deg")
        satellite = SkyCoord(ra=Ra_sat, dec=De_sat, frame='icrs', unit="deg")

        # Calculate the angular separation in arcseconds
        sep = center.separation(satellite)

        # Convert angular separation to physical distance in kpc using RedMapper cosmology
        arcsec_per_kpc = cosmo_dist.arcsec_per_kpc_proper(sat['Z_halo'])
        sat_x = (sep.arcsecond/arcsec_per_kpc.value) * \
            1.429  # distance in our cosmology

        # sat_x = dist_file[s]['R0'] * 1000 * 1.429 #Mpc*1000 convert coords to kpc
        # sat_x = sat['R'] * 1000 * 1.429 #Mpc*1000 convert coords to kp
        sat_y = 0

        # print('calculated offset in: ', time.time() - time_coords)

        # time_area = time.time()

        S = [np.pi*((r+threshold)**2-(r-threshold)**2)
             for r in ring_radii]  # area of rings
        # Calculate the distances for all random points at once
        distances = np.sqrt((random_radii_x - sat_x)**2 +
                            (random_radii_y - sat_y)**2)

        # Create an empty array to store the counts for each ring
        # counts in the rings
        ring_counts = np.zeros(len(ring_radii), dtype=np.int64)
        # counts in enclosed circles
        circle_counts = np.zeros(len(ring_radii), dtype=np.int64)

        # Iterate over each ring radius and count the points within each ring
        for i in range(len(ring_radii)):
            mask = np.logical_and(ring_radii[i] - threshold <= distances,
                                  distances <= ring_radii[i] + threshold)  # mask points that are within ring

            # get surface density in rings
            ring_counts[i] = np.sum(mask)*mass_per_point/S[i]

        for i in range(len(ring_radii)):  # the same but iterate each circle
            # mask = np.logical_and(0 <= distances, distances <= ring_radii[i] - threshold) #!!!
            mask = np.logical_and(
                0 <= distances, distances <= ring_radii[i])  # !!!

            # circle_counts[i] = np.sum(mask)*mass_per_point/(np.pi*(ring_radii[i]- threshold)**2) #!!!
            circle_counts[i] = np.sum(
                mask)*mass_per_point/(np.pi*(ring_radii[i])**2)  # !!!

        sums = np.array([DeltalessR - DeltaR for DeltalessR,
                        DeltaR in zip(circle_counts, ring_counts)])

        DeltaSigmas = np.add(DeltaSigmas, np.array(sums))

        # print('calculated area differences and added it in: ',
        #       time.time() - time_area)

    t = time.time() - debug_start
    print(
        f"Finished calculating {s} sat after",
        t,
    )
    avgDsigma = DeltaSigmas/len(lenses)  # average Delta Sigma of all all lenses
    table = np.column_stack((ring_radii, avgDsigma[0]))
    np.savetxt(f'{bin}_rayleigh.txt', table,
               delimiter='\t', fmt='%f')  # save average delta sigma

    plt.plot(ring_radii, avgDsigma[0]/1e6,
             color='black', linewidth=0.5, linestyle='--')
    plt.show()
