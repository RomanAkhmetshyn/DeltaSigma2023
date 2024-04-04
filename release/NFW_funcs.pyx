# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:34:27 2023

@author: Roman Akhmetshyn
e-mail: romix_aa@ukr.net

sample_nfw - cythonized Isaac's code
quick_MK_profile - get Monte-Carlo random points that follow NFW profile
"""

import time
import warnings
from numbers import Number
import cython
from colossus.halo import concentration, profile_nfw
from scipy.interpolate import interp1d
import numpy as np
cimport numpy as np
np.import_array()

# Compile Typing Definitions
ctypedef np.float64_t F64
from colossus.cosmology import cosmology
cdef dict params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")



@cython.boundscheck(False)
@cython.wraparound(False)

def sample_nfw(
    np.ndarray[F64, ndim=1] masses,
    np.ndarray[F64, ndim=1] redshifts,
    double mass_per_point,
    np.ndarray[F64, ndim=2] offsets,  # I'm leaving the input checking to the user... Must be scalar or 2D
    np.ndarray[F64, ndim=1] cs=None,
    np.ndarray[F64, ndim=2] seeds=None,  # I'm leaving the input checking to the user... Must be scalar or 2D
    str concentration_model="duffy08",
    str mdef="200m",
    int cdf_resolution=1000,
    bint return_xy=False,
    bint verbose=False,
):
    """
    Monte Carlo sample points that follow a (projected) NFW density profile. That is, the
    density of the points is proportional to the 2D projected density of the NFW halo.

    WARNING: I AM LEAVING THE INPUT CHECKING OF `offsets` AND `seeds` to the USER!
    **`offsets` and `seeds` must be either a SCALAR or a 2D ARRAY_LIKE, NOT 1D ARRAYS!**

    Parameters
    ----------
      masses :: scalar or (N,) 1D array_like of floats
        The masses of the halos in units of Msun/h. Must have the same shape as
        `redshifts`.

      redshifts :: scalar or (N,) 1D array_like of floats
        The redshifts of the halos. Must have the same shape as `masses`.

      mass_per_point :: int or float
        The 2D projected mass that each random point represents, in units of Msun/h. The
        larger this number, the fewer number of points generated per halo. Also, more
        massive halos will have more points representing its NFW profile compared to less
        massive halos. Please be reasonable when choosing this number (i.e., not to small
        or too large).

      offsets :: float or (N,M) 2D array_like of floats (optional)
        The offsets to use for the halo centres, in kpc/h. If a single value, the same
        offset is applied to all halos. If 2D array_like, the length of the 1st axis must
        match the length of `masses`; the lengths of the 2nd axes need not be the same. If
        2D array_like, the shape must match the shape of `seeds`.

      (OLD DESCRIPTION) offsets :: float or (N,) or (N,M) array_like of floats (optional)
        The offsets to use for the halo centres, in kpc/h. If a single value, the same
        offset is applied to all halos. If 1D array_like, the length must match the length
        of `masses`. If 2D array_like, the length of the 1st axis must match the length of
        `masses`; the lengths of the 2nd axes need not be the same. If 2D array_like, the
        shape must match the shape of `seeds`.

      cs :: (N,) array_like of floats or None (optional)
        The concentrations of the NFW halos. If None, the concentrations will be
        determined using the halo mass, redshift, concentration model (e.g., Duffy et al.
        2008), and the mass definition (e.g., 200x the matter overdensity).

      seeds :: int or (N,M) 2D array_like of ints or None (optional)
        The seed or seeds to use for the random number generator. If an integer, the same
        seed is used for all halos. If 2D array_like, the shape must match the shape of
        the 2D `offsets`.

      (OLD DESCRIPTION) seeds :: int or (N,) or (N,M) array_like of ints or None
      (optional)
        The seed or seeds to use for the random number generator. If an integer, the same
        seed is used for all halos. If array_like, the shape must match the shape of
        `offsets`.

      concentration_model :: str (optional)
        The concentration model to use if `cs` is None. See `colossus.halo.concentration`
        documentation (<https://bdiemer.bitbucket.io/colossus/halo_concentration.html>)
        for more details.

      mdef :: str (optional)
        The spherical overdensity mass definition to use if `cs` is None. See the colossus
        documentation for more details:
        <https://bdiemer.bitbucket.io/colossus/halo_mass.html>.

      cdf_resolution :: int (optional)
        The number of points between [0, virial radius] to use in the inverse transform
        sampling. Note that setting the cdf_resolution to a very high number (e.g.,
        100000) will be very slow!

        (Technically, this should be called `r_resolution` or whatnot, but this might also
        be confusing because we are generating random radii using a CDF, and the CDF was
        generated using radii at this resolution... Hence `cdf_resolution`...)

      return_xy :: bool (optional)
        If True, return (x, y) coordinates instead of (r, theta).

      verbose :: bool (optional)
        If True, prints out information about the time taken for some of the steps and
        other information that may be useful for debugging.

    Returns
    -------
      random_x_or_r :: 1D np.ndarray
        The radii (or x-coordinates, if `return_xy` is True) of the random points.

      random_y_or_theta :: 1D np.ndarray
        The angle (or y-coordinates, if `return_xy` is True) of the random points.
    """
    #
    # Check some inputs. If the user tries moderately hard, they can still break this
    #
    cdef bint is_1d_mass = False
    cdef list masses_list
    cdef list redshifts_list
    cdef list cs_list
    cdef list seeds_list=[]
    cdef np.ndarray[F64, ndim=2] orig_seed
    cdef tuple masses_shape
    
    cdef dict params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
    cosmology.addCosmology("737", params)
    cosmo = cosmology.setCosmology("737")
    
    if isinstance(masses, Number):
        masses_list = [masses]
        is_1d_mass = True

    if len(np.shape(masses)) != 1:
        raise ValueError("`masses` must be a scalar or 1D array_like")
    
    masses_shape = np.shape(masses)  # masses already 1D
    #
    if isinstance(redshifts, Number):
        redshifts_list = [redshifts]
    
    
    if masses_shape != np.shape(redshifts):
        raise ValueError("`masses` and `redshifts` must have the same length")
    #
    if not isinstance(mass_per_point, Number):
        raise ValueError("`mass_per_point` must be an int or float")
    #
    if isinstance(offsets, Number):
        if verbose:
            print("offset is number")
        offsets = np.full((len(masses), 1), offsets, dtype=np.float64)  # 2D array
    #
    # leave input checking to the user...
    #
    # elif len(np.shape(offsets)) == 1:
    #     if verbose:
    #         print("offset is 1D")
    #     if not is_1d_mass and np.shape(offsets) != masses_shape:
    #         raise ValueError("1D `offsets` and `masses` must have the same length")
    #     offsets = np.asarray(offsets)[np.newaxis].T  # convert to 2D array
    # elif len(np.shape(offsets)) == 2:
    #     if verbose:
    #         print("offset is 2D")
    #     if np.shape(offsets)[0] != len(masses):
    #         raise ValueError(
    #             "The 1st dimension of 2D `offsets` must have the same length as `masses`"
    #             )
    # else:
    #     raise ValueError("`offsets` must be a scalar, 1D array_like, or 2D array_like")
    #
    # At this point, `offsets` is a 2D list/array
    #
    if cs is not None:
        if np.shape(cs) != masses_shape:
            raise ValueError("`cs` and `masses` must have the same length")
    else:
        cs_list = [None] * len(masses)
    #
    if seeds is None or isinstance(seeds, (int, np.integer)):
        if verbose:
            print("seeds is None or a number")
        orig_seed = seeds
        # seeds = []
        for i in range(len(offsets)):
            seeds_list.append([orig_seed] * len(offsets[i]))
    #
    # leave input checking to the user...
    #
    # else:
    #     # Check that `seeds` has same shape as `offsets`
    #     if len(seeds) != len(offsets):
    #         raise ValueError("`seeds` must be None or match the shape of `offsets`")
    #     elif len(np.shape(seeds)) == 1:
    #         if verbose:
    #             print("seeds is 1d")
    #         seeds = np.asarray(seeds)[np.newaxis].T  # convert to 2D array
    #     elif len(np.shape(seeds)) == 2:
    #         if verbose:
    #             print("seeds is 2D")
    #         for i in range(len(offsets)):
    #             print(i, len(seeds[i]), len(offsets[i]))
    #             if len(seeds[i]) != len(offsets[i]):
    #                 raise ValueError(
    #                     "The shape of `seeds` must match the shape of `offsets`"
    #                     )
    #     else:
    #         raise ValueError(
    #             "`seeds` must be at most 2D and match the shape of `offsets`"
    #             )
    #
    # At this point, `seeds` is a 2D list/array with exact same shape as `offsets`
    #
    if verbose:
        print("Offsets:", offsets)
        print("Seeds:", seeds_list)
    #
    # Iterate over halos and generate random points
    #
    cdef np.ndarray interp_2d_encl_masses
    
    cdef np.ndarray[F64, ndim=1] random_x_or_r =  np.empty(0, dtype=np.float64)
    cdef np.ndarray[F64, ndim=1] random_y_or_theta = np.empty(0, dtype=np.float64)
    for mass, redshift, c, offset_1d, seed_1d in zip(
        masses, redshifts, cs_list, offsets, seeds_list
    ):
        #
        # Define NFW halo object
        #
        if c is None:
            c = concentration.concentration(
                M=mass, mdef=mdef, z=redshift, model=concentration_model
            )
            
        
        halo_profile = profile_nfw.NFWProfile(M=mass, c=c, z=redshift, mdef=mdef)
        central_density, scale_radius = halo_profile.getParameterArray()
        virial_radius = scale_radius * c
        #
        # Determine CDF of projected (2D) NFW enclosed mass
        #
        interp_radii = np.linspace(0, virial_radius, cdf_resolution)
        if verbose:
            print("-----\nBegin calculating enclosed mass with colossus")
        debug_start = time.time()
        # Temporarily ignore division by zero and overflow warnings
        with np.errstate(divide="ignore", over="ignore"):
            interp_delta_sigmas = halo_profile.deltaSigma(interp_radii)
            interp_surface_densities = halo_profile.surfaceDensity(interp_radii)
        # Correct delta sigmas and surface densities at r=0 to be zero
        interp_delta_sigmas[0] = 0.0
        interp_surface_densities[0] = 0.0
        interp_2d_encl_masses = (
            np.pi * interp_radii**2 * (interp_delta_sigmas + interp_surface_densities)
        )
        if verbose:
            print(
                "Finished calculating enclosed mass with colossus after",
                time.time() - debug_start,
            )
        #
        # Determine number of points to generate for this halo
        #
        
        n_points = round(interp_2d_encl_masses[-1:][0] / (mass_per_point * len(offset_1d)))
        if n_points == 0:
            # Not using warning module for now
            print(
                "WARNING: The mass per point is larger than the projected halo mass "
                + "at its virial radius. This will result in no points being "
                + "generated for this halo."
            )
            print(f"This halo has mass: {mass:.2f}, redshift: {redshift:.4f}")
            continue
        if verbose:
            print("For each offset, will generate", n_points, "points for this halo")
        #
        # Make 1D interpolator for this halo
        #
        if verbose:
            print("Begin creating 2D NFW CDF interpolator")
        debug_start2 = time.time()
        interp_normed_2d_encl_masses = interp1d(
            interp_2d_encl_masses / interp_2d_encl_masses[-1:][0],
            interp_radii,
            assume_sorted=True,
        )
        if verbose:
            print(
                "Finished creating 2D NFW CDF interpolator after",
                time.time() - debug_start2,
            )
            print()
        for offset, seed in zip(offset_1d, seed_1d):
            #
            # Generate random points for this halo + offset combination
            #
            rng = np.random.default_rng(seed)
            if verbose:
                print("Offset (kpc/h):", offset, "\tSeed:", seed)
            offset_angle = rng.uniform(0, 2 * np.pi)
            offset_x = offset * np.cos(offset_angle)
            offset_y = offset * np.sin(offset_angle)
            #
            random_cdf_yvals = rng.uniform(0, 1, size=n_points)
            if verbose:
                print("Begin interpolation")
            debug_start3 = time.time()
            random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)
            if verbose:
                print("Finished interpolation after", time.time() - debug_start3)
            random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
            random_radii_x = random_radii * np.cos(random_azimuths) + offset_x
            random_radii_y = random_radii * np.sin(random_azimuths) + offset_y
            if verbose:
                print("Begin extending list")
            debug_start4 = time.time()
            if return_xy:
                random_x_or_r=np.concatenate([random_x_or_r, random_radii_x])
                random_y_or_theta=np.concatenate([random_y_or_theta, random_radii_y])
            else:
                random_x_or_r=np.concatenate([random_x_or_r, np.sqrt(random_radii_x**2 + random_radii_y**2)])
                random_y_or_theta=np.concatenate([random_y_or_theta, np.arctan2(random_radii_y, random_radii_x)])
            if verbose:
                print("Finished extending list after", time.time() - debug_start4)
                print()
    return random_x_or_r, random_y_or_theta


def quick_MK_profile(double halo_mass,
                     double halo_z,
                     double mass_per_point,
                     str concentration_model="duffy08",
                     str mdef="200m",
                     int cdf_resolution=1000):
    """
    

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
    
    c=concentration.concentration(
        M=halo_mass, mdef="200m", z=halo_z, model=concentration_model
    ) #calculate concentration using colossus

    halo_profile = profile_nfw.NFWProfile(M=halo_mass, c=c, z=halo_z, mdef=mdef) #build host halo NFW

    scale_radius = halo_profile.getParameterArray()[1] #get NFW profile parameter R_scale
    virial_radius = scale_radius * c #R_vir= concentration * R_scale
    #
    # Determine CDF of projected (2D) NFW enclosed mass
    #CDF - cumulative distribution function
    #
    interp_radii = np.linspace(0, virial_radius, cdf_resolution) #distance for cdf
    

    # Temporarily ignore division by zero and overflow warnings
    with np.errstate(divide="ignore", over="ignore"):
        interp_delta_sigmas = halo_profile.deltaSigma(interp_radii)
        interp_surface_densities = halo_profile.surfaceDensity(interp_radii)
    # Correct delta sigmas and surface densities at r=0 to be zero
    interp_delta_sigmas[0] = 0.0
    interp_surface_densities[0] = 0.0
    interp_2d_encl_masses = (
        np.pi * interp_radii**2 * (interp_delta_sigmas + interp_surface_densities)
    )



    n_points = round(interp_2d_encl_masses[-1:][0] / (mass_per_point))
    # print("For each offset, will generate", n_points, "points for this halo")
    #
    # Make 1D interpolator for this halo
    #

    interp_normed_2d_encl_masses = interp1d(
        interp_2d_encl_masses / interp_2d_encl_masses[-1:][0],
        interp_radii,
        assume_sorted=True,
    )

    

    #
    # Generate random points for this halo + offset combination
    #
    rng = np.random.default_rng()
    offset=0
    offset_angle = rng.uniform(0, 2 * np.pi)
    offset_x = offset * np.cos(offset_angle)
    offset_y = offset * np.sin(offset_angle)
    #
    random_cdf_yvals = rng.uniform(0, 1, size=n_points)
    # print("Begin interpolation")

    random_radii = interp_normed_2d_encl_masses(random_cdf_yvals)

    random_azimuths = rng.uniform(0, 2 * np.pi, size=n_points)
    random_radii_x = random_radii * np.cos(random_azimuths) + offset_x
    random_radii_y = random_radii * np.sin(random_azimuths) + offset_y
    # print("Begin extending list")

    #if return_xy:

    #else:
    # random_r=np.array([ np.sqrt(random_radii_x**2 + random_radii_y**2)])
    # random_theta=np.array([ np.arctan2(random_radii_y, random_radii_x)])
    # print("Finished extending list after", time.time() - debug_start4)
    # print()
    
    return random_radii_x, random_radii_y