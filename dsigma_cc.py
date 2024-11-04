# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:49:15 2023

python example_dsigma_linux.py 1 ShapePipe redmapper -cdb 0.6 -ub
"""

"""
example_dsigma.py

Isaac Cheng - December 2022
EDITS: Jack Elvin-Poole - June 2023

Example script measures the excess surface density around a given lens sample using UNIONS sources

usage: example_dsigma.py [-h] [-ub] [-zc Z_CALIB] [-wc W_CALIB] jobid source_catalog lens_catalog cluster_dist_bin

Example:
    python example_dsigma.py 1 ShapePipe mergers -zc 2.0

jobid=1 is arbitrary
cluster_dist_bin=99 does nothing for merger lenses

run this to get help:
python example_dsigma.py --help

WARNING: check to make sure random subtraction is on/off (as desired) before using this
script! (see the `kwargs` variable).
"""
import argparse
import gc

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from dsigma.helpers import dsigma_table
from dsigma.jackknife import (
    compute_jackknife_fields,
    jackknife_resampling,
)
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density



######### hardcoded CONFIG. Set any paths you need here to your machine #########
lensfit_file = "<path to lens fit catalog>"
# shapepipe_file = "D:/unions_test7000.fits"
# shapepipe_file = "/home/romanakh/projects/def-mjhudson/romanakh/unions_shapepipe_2022_v1.0.fits"
shapepipe_file = "/home/romanakh/projects/def-mjhudson/romanakh/unions_shapepipe_2022_v1.3.fits"
# calib_file = "/home/romanakh/projects/def-mjhudson/romanakh/calib_ShapePipe_v1.1.fits"
calib_file = "/home/romanakh/projects/def-mjhudson/romanakh/calib_nz_shapepipe_1500_v1.3.fits"

# redmapper_file = "/home/romanakh/projects/def-mjhudson/romanakh/dr8_redmapper_v6.3.1_members_masked.fits"
redmapper_file = "/home/romanakh/projects/def-mjhudson/romanakh/redmapper_mnc_allz.fits"
mergers_file = "/Users/jackelvinpoole/UNIONS/mergers/data/merger_table_environment.fits"

randoms_file = "/home/romanakh/projects/def-mjhudson/romanakh/dr8_run_redmapper_v6.3.1_randcat_z0.05-0.60_lgt020.fit"

# radius_bins = np.linspace(0.01, 1.5, 16)  # Mpc, almost replicate Li+2016
radius_bins=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.792, 0.93, 1.075,  1.217, 1.358, 1.5]
radius_bins = [num * 1.429 for num in radius_bins]
cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
num_jackknife_regions = 100
distance_threshold = 10 # in degrees. Used for JK. should be > typical distance between lenses, 
                        # but < size of smallest contiguous field

def main(
    jobid,
    source_catalog,
    lens_catalog,
    cluster_dist_bin,
    use_boost=True,
    z_calib=2.0,
    w_calib=1.0,
):
    """
    Calculates the excess surface density measured from a galaxy's tangential shear using
    `dsimga` and outputs results in a .csv file.

    Parameters
    ----------
      jobid :: str
        The slurm job ID.

      source_catalog :: "Lensfit" or "ShapePipe"
        The catalog to use for the galaxy-galaxy lensing sources.

      lens_catalog :: 'redmapper' or 'mergers'
        The catalog to use for the galaxy-galaxy lensing lenses.
        You can add your own manually

      cluster_dist_bin :: 0.1, 0.3, or 0.6
        The cluster-centric distance bin of the satellite galaxies. In units of h^-1 Mpc,
        use 0.1 for [0.1, 0.3], 0.3 for [0.3, 0.6], and 0.6 for [0.6, 0.9].

      use_boost :: bool (optional)
        If True, apply the boost factor correction
        (<https://dsigma.readthedocs.io/en/latest/background.html#boost-factor>), which is
        important since our lenses are in galaxy clusters. Requires a random catalog to be
        provided.

      z_calib :: int or float (optional)
        The nominal redshift to assign to each source in the calibration and source
        catalogs. The value of `z_calib` should not affect final results.

      w_calib :: int or float (optional)
        The inverse variance weight to assign to each source in the calibration catalog.

    Returns
    -------
      None
    """
    print("Loaded all modules")
    #
    # Check arguments
    #
    if not isinstance(jobid, str):
        raise ValueError("`jobid` must be a string")
    if source_catalog != "Lensfit" and source_catalog != "ShapePipe":
        raise ValueError("`source_catalog` must be either 'Lensfit' or 'ShapePipe'")
    if lens_catalog != "redmapper" and lens_catalog != "mergers":
        raise ValueError("`lens_catalog` must be either 'redmapper' or 'mergers', you can add your own manually")
    if not isinstance(use_boost, bool):
        raise ValueError("`use_boost` must be either True or False.")

    if lens_catalog == "redmapper":
        cluster_dist_bins_end = {0.1: 0.3, 0.3: 0.6, 0.6: 0.9}
        if cluster_dist_bin not in cluster_dist_bins_end:
            raise ValueError(
                f"{cluster_dist_bin} is not a valid cluster-centric distance bin!"
            )
    if not isinstance(z_calib, (int, float)) or not isinstance(w_calib, (int, float)):
        raise ValueError("`z_calib` and `w_calib` must be ints or floats.")
    print("=====")
    print("Job ID:", jobid)
    print("Using boost factor:", use_boost)
    print("=====")
    #
    # Read source data
    #
    if source_catalog == "Lensfit":
        source_path = lensfit_file
        data_kwargs = {
            "ra": "ra",  # degres
            "dec": "dec",  # degrees
            "w": "w",  # ellipticity weight
            "e_1": "e1",  # 1st component of ellipticity
            "e_2": "e2",  # 2nd component of ellipticity
            "e_2_convention": "standard",  # "standard" or "flipped"
            "z": "z",  # best-fit photometric redshift
            "z_err": "z_err",  # uncertainty in photometric redshift
        }
    else:
        source_path = shapepipe_file
        data_kwargs = {
            "ra": "RA",  # degres
            "dec": "Dec",  # degrees
            "w": "w",  # ellipticity weight
            "e_1": "e1",  # 1st component of ellipticity
            "e_2": "e2",  # 2nd component of ellipticity
            "e_2_convention": "standard",  # "standard" or "flipped"
            "z": "z",  # best-fit photometric redshift
            "z_err": "z_err",  # uncertainty in photometric redshift
        }
    print(f"Ingesting {source_catalog} sources from {source_path}")
    data = fits.getdata(source_path, 1)
    print(len(data))
    #
    # Need fake redshift
    #
    print("Adding 'fake' source redshifts")
    # Construct empty record array
    new_dtype = np.dtype(data.dtype.descr + [("z", "f8"), ("z_err", "f8")])
    new_data = np.zeros(data.shape, dtype=new_dtype)
    # Copy over values
    for colname in data.columns.names:
        new_data[colname] = data[colname]
    # Set fake redshifts (these should be the same as the nominal redshift in the
    # calibration catalog!)
    new_data["z"] = z_calib
    new_data["z_err"] = 0.0  # z_err only affects error bars (not `ds` value)
    #
    # Free memory
    #
    del data
    gc.collect()
    #
    # Convert to format required by dsigma (an `astropy.table.Table` object)
    #
    print("Converting to dsigma table...")
    sources = dsigma_table(new_data, "source", **data_kwargs)
    #
    # Read calibration catalog
    #
    calib_path = calib_file.format(source_catalog=source_catalog)
    print(f"Reading calibration catalog from {calib_path}")
    calib = Table.read(calib_path)
    print(f"Setting nominal redshifts to {z_calib}...")
    calib["z"] = z_calib
    print(f"Setting the inverse variance weights to {w_calib}...")
    calib["w"] = w_calib
    print("Converting to dsigma table...")
    calib = dsigma_table(calib, "calibration", z_true="z_true", w_sys="w_sys", w="w")

    #
    # Read lens data
    #
    if lens_catalog == "redmapper":
        print(
            "Cluster-centric distance bin = "
            + f"[{cluster_dist_bin}, {cluster_dist_bins_end[cluster_dist_bin]})"
        )
        lenses_path = redmapper_file
        print(f"Reading lenses from {lenses_path}")
        lenses = Table.read(lenses_path)
        #
        # Slice lenses into cluster-centric radius bin
        #
        lenses_mask = (
            (lenses["R"] >= cluster_dist_bin)
            & (lenses["R"]< cluster_dist_bins_end[cluster_dist_bin])
            & (lenses["PMem"] > 0.8)
            # & (lenses["ZSPEC"] > -1)
        )
        lenses = lenses[lenses_mask]
        print(
            "Number of lenses in cluster-centric distance bin = "
            + f"[{cluster_dist_bin}, {cluster_dist_bins_end[cluster_dist_bin]}):",
            len(lenses),
        )
        print("Converting to dsigma table...")
        lenses = dsigma_table(
            lenses,
            "lens",
            # z="ZSPEC",  # spectroscopic redshift
            z="z_any",
            # ra="RA",  # right ascension in degrees
            # dec="DEC",  # declination in degrees
            ra="RAJ2000",  # right ascension in degrees
            dec="DEJ2000",  # declination in degrees
            w_sys=1,  # systematic weight
        )
    elif lens_catalog == "mergers":
        lenses_path = mergers_file
        print(f"Reading lenses from {lenses_path}")
        lenses = Table.read(lenses_path)


        print(
            "Number of lenses = ",
            len(lenses),
        )

        print("Converting to dsigma table...")
        lenses = dsigma_table(
            lenses,
            "lens",
            z="z_spec",  # spectroscopic redshift
            ra="ra",  # right ascension in degrees
            dec="decl",  # declination in degrees
            w_sys=1,  # systematic weight
        )
        #
    #
    # Read random catalog
    #
    if use_boost:
        randoms_path = randoms_file
        print(f"Using redMaPPer v6.3.1 random catalog from {randoms_path}")
        randoms = Table.read(randoms_path)
        # Verified "LAMBDA_IN" == "AVG_LAMBDAOUT" and "SIGMA_LAMBDAOUT" == 0 for all randoms
        randoms = randoms[(randoms["ZTRUE"] < 0.5) & (randoms["LAMBDA_IN"] >= 20)]
        print("Converting to dsigma table...")
        randoms = dsigma_table(
            randoms, "lens", z="ZTRUE", ra="RA", dec="DEC", w_sys="WEIGHT"
        )
    else:
        randoms = None
    #
    # Set lens-source separation cut so z_lens + 0.1 < z_source
    #
    #JACK: REMOVED THE SOURCE-LENS SEPARATION CUT
    #print("Setting lens-source separation cut...")
    #add_maximum_lens_redshift(sources, dz_min=0.1, apply_z_low=False)
    #add_maximum_lens_redshift(calib, dz_min=0.1, apply_z_low=False)
    #
    # Pre-compute lensing statistics
    #
    print("Pre-computing lensing statistics...")

    precompute(
        lenses,
        sources,
        radius_bins,
        table_c=calib,
        cosmology=cosmology,
        comoving=False,
    )
    if use_boost:
        precompute(
            randoms,
            sources,
            radius_bins,
            table_c=calib,
            cosmology=cosmology,
            comoving=False,
        )
    # with open(
    #     "./output/"
    #     + "example_precomputed_lenses_"
    #     + f"{catalog}_clusterDist{cluster_dist_bin}_randoms{use_boost}_{jobid}.pkl",
    #     "wb",
    # ) as f:
    #     dill.dump(lenses, f)
    #
    # Drop all lenses that do not have any nearby source
    #
    lenses["n_s_tot"] = np.sum(lenses["sum 1"], axis=1)
    lenses = lenses[lenses["n_s_tot"] > 0]
    num_lenses = len(lenses)
    print("Number of lenses with nearby sources:", num_lenses)
    if use_boost:
        randoms["n_s_tot"] = np.sum(randoms["sum 1"], axis=1)
        randoms = randoms[randoms["n_s_tot"] > 0]
        num_randoms = len(randoms)
        print("Number of randoms with nearby sources:", num_randoms)
    #
    # Divide into different jackknife fields
    #
    print("Dividing into jackknife fields...")
    global num_jackknife_regions
    if num_lenses < num_jackknife_regions:
        print("N JK > N lenses, setting Njk=Nlens")
        num_jackknife_regions = num_lenses
        
    
    jackknife_centers = compute_jackknife_fields(lenses, num_jackknife_regions, distance_threshold=distance_threshold, weights=lenses["n_s_tot"])
    if use_boost:
        jackknife_centers_rand = compute_jackknife_fields(randoms, jackknife_centers )
    s="""
    add_continous_fields(lenses, distance_threshold=2)  # no need to do this for randoms
    # NOTE: I added line 153 of
    # /home/i8cheng/projects/def-mjhudson/i8cheng/dsigma_env/lib/python3.10/site-packages/dsigma/jackknife.py
    # for debugging, so you probably won't get the same verbose output as me (i.e., the
    # line containing "(in jackknife.py)" in the slurm output file)
    num_lenses_fields = len(np.unique(lenses["field"]))
    print(f"Number of fields in lenses table: {num_lenses_fields}")
    # N.B. the following is required:
    # tot num jackknife regions > num fields, AND number of lenses with nearby sources in
    # each field (number of "samples") > num jackknife regions in that field
    #
    # Also, slicing the lenses into redshift & richness bins ahead of time causes the data
    # to be very spatially fragmented, hence the need for a small number of jackknife
    # regions per "survey field"
    #
    # Want to maximize tot number of jackknife regions (up to, e.g., 4 * num fields) but
    # ensure tot num jackknife regions ("n_jk") > num fields and n_jk < num lenses with
    # nearby sources
    # N.B. num_lenses >= num_lenses_fields
    num_jackknife_regions = np.min((4 * num_lenses_fields, num_lenses))
    print(f"Using {num_jackknife_regions} jackknife regions")
    jackknife_centers = jackknife_field_centers(
        lenses, num_jackknife_regions, weight="n_s_tot"
    )
    """

    # add_jackknife_fields(lenses, jackknife_centers)
    # if use_boost:
    #     # Use same jackknife centers as lenses
    #     add_jackknife_fields(randoms, jackknife_centers)
    
    # Choose correction factors and other options
    kwargs = {
        "return_table": True,
        "scalar_shear_response_correction": False,  # very important. Requires "m" column
        "shear_responsivity_correction": False,  # very important. Requires something...
        "boost_correction": use_boost,  # need random catalog
        # "random_subtraction": use_boost,  # highly recommended to set to True
        "random_subtraction": False,  # highly recommended to set to True
        "photo_z_dilution_correction": True,  # highly recommended
        "table_r": randoms,  # random catalog if `use_boost` is True
    }
    print(f"NO RANDOM SUBTRACTION! ONLY BOOST FACTOR ({use_boost}) + PHOT-Z CORRECTIONS!")

    #
    # Stack lensing signal
    #
    print("Stacking lensing signal...")
    result = excess_surface_density(lenses, **kwargs)
    kwargs["return_table"] = False
    # print(randoms.colnames)
    covmat = jackknife_resampling(excess_surface_density, lenses, **kwargs)
    result["ds_err"] = np.sqrt(
        np.diag(covmat)
    )
    #
    result.write(
        "./output/roman_esd_70"
        + f"{source_catalog}_{lens_catalog}_clusterDist{cluster_dist_bin}_randoms{use_boost}_{jobid}.csv",
    overwrite=True)

    #also saving full covarinace matrix
    np.savetxt(        "./output/roman_esd_70"
        + f"{source_catalog}_{lens_catalog}_clusterDist{cluster_dist_bin}_randoms{use_boost}_{jobid}_covmat.txt",
         covmat)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs `dsigma` on the Lensfit or ShapePipe data sets",
        prog="example_dsigma.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "jobid",
        type=str,
        help="The job ID of the slurm job. In a bash script, job ID can be accessed "
        + "using ${SLURM_JOB_ID}.",
    )
    parser.add_argument(
        "source_catalog",
        type=str,
        help="'Lensfit' or 'ShapePipe'. Specifies the catalog to use for galaxy-galaxy "
        + "lensing sources.",
    )
    parser.add_argument(
        "lens_catalog",
        type=str,
        help="'redmapper' or 'mergers'. The catalog to use for the galaxy-galaxy lensing lenses. "
        + "You can add your own manually",
    )
    parser.add_argument(
        "-cdb",
        "--cluster_dist_bin",
        type=float,
        default=99.,
        help="The cluster-centric distance bin of the satellite galaxies. In units of "
        + "h^-1 Mpc, use 0.1 for [0.1, 0.3], 0.3 for [0.3, 0.6], and 0.6 for [0.6, 0.9].",
    )
    parser.add_argument(
        "-ub",
        "--use-boost",
        action="store_true",
        default=False,
        help="If True, apply the boost factor correction, which is important since our "
        + "lenses are in galaxy clusters. Requires a random catalog to be provided.",
    )
    parser.add_argument(
        "-zc",
        "--z-calib",
        type=float,
        default=2.0,
        help="The nominal redshift to assign to each source in the calibration and "
        + "source catalog. The value of `z_calib` should not affect final results...",
    )
    parser.add_argument(
        "-wc",
        "--w-calib",
        type=float,
        default=1.0,
        help="The inverse variance weight to assign to each source in the calibration "
        + "catalog.",
    )

    args = vars(parser.parse_args())
    main(
        jobid=args["jobid"],
        source_catalog=args["source_catalog"],
        lens_catalog=args["lens_catalog"],
        cluster_dist_bin=args["cluster_dist_bin"],
        use_boost=args["use_boost"],
        z_calib=args["z_calib"],
        w_calib=args["w_calib"],
    )