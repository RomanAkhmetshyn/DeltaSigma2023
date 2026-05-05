import numpy as np
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from sklearn.neighbors import BallTree
from tqdm import trange
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn.neighbors import BallTree
from pathlib import Path
from tqdm import trange
from colossus.halo import mass_so
from astropy.cosmology import FlatLambdaCDM
cosmo_dist = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.049)
from colossus.cosmology import cosmology
params = {"flat": True, "H0": 70, "Om0": 0.3,
          "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

# --------------------------------------------------
# File paths
# --------------------------------------------------
clusters_file = Path(r'C:\catalogs\clusters_w_centers.fit')
galaxies_file = Path(r'C:\catalogs\unions_shapepipe_2022_v1.3.fits')
cluster_mass_file = Path(
    r'C:\Users\romix\Documents\GitHub\DeltaSigma2023\forJack\members_n_clusters_200m.fits')
members_file = Path(r'C:\catalogs\members_n_clusters_masked.fits')

# --------------------------------------------------
# Load cluster centers
# --------------------------------------------------
with fits.open(clusters_file) as hdul:
    cluster_data = hdul[1].data
    cluster_ids_all = cluster_data['ID']
    cluster_ra_all = cluster_data['RAJ2000']
    cluster_dec_all = cluster_data['DEJ2000']

# --------------------------------------------------
# Load galaxy catalog
# --------------------------------------------------
with fits.open(galaxies_file) as hdul:
    galaxy_data = hdul[1].data
    galaxy_ra = galaxy_data['RA']
    galaxy_dec = galaxy_data['Dec']

# --------------------------------------------------
# Load cluster mass + redshift (unique per ID)
# --------------------------------------------------
with fits.open(cluster_mass_file) as hdul:
    mass_data = hdul[1].data

_, idx = np.unique(mass_data['ID'], return_index=True)
mass_unique = mass_data[np.sort(idx)]

mass_ids = mass_unique['ID']
cluster_mass_all = mass_unique['M_halo']
cluster_z_all = mass_unique['Z_halo']

# --------------------------------------------------
# Load member catalog (for selecting lens clusters)
# --------------------------------------------------
lenses_for_clusters = Table.read(members_file)

# --------------------------------------------------
# Build BallTree for galaxies
# --------------------------------------------------
galaxy_coords = np.vstack(
    [np.radians(galaxy_ra), np.radians(galaxy_dec)]
).T

print("Building BallTree for galaxies...")
galaxy_tree = BallTree(galaxy_coords, metric='haversine')

# --------------------------------------------------
# Radial bins
# --------------------------------------------------
bins = ['0103', '0306', '0609']

for bin_name in bins:

    print(f"\n========== BIN {bin_name} ==========")

    if bin_name == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin_name == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    # ----------------------------------------------
    # Select lenses in this radial bin
    # ----------------------------------------------
    mask = (
        (lenses_for_clusters["R"] >= lowlim)
        & (lenses_for_clusters["R"] < highlim)
        & (lenses_for_clusters["PMem"] > 0.8)
    )

    lenses_bin = lenses_for_clusters[mask]

    # Get unique cluster IDs hosting these lenses
    selected_cluster_ids = np.unique(lenses_bin["ID"])

    print("Number of unique clusters in bin:", len(selected_cluster_ids))

    # ----------------------------------------------
    # Match these IDs to cluster catalog
    # ----------------------------------------------
    cluster_mask = np.isin(cluster_ids_all, selected_cluster_ids)

    cluster_ra = cluster_ra_all[cluster_mask]
    cluster_dec = cluster_dec_all[cluster_mask]
    cluster_ids = cluster_ids_all[cluster_mask]

    # Match mass/redshift to same IDs
    mass_mask = np.isin(mass_ids, selected_cluster_ids)
    cluster_mass = cluster_mass_all[mass_mask]
    cluster_z = cluster_z_all[mass_mask]

    # Ensure consistent ordering by ID
    order = np.argsort(cluster_ids)
    cluster_ids = cluster_ids[order]
    cluster_ra = cluster_ra[order]
    cluster_dec = cluster_dec[order]

    mass_order = np.argsort(mass_ids[mass_mask])
    cluster_mass = cluster_mass[mass_order]
    cluster_z = cluster_z[mass_order]

    # Convert coordinates to radians
    cluster_coords = np.vstack(
        [np.radians(cluster_ra), np.radians(cluster_dec)]
    ).T

    # ----------------------------------------------
    # Match galaxies within R200m
    # ----------------------------------------------
    counts_per_cluster = []
    used_clusters = 0

    print("Matching clusters...")

    for i in trange(len(cluster_coords), desc="Clusters matched"):

        R200m = mass_so.M_to_R(cluster_mass[i],
                               cluster_z[i],
                               '200m')  # kpc

        arcsec_per_kpc = cosmo_dist.arcsec_per_kpc_proper(cluster_z[i])

        radius = R200m * arcsec_per_kpc / 206265  # radians

        indices = galaxy_tree.query_radius(
            [cluster_coords[i]],
            r=radius
        )[0]

        count = len(indices)

        if count > 0:
            counts_per_cluster.append(count)
            used_clusters += 1

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------
    total_galaxies = np.sum(counts_per_cluster)
    median_galaxies = (
        np.median(counts_per_cluster) if counts_per_cluster else 0
    )
    clusters_with_10_or_more = np.sum(
        np.array(counts_per_cluster) >= 10
    )

    print("\n--- Results ---")
    print(f"Total galaxies within R200m: {total_galaxies}")
    print(f"Median galaxies per cluster: {median_galaxies:.2f}")
    print(f"Clusters with ≥10 galaxies: {clusters_with_10_or_more}")
    print(f"Clusters used: {used_clusters} / {len(cluster_coords)}")
