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

# File paths
clusters_file = Path(r'C:\catalogs\clusters_w_centers.fit')
galaxies_file = Path(r'C:\catalogs\unions_shapepipe_2022_v1.3.fits')
cluster_mass_file = Path(
    r'C:\Users\romix\Documents\GitHub\DeltaSigma2023\forJack\members_n_clusters_200m.fits')

# Load cluster centers
with fits.open(clusters_file) as hdul:
    cluster_data = hdul[1].data
    cluster_ra = cluster_data['RAJ2000']
    cluster_dec = cluster_data['DEJ2000']

# Load galaxy catalog
with fits.open(galaxies_file) as hdul:
    galaxy_data = hdul[1].data
    galaxy_ra = galaxy_data['RA']
    galaxy_dec = galaxy_data['Dec']

with fits.open(cluster_mass_file) as hdul:
    mass_data = hdul[1].data

    # Get unique IDs and their first occurrence indices
    _, idx = np.unique(mass_data['ID'], return_index=True)

    # Use indices to get unique rows
    mass_unique = mass_data[np.sort(idx)]

    cluster_mass = mass_unique['M_halo']
    cluster_z = mass_unique['Z_halo']

# Convert coordinates to radians for BallTree
cluster_coords = np.vstack([np.radians(cluster_ra), np.radians(cluster_dec)]).T
galaxy_coords = np.vstack([np.radians(galaxy_ra), np.radians(galaxy_dec)]).T

# Build BallTree for fast nearest-neighbor search
print("Building BallTree for galaxies...")
# Haversine accounts for curvature
galaxy_tree = BallTree(galaxy_coords, metric='haversine')

# Search radius: 9 arcminutes → radians
# radius = (3 / 60) * (np.pi / 180)  # Convert arcmin to radians

# Find galaxies within 9 arcmin for each cluster
print("Matching clusters...")
counts_per_cluster = []
used_clusters = 0  # Track how many clusters have at least 1 galaxy

for i in trange(len(cluster_coords), desc="Clusters matched"):

    R200m = mass_so.M_to_R(cluster_mass[i], cluster_z[i], '200m')  # kpc

    arcsec_per_kpc = cosmo_dist.arcsec_per_kpc_proper(cluster_z[i])

    radius = R200m * arcsec_per_kpc / 206265

    indices = galaxy_tree.query_radius([cluster_coords[i]], r=radius)[0]
    count = len(indices)

    if count > 0:  # Only store clusters with at least 1 galaxy
        counts_per_cluster.append(count)
        used_clusters += 1

# Calculate total and average
total_galaxies = np.sum(counts_per_cluster)
average_galaxies = np.median(counts_per_cluster) if counts_per_cluster else 0

# Count clusters with at least 10 galaxies
clusters_with_10_or_more = np.sum(np.array(counts_per_cluster) >= 10)

# Print results
print("\n--- Results ---")
print(f"Total number of galaxies within R200 arcmin: {total_galaxies}")
print(f"Average number of galaxies per cluster: {average_galaxies:.2f}")
print(
    f"Number of clusters with at least 10 galaxies: {clusters_with_10_or_more}")
print(
    f"Number of clusters actually used: {used_clusters} (out of {len(cluster_ra)})")
