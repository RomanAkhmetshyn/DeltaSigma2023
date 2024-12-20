import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM


# Define your custom cosmology with H0=70 and Om0=0.3
H0 = 100  # Hubble constant
cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Ob0=0.049)

# Read the cluster file and extract RA0 and Dec0
cluster_file = Table.read('C:/catalogs/clusters_w_centers.fit')
Ra0 = cluster_file['RA0deg'][0]  # Assuming this is the central RA in degrees
Deg0 = cluster_file['DE0deg'][0]  # Assuming this is the central Dec in degrees
print(Ra0, Deg0)
# Set an arbitrary fixed point RA1, Dec1
Ra1, Dec1 = 239.8523343, 27.2168499  # Example values in degrees

# Number of random points to generate for each cluster
size = 10000

# Generate random radial distances using Rayleigh distribution
scale = 0.08  # Example scale in degrees, adjust as needed
radii = np.random.rayleigh(scale, size=size)

# Generate random angles uniformly between 0 and 2*pi
angles = np.random.uniform(0, 2 * np.pi, size=size)

# Convert polar coordinates (radii, angles) into random RA/Dec around the central RA0, Dec0
Ra_random = Ra0 + radii * np.cos(angles)  # Random RA values around Ra0
Dec_random = Deg0 + radii * np.sin(angles)  # Random Dec values around Dec0

# Create SkyCoord objects for the random points and the arbitrary point Ra1, Dec1
c1 = SkyCoord(ra=Ra_random, dec=Dec_random, frame='icrs', unit="deg")
c2 = SkyCoord(ra=Ra1, dec=Dec1, frame='icrs', unit="deg")

# Calculate the angular separation in arcseconds
sep = c1.separation(c2)

# Set an arbitrary redshift value (z)
z = 0.09  # Example redshift, adjust as needed

# Convert angular separation to physical distance in kpc using your custom cosmology model
arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z)
distance = distance = (sep.arcsecond/arcsec_per_kpc.value)

# Plot the 2D random distribution
plt.figure(figsize=(6, 6))
plt.scatter(Ra1, Dec1, marker='*', zorder=5, label='sat')
plt.scatter(Ra0, Deg0, marker='^', zorder=5, label='BCG')
plt.scatter(Ra_random, Dec_random, s=0.1, alpha=0.6)
plt.title('Random Rayleigh Distribution around BCG')
plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')
# plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Plot a histogram of the distances in kpc
plt.figure(figsize=(8, 6))
plt.hist(distance, bins=50, alpha=0.7, color='b')
plt.vlines(1080, 0, 700, color='r', label='Rykoff dist')
plt.title(f'Histogram of Distances to satellite')
plt.xlabel('Distance (kpc)')
plt.ylabel('N')
plt.grid(True)
plt.legend()
plt.show()
