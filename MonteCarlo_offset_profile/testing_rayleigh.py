import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import rayleigh
np.random.seed(0)
# Define your custom cosmology with H0=70 and Om0=0.3
H0 = 100  # Hubble constant
cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Ob0=0.049)


def sample_offset_radius(sigma, size=1):
    u = np.random.uniform(0, 1, size=size)
    R_off = sigma * np.sqrt(-2 * np.log(1 - u))
    return R_off


cluster_file = Table.read('C:/catalogs/clusters_w_centers.fit')
Ra0 = cluster_file['RA0deg'][0]  # bcg
Deg0 = cluster_file['DE0deg'][0]  # bcg
print(Ra0, Deg0)
# satellite RA1, Dec1
Ra1, Dec1 = 239.8523343, 27.2168499

# Number of random points to generate for each cluster
size = 10000
z = 0.09296000000000001  # redshift
arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z)

scale = 0.32*1000  # sigma
# radii_angular = rayleigh.rvs(scale, size=size)
radii = np.random.rayleigh(scale, size=size)*1  # kps

# radii = sample_offset_radius(scale, size)

radii_angular = radii * arcsec_per_kpc.value  # arcsec
radii_angular = radii_angular/3600  # deg

# plt.hist(radii_angular, bins=100)
# plt.show()

# Generate random angles uniformly between 0 and 2*pi
angles = np.random.uniform(0, 2 * np.pi, size=size)


Ra_random = Ra0 + (radii_angular * np.cos(angles)) / \
    np.cos(np.deg2rad(Deg0))
Dec_random = Deg0 + radii_angular * \
    np.sin(angles)


c1 = SkyCoord(ra=Ra_random, dec=Dec_random, frame='icrs', unit="deg")
c2 = SkyCoord(ra=Ra1, dec=Dec1, frame='icrs', unit="deg")

# angular separation in arcseconds
sep = c1.separation(c2)


distance_ang = sep.arcsecond/arcsec_per_kpc.value  # kps


# plt.figure(figsize=(6, 6))
# plt.scatter(Ra1, Dec1, marker='*', zorder=5, label='sat')
# plt.scatter(Ra0, Deg0, marker='^', zorder=5, label='BCG')
# plt.scatter(Ra_random, Dec_random, s=0.1, alpha=0.6)
# plt.title('Random Rayleigh Distribution around BCG')
# plt.xlabel('RA (degrees)')
# plt.ylabel('Dec (degrees)')
# # plt.grid(True)
# plt.legend()
# # plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

c1 = SkyCoord(ra=Ra0, dec=Deg0, frame='icrs', unit="deg")
c2 = SkyCoord(ra=Ra1, dec=Dec1, frame='icrs', unit="deg")
sep = c1.separation(c2)
# distance2 = sep.arcsecond/arcsec_per_kpc.value
distance2 = 1080

radii = np.random.rayleigh(scale, size=size)  # kps

# plt.hist(radii, bins=100)
# plt.vlines(distance2, 0, 700, color='r', label='Rykoff dist')
# plt.show()
distance = np.sqrt(distance2**2-radii**2-2*distance2*radii*np.cos(angles))

# Plot a histogram of the distances in kpc
plt.figure(figsize=(8, 6))
plt.hist(distance, bins=100, alpha=0.7, color='b')
plt.hist(distance_ang, bins=100, alpha=0.7, color='b')
plt.vlines(distance2, 0, 700, color='r', label='Rykoff dist')
plt.title(f'Histogram of Distances to satellite')
plt.xlabel('Distance (kpc)')
plt.ylabel('N')
plt.grid(True)
plt.legend()
plt.show()
