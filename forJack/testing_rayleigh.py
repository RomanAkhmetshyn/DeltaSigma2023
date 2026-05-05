from scipy.stats import rayleigh
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from natsort import natsorted, ns
import glob

np.random.seed(42)
mpl.rcParams['figure.dpi'] = 300
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

scale = 0.3*1000  # sigma
# radii_angular = rayleigh.rvs(scale, size=size)
radii = np.random.rayleigh(scale, size=size)  # kps

# radii = sample_offset_radius(scale, size)

radii_angular = radii * arcsec_per_kpc.value  # arcsec
radii_angular = radii_angular/3600  # deg

fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axis
bg_color = '#FBFEF9'

# Set background color
fig.patch.set_facecolor(bg_color)   # Set figure background
ax.set_facecolor(bg_color)          # Set axis background

# Plot histogram
ax.hist(radii, bins=100, color='#1E3E48', alpha=0.7)
ax.vlines(scale, 0, 330, color='r', label=f'$\sigma=$ {scale} kpc', ls='--')
# Labels and title
ax.set_title('BCG offset distance distribution', fontsize=18)
ax.set_xlabel('R$_{off}$', fontsize=16)
ax.set_ylabel('N', fontsize=16)
ax.set_ylim(0, 320)
ax.set_xlim(0, 1180)
plt.legend(fontsize=16)
plt.show()

# Generate random angles uniformly between 0 and 2*pi
angles = np.random.uniform(0, 2 * np.pi, size=size)
# angles = np.random.uniform(-3 * np.pi / 4, 3 * np.pi / 4, size=size)
# angles = np.random.uniform(3 * np.pi / 4, 5*np.pi/4, size=size)
# angles = np.random.uniform(0, np.pi/2+np.pi/12, size=size)
# angles = np.random.uniform(-np.pi/12, np.pi/12, size=size)

# angles1 = np.random.uniform(-np.pi/12, np.pi/12, size=size)
# angles2 = np.random.uniform(np.pi-np.pi/12, np.pi+np.pi/12, size=size)
# angles3 = np.random.uniform(-np.pi / 2-np.pi/24, -np.pi / 2+np.pi/24, size=size)
# combined_angles = np.concatenate([angles1, angles2])
# angles = np.random.choice(combined_angles, size=size)


Ra_random = Ra0 + (radii_angular * np.cos(angles)) / \
    np.cos(np.deg2rad(Deg0))
Dec_random = Deg0 + radii_angular * \
    np.sin(angles)


c1 = SkyCoord(ra=Ra_random, dec=Dec_random, frame='icrs', unit="deg")
c2 = SkyCoord(ra=Ra1, dec=Dec1, frame='icrs', unit="deg")

# angular separation in arcseconds
sep = c1.separation(c2)


distance_ang = sep.arcsecond/arcsec_per_kpc.value  # kps


bg_color = '#FBFEF9'

# First Plot: Random Rayleigh Distribution
fig1, ax1 = plt.subplots(figsize=(6, 6))
fig1.patch.set_facecolor(bg_color)   # Set figure background
ax1.set_facecolor(bg_color)          # Set axes background

# Scatter plots
ax1.scatter(Ra1, Dec1, marker='*', zorder=5, label='satellite')
ax1.scatter(Ra0, Deg0, marker='^', zorder=5, label='RedMapper BCG')
ax1.scatter(Ra_random, Dec_random, s=0.1, alpha=0.6)

# Labels, title, legend
ax1.set_title('Random Rayleigh Distribution around BCG', fontsize=18)
ax1.set_xlabel('RA')
ax1.set_ylabel('Dec')
ax1.legend()
ax1.set_aspect('equal', adjustable='box')

plt.show()

# SkyCoord calculation
c1 = SkyCoord(ra=Ra0, dec=Deg0, frame='icrs', unit="deg")
c2 = SkyCoord(ra=Ra1, dec=Dec1, frame='icrs', unit="deg")
sep = c1.separation(c2)
distance2 = 1080

# Rayleigh distribution
radii = np.random.rayleigh(scale, size=size)
distance = np.sqrt(distance2**2 + radii**2 - 2 *
                   distance2 * radii * np.cos(angles))


# Second Plot: Distance Histogram
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.patch.set_facecolor(bg_color)   # Set figure background
ax2.set_facecolor(bg_color)          # Set axes background

# Histograms
ax2.hist(distance, bins=100, alpha=0.7, color='b',
         histtype='step')
ax2.hist(distance_ang, bins=100, alpha=0.7, color='r',
         histtype='step')

# Reference line
ax2.vlines(distance2, 0, 400, color='r', label='RedMapper BCG dist', ls='--')

# Labels, title, legend
ax2.set_title(f'Distances to satellite, $\\sigma=$ {scale} kpc')
ax2.set_xlabel('Distance (kpc)')
ax2.set_ylabel('N')
ax2.legend()
ax2.set_xlim(100, 2900)
ax2.set_ylim(0, 360)
plt.show()
