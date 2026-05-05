import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from natsort import natsorted, ns
import glob

np.random.seed(42)
mpl.rcParams['figure.dpi'] = 300

profile_folder = 'new-test/'
# profile_folder = 'C:\\Users\\romix\\Documents\\GitHub\DeltaSigma2023\\forJack\\'

profiles = [
    '0103(H70)_ext(R0).txt',
    '0103(H70)_ext(R1).txt',
    '0306(H70)_ext(R0).txt',
    '0306(H70)_ext(R1).txt',
    '0609(H70)_ext(R0).txt',
    '0609(H70)_ext(R1).txt',
    # '0609_(200m)_ext(R0).txt',
    # '0609_(200m)_ext(R1).txt',
    # '0306_(200m)_ext(R0).txt',
    # '0306_(200m)_ext(R1).txt',
    # '0103_(200m)_ext(R0).txt',
    # '0103_(200m)_ext(R1).txt'
    # '0103(H70)_(R0)colossus.txt',
    # '0306(H70)_(R0)colossus.txt',
    # '0609(H70)_(R0)colossus.txt',
    # '0103C_Xu.txt',
    # '0306C_Xu.txt',
    # '0609C_Xu.txt',
    # '0103(H70)_ext(R1).txt',
    # '0306(H70)_ext(R1).txt',
    # '0103_142.9_rayleigh.txt',
    # '0306_142.9_rayleigh.txt',
    # '0609_142.9_rayleigh.txt',
    # '0103_142.9_rayleigh-skew.txt',
    # '0306_142.9_rayleigh-skew.txt',
    # '0609_257.22_rayleigh-skew.txt',
]

colors = ['#1E3E48', '#69995D', '#7E1946', '#1E3E48', '#69995D', '#7E1946']
# colors = ['red', 'k']

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#FBFEF9')   # Set figure background color
ax.set_facecolor('#FBFEF9')          # Set axis background color

# Plotting
for i, file in enumerate(profiles):
    halo = np.genfromtxt(profile_folder + file)
    R = halo[0:, 0] / 1000
    DS = halo[0:, 1] / 1000000

    if 'R0' in file:
        ax.plot(R, DS, linestyle='-', c=colors[i], alpha=1)
    else:
        ax.plot(R, DS, c=colors[i], ls='--', alpha=0.6)

# Labels and text
# ax.text(2.5, 40, 'R=R(BCG)', fontsize=16,
#         color='black', ha='right', va='center')
# ax.text(2.5, -10, r'R=P(R,$\sigma$)', fontsize=16,
#         color='blue', ha='right', va='center')

# Load and plot data with error bars
data_path = 'C:/scp'
df = pd.read_csv(
    data_path + '/roman_esd_70ShapePipe_redmapper_clusterDist0.1_randomsTrue_1.csv')
ds = df['ds'].values
rp = df['rp'].values
ds_err = df['ds_err']

ax.set_ylabel(
    r'$\Delta \Sigma (R) \, [M_\odot / \mathrm{pc}^2]$', fontsize=12, labelpad=0)
ax.set_xlabel(r'$R [\mathrm{Mpc}]$', fontsize=12)

# Error bars (if needed)
# ax.errorbar(rp, ds, ds_err, fmt='o', label='dsigma Data',
#              markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)

# Axis limits
# ax.set_xlim(0.04, 2.5)
ax.set_ylim(-40, 110)

# Show the plot
plt.show()
