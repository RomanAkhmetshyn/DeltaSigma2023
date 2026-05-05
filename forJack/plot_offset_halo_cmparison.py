import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from natsort import natsorted, ns
import glob
from scipy.interpolate import interp1d

np.random.seed(42)
mpl.rcParams['figure.dpi'] = 300

profile_folder = 'C:/Users/romix/Documents/GitHub/DeltaSigma2023/MonteCarlo_offset_profile/new-test/'
# profile_folder = 'C:\\Users\\romix\\Documents\\GitHub\DeltaSigma2023\\forJack\\'

profiles = [
    '0103(H70)_ext(R0).txt',
    # '0103(H70)_ext(R1).txt',
    '0306(H70)_ext(R0).txt',
    # '0306(H70)_ext(R1).txt',
    '0609(H70)_ext(R0).txt',
    # '0609(H70)_ext(R1).txt',
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
# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))


# Plotting
for i, file in enumerate(profiles):
    halo = np.genfromtxt(profile_folder + file)
    R = halo[1:, 0] / 1000
    DS = halo[1:, 1] / 1000000
    label = 'R=R(BCG)' if i == 0 else '_nolegend_'
    ax.plot(R, DS, c='k', ls='--', alpha=0.6, label=label)

# Labels and text
# ax.text(2.0, 60, 'R=R(BCG)', fontsize=16,
#         color='black', ha='right', va='center')
# ax.text(2.0, 10, r'R=P(R,$\sigma$)', fontsize=16,
#         color='r', ha='right', va='center')

ax.text(1.1, 68, r'$0.1\leq r_p < 0.3~h^{-1}Mpc$', fontsize=18,
        color='blue', ha='center', va='center')

ax.text(1.4, 38, r'$0.3\leq r_p < 0.6~h^{-1}Mpc$', fontsize=18,
        color='blue', ha='center', va='center')

ax.text(1.8, 13, r'$0.6\leq r_p < 0.9~h^{-1}Mpc$', fontsize=18,
        color='blue', ha='center', va='center')


###
patterns = '0103', '0306', '0609'

for pattern in patterns:
    file_pattern = f"{pattern}_*_rayleigh.txt"
    # Get all matching files
    files = natsorted(glob.glob(file_pattern))

    scales = []  # To store scale values
    models = []  # To store the second column of data

    # Extract scale and second column data
    for file in files:
        # Extract scale value from the filename (split by '_' and '.txt')
        scale = float(file.split('_')[1])
        scales.append(scale)

        # Load the second column from the file
        # Only load the second column (index 1)
        data = np.loadtxt(file, usecols=1)
        models.append(data)

    scales = np.array(scales)
    models = np.array(models)  # Shape: (num_scales, num_points)

    target_scale = 143

    interpolated_model = np.zeros_like(models[0])
    for i in range(models.shape[1]):
        interp_func = interp1d(
            scales, models[:, i], kind='cubic', fill_value="extrapolate")
        interpolated_model[i] = interp_func(target_scale)

    r = np.loadtxt(file, usecols=0)
    label = 'R=P(R,$\sigma$)' if pattern == patterns[0] else '_nolegend_'
    plt.plot(r / 1000, interpolated_model / 1000000,
             color='r', label=label)


ax.set_ylabel(
    r'$\Delta \Sigma (R) \, [M_\odot / \mathrm{pc}^2]$', fontsize=18, labelpad=0)
ax.set_xlabel(r'$R [\mathrm{Mpc}]$', fontsize=18)
# Axis limits
ax.set_xlim(0.04, 2.5)
ax.set_ylim(-25, 80)
ax.tick_params(axis='both', labelsize=16)  # Change 14 to your desired font size

plt.legend(fontsize=18)
# Show the plot
plt.tight_layout()
# plt.savefig('halo_models.pdf', dpi=300)
plt.show()
