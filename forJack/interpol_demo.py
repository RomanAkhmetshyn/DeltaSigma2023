# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:34:16 2024

@author: romix
"""

import glob
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgba
mpl.rcParams['figure.dpi'] = 300

file_pattern = "0609_*_rayleigh.txt"
files = sorted(glob.glob(file_pattern))  # Get all matching files

scales = []  # To store scale values
models = []  # To store the second column of data

# Extract scale and second column data
for file in files:
    # Extract scale value from the filename (split by '_' and '.txt')
    scale = float(file.split('_')[1])
    scales.append(scale)

    # Load the second column from the file
    data = np.loadtxt(file, usecols=1)  # Only load the second column (index 1)
    models.append(data)

scales = np.array(scales)
models = np.array(models)  # Shape: (num_scales, num_points)


target_scale = 0.045

plt.figure(figsize=(10, 6))
for i in range(len(files)):
    brightness = (i + 1) / len(files)  # Calculate brightness (0 to 1 scale)
    color = to_rgba('grey', alpha=brightness)  # Convert grey to a lighter shade
    plt.plot(models[i], color=color, label=f'Scale {scales[i]:.6f}')

# Step 3: Customize and show the plot

plt.legend(loc='upper left', bbox_to_anchor=(
    1, 1), fontsize='small')  # Optional
plt.tight_layout()


interpolated_model = np.zeros_like(models[0])
for i in range(models.shape[1]):
    interp_func = interp1d(
        scales, models[:, i], kind='cubic', fill_value="extrapolate")
    interpolated_model[i] = interp_func(target_scale)

plt.plot(interpolated_model, color='r')

plt.show()
