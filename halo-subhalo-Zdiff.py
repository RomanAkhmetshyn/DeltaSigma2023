# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:27:07 2023

@author: Admin
"""
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt


lenses = Table.read("./data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits")
data_mask = (
        (lenses["R"] >= 0.1)
        & (lenses["R"] < 0.9)
        & (lenses["PMem"] > 0.8)
        & (lenses["zspec"] > -1)
       
    )
lenses = lenses[data_mask]

differences = []
pmem_values = []

zs=[]

# Iterate through each row in the table
for row in lenses:
    zspec = row["zspec"]
    z_halo = row["Z_halo"]
    pmem = row["PMem"]
    
    zs.append(zspec)
    # Calculate the difference between zspec and Z_halo
    difference = zspec - z_halo
    
    # Save the difference and corresponding PMem value
    differences.append(difference)
    pmem_values.append(pmem)
    
plt.scatter(pmem_values, differences, s=1, alpha=0.5)

# Set the labels and title
plt.xlabel('PMem')
plt.ylabel('Difference (zspec - Z_halo)')
plt.title('Difference vs PMem ')

# Show the plot
plt.show()

pmem_array = np.array(pmem_values)
diff_array = np.array(differences)
zs=np.array(zs)
combined_array = np.column_stack((pmem_array, diff_array, np.array(zs)))



# Save the combined array as a text file
# np.savetxt('pmem_diff.csv', combined_array, delimiter=',', fmt='%.6f')

# num_groups = len(pmem_values) // 100

# # Create empty lists to store the standard deviations and corresponding PMem values
# std_deviations = []
# group_pmem_values = []

# # Iterate through each group of 100 data points
# for i in range(num_groups):
#     start_idx = i * 100
#     end_idx = start_idx + 100
    
#     group_differences = differences[start_idx:end_idx]
#     group_pmem = pmem_values[start_idx:end_idx]
    
#     # Calculate the standard deviation for the group
#     std_deviation = np.mean(group_differences)
    
#     # Save the standard deviation and corresponding PMem value for the group
#     std_deviations.append(std_deviation)
#     group_pmem_values.append(group_pmem[0])  # Save the PMem value for the group (assuming it's the same for all data points in the group)

# # Create a scatter plot of standard deviations vs PMem values
# plt.scatter(group_pmem_values, std_deviations, s=50, alpha=0.7)

# # Set the labels and title
# plt.xlabel('PMem')
# plt.ylabel('Standard Deviation of Differences')
# plt.title('Standard Deviation of Differences vs PMem (for every 100 data points)')

# # Show the plot
# plt.show()







