# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:10:30 2025

@author: romix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


mpl.rcParams['figure.dpi'] = 300
lenses = Table.read("members_n_clusters_masked.fits")
data_mask = (
    (lenses["R"] >= 0.1)
    & (lenses["R"] < 0.9)
    & (lenses["PMem"] > 0.8)
    # & (lenses["zspec"] > -1)

)
lenses = lenses[data_mask]
# lenses.write("C:\\catalogs\\filtered_members.fits", overwrite=True)

total_diff = lenses['zspec'] - lenses['Z_halo']
print(np.percentile(np.abs(total_diff.compressed()), 98))


d1 = cosmo.comoving_distance(lenses['zspec']).value      # in Mpc
d2 = cosmo.comoving_distance(lenses['Z_halo']).value   # in Mpc

cosmo_diff = np.abs(d1-d2)
print(np.percentile(cosmo_diff, 10), 'Mpc')
# plt.hist(cosmo_diff, bins=50)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()


lens1 = lenses[(lenses["R"] >= 0.1) & (lenses["R"] < 0.3)]
lens3 = lenses[(lenses["R"] >= 0.4) & (lenses["R"] < 0.6)]
lens6 = lenses[(lenses["R"] >= 0.6) & (lenses["R"] < 0.9)]

diff1 = lens1['zspec'] - lens1['Z_halo']
diff3 = lens3['zspec'] - lens3['Z_halo']
diff6 = lens6['zspec'] - lens6['Z_halo']

fig = plt.figure(figsize=(7, 6))

plt.hist(np.abs(diff1), histtype='step', bins=50,
         color='g', label='$0.1\\leq r_{p}<0.3~h^{-1}Mpc$')
plt.hist(np.abs(diff3), histtype='step', bins=50,
         color='b', label='$0.3\\leq r_{p}<0.6~h^{-1}Mpc$')
plt.hist(np.abs(diff6), histtype='step', bins=50,
         color='r', label='$0.6\\leq r_{p}<0.9~h^{-1}Mpc$')

plt.axvline(np.percentile(np.abs(diff1.compressed()), 98),
            color='g', linestyle='--')
plt.axvline(np.percentile(np.abs(diff3.compressed()), 98),
            color='b', linestyle='--')
plt.axvline(np.percentile(np.abs(diff6.compressed()), 98),
            color='r', linestyle='--')

print(np.percentile(np.abs(diff1.compressed()), 98))
print(np.percentile(np.abs(diff3.compressed()), 98))
print(np.percentile(np.abs(diff6.compressed()), 98))

plt.yscale('log')
# plt.xscale('log')
plt.xlim(0, 0.3)
plt.xlabel('z$_{spec}$ - z$_{ph}$', fontsize=18)
plt.ylabel('N sat', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
# plt.savefig('zdiffmem_hist.pdf', dpi=300)
plt.show()

# %%

# lenses = Table.read("members_n_clusters_masked.fits")
# data_mask = (
#     (lenses["R"] >= 0.1)
#     & (lenses["R"] < 0.9)
#     & (lenses["PMem"] > 0.8)
#     & (lenses["zspec"] > -1)

# )
# lenses = lenses[data_mask]
# total_diff = lenses['zspec'] - lenses['Z_halo']

# plt.scatter(lenses['PMem'], np.abs(total_diff), s=0.1)
# plt.yscale('log')
# plt.show()
