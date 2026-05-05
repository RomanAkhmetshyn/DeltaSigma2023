# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:40:35 2026

@author: romix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table, vstack, Column

mpl.rcParams['figure.dpi'] = 300


# lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits")

# close_mask = (
#     (lenses["R"] >= 0.1) &
#     (lenses["R"] < 0.3) &
#     (lenses["PMem"] > 0.8)
# )

# mid_mask = (
#     (lenses["R"] >= 0.3) &
#     (lenses["R"] < 0.6) &
#     (lenses["PMem"] > 0.8)
# )

# far_mask = (
#     (lenses["R"] >= 0.6) &
#     (lenses["R"] < 0.9) &
#     (lenses["PMem"] > 0.8)
# )

# close_lens = lenses[close_mask]
# mid_lens = lenses[mid_mask]
# far_lense = lenses[far_mask]

# ids_close = set(close_lens["ID"])
# ids_mid = set(mid_lens["ID"])
# ids_far = set(far_lense["ID"])

# common_ids = ids_close & ids_mid & ids_far
# common_ids = sorted(common_ids)

# mask_common = [id_ in common_ids for id_ in close_lens["ID"]]
# close_lenses_common = close_lens[mask_common]

# mask_common = [id_ in common_ids for id_ in mid_lens["ID"]]
# mid_lenses_common = mid_lens[mask_common]

# mask_common = [id_ in common_ids for id_ in far_lense["ID"]]
# far_lenses_common = far_lense[mask_common]

# from astropy.table import vstack

# combined = vstack([close_lenses_common, mid_lenses_common, far_lenses_common])
# combined.write("C:/catalogs/lenses_w_common_clusters.fits", overwrite=True)

# %%

# lenses = Table.read("C:/catalogs/lenses_w_common_clusters.fits")
colors = ['red', 'blue', 'green']
bins = ['0103', '0306', '0609']
for i, bin in enumerate(bins):
    clusters = Table.read("C:/catalogs/clusters_w_centers.fit")
    lenses_for_clusters = Table.read("C:/catalogs/lenses_w_common_clusters.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (lenses_for_clusters["R"] >= lowlim)
        & (lenses_for_clusters["R"] < highlim)
        & (lenses_for_clusters["PMem"] > 0.8)
    )
    lenses_for_clusters = lenses_for_clusters[data_mask]
    IDs = np.unique(lenses_for_clusters["ID"])
    print('unique clusters: ', len(IDs))
    matched = clusters[np.isin(clusters["ID"], IDs)]
    lambda_values = matched["lambda"]

    print('mean richness:', np.mean(lambda_values))
    print('mean pmem: ', np.mean(lenses_for_clusters["PMem"]))

    plt.hist(lambda_values, bins=50, histtype='step',
             linewidth=2, color=colors[i])

plt.yscale('log')
plt.xlabel(r'$\lambda$ (richness)', fontsize=22)

plt.tight_layout()
# plt.savefig('combined_histograms.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%

colors = ['red', 'blue', 'green']
bins = ['0103', '0306', '0609']
for i, bin in enumerate(bins):
    clusters = Table.read("C:/catalogs/clusters_w_centers.fit")
    lenses_for_clusters = Table.read("C:/catalogs/lenses_w_common_clusters.fits")
    # lenses_for_clusters = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (lenses_for_clusters["R"] >= lowlim)
        & (lenses_for_clusters["R"] < highlim)
        & (lenses_for_clusters["PMem"] > 0.8)
    )
    lenses_for_clusters = lenses_for_clusters[data_mask]
    IDs = lenses_for_clusters["ID"]
    idx = np.searchsorted(clusters["ID"], IDs)
    cluster_lambda = clusters["lambda"]
    lambda_values = cluster_lambda[idx]

    # print('mean richness:', np.mean(lambda_values))
    # print('mean pmem: ', np.mean(lenses_for_clusters["PMem"]))

    plt.hist(lambda_values, bins=50, histtype='step',
             linewidth=2, color=colors[i], density=True)

plt.yscale('log')
plt.xlabel(r'$\lambda$ (richness) associated with lenses', fontsize=14)
plt.title('common only')
# plt.title('original')
# plt.ylim(10, 1e5)
plt.tight_layout()
# plt.savefig('combined_histograms.pdf', dpi=300, bbox_inches='tight')
plt.show()
