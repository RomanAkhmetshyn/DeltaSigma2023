# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:00:02 2023

@author: Admin
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

catalogs = ["Lensfit", "ShapePipe"]
# The min and max radius of each cluster-centric distance bin
cluster_dist_bins = [0.1, 0.3, 0.6]
cluster_dist_bins_end = {0.1: 0.3, 0.3: 0.6, 0.6: 0.9}

csv_files = []
for cluster_dist_bin in cluster_dist_bins:
    for catalog in catalogs:
        glob_pattern = (
            "D:/GitHub/summer-research/data/dsigma_measurements/output/example_esd_"
            + f"{catalog}_clusterDist{cluster_dist_bin}_randomsTrue_*.csv"
        )
        csv_files.extend(glob.glob(glob_pattern))
# print(csv_files)

markers = {"Lensfit": "o", "ShapePipe": "s"}
colors = {"Lensfit": "C0", "ShapePipe": "C1"}

csv_idx = 0
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 7.5))
for ax, cluster_dist_bin in zip(axs, cluster_dist_bins):
    for catalog in catalogs:
        data = pd.read_csv(csv_files[csv_idx], usecols=["rp", "ds", "ds_err"])
        ax.errorbar(
            x=data["rp"],
            y=data["ds"],
            yerr=data["ds_err"],
            marker=markers[catalog],
            color=colors[catalog],
            ls="none",
            capsize=2,
            label=catalog,
            alpha=0.5,
        )
        csv_idx += 1
    if ax == axs[0]:
        ax.legend()  # only add legend to top plot
    # Manually set x- and y-limits
    ax.set_xlim(0.05, 1.5)
    if cluster_dist_bin == 0.1:
        ax.set_ylim(-40, 160)
        ax.set_yticks(np.arange(-40, 161, 40))
    else:
        ax.set_ylim(-40, 100)
        ax.set_yticks(np.arange(-40, 101, 20))
    # Add cluster-centric distance bin label
    ax.text(
        0.03,
        0.85,
        f"$r_p \in [{cluster_dist_bin}, {cluster_dist_bins_end[cluster_dist_bin]})$",
        fontsize=13,
        transform=ax.transAxes,
    )
    ax.grid(False)
fig.supxlabel(r"$R$ [h$^{-1}$ Mpc]")
fig.supylabel(r"$\Delta\Sigma$ [$\rm h^{-1}\, M_\odot\, pc^{-2}$]")
fig.tight_layout()
# fig.savefig("./output/example_esd.jpg", dpi=300)
plt.show()