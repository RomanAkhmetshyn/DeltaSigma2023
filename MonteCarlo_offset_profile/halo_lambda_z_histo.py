import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


bins = ['0103', '0306', '0609']
colors = ['red', 'blue', 'green']
labels = [r'$0.1/leq r_p < 0.3~h^{-1}Mpc$',
          r'$0.3/leq r_p < 0.6~h^{-1}Mpc$', r'$0.6/leq r_p < 0.9~h^{-1}Mpc$']

# Create 1 row x 4 columns of subplots
fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# ---------- Plot 1: Stellar Mass Histogram ----------
for i, bin in enumerate(bins):
    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    lenses = Table.read("C:/catalogs/redmapper_members_n_clusters_MASS.fits")
    nomass_lens = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    data_mask = (
        (lenses["R"] >= lowlim) &
        (lenses["R"] < highlim) &
        (lenses["PMem"] > 0.8)
    )
    another_data_mask = (
        (nomass_lens["R"] >= lowlim) &
        (nomass_lens["R"] < highlim) &
        (nomass_lens["PMem"] > 0.8)
    )

    lenses = lenses[data_mask]
    lenses['MASS_BEST'] = np.power(10, lenses['MASS_BEST'])

    axs[0].hist(np.log10(lenses['MASS_BEST']), bins=50, histtype='step',
                color=colors[i], linewidth=2)
    # axs[0].axvline(np.mean(np.log10(lenses['MASS_BEST'])),
    #                color=colors[i], linestyle='--', linewidth=2)

axs[0].set_xlim(8.5, 12.2)
axs[0].set_ylim(1, 100000)
axs[0].set_yscale('log')
axs[0].set_xlabel(r'$/log(M_/ast [M_/odot])$', fontsize=22)
axs[0].set_ylabel('Count', fontsize=22)
# axs[0].legend()
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].tick_params(axis='both', which='minor', labelsize=18)

# ---------- Plot 2: Redshift Histogram + Cluster zlambda ----------
for i, bin in enumerate(bins):
    z_lenses = Table.read("C:/catalogs/redmapper_mnc_allz.fits")
    clusters = Table.read("C:/catalogs/clusters_w_centers.fit")
    lenses_for_clusters = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (z_lenses["R"] >= lowlim) &
        (z_lenses["R"] < highlim) &
        (z_lenses["PMem"] > 0.8)
    )

    z_lenses = z_lenses[data_mask]
    redshifts = z_lenses['z_any']

    axs[1].hist(redshifts, bins=50, histtype='step',
                linewidth=2, color=colors[i])

    # Add zlambda histogram (dashed style)
    cluster_mask = (
        (lenses_for_clusters["R"] >= lowlim) &
        (lenses_for_clusters["R"] < highlim) &
        (lenses_for_clusters["PMem"] > 0.8)
    )
    lenses_for_clusters = lenses_for_clusters[cluster_mask]
    IDs = np.unique(lenses_for_clusters["ID"])
    matched = clusters[np.isin(clusters["ID"], IDs)]
    zlambda_values = matched["zlambda"]

    axs[1].hist(zlambda_values, bins=50, histtype='step',
                linewidth=2, color=colors[i], linestyle='--', alpha=0.7)

axs[1].set_xlim(0, 0.8)
axs[1].set_ylim(1, 100000)
axs[1].set_yscale('log')
axs[1].set_xlabel(r'$z_{spec}$' + ' or ' + r'$z_{photo}$', fontsize=22)
# axs[1].legend(fontsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[1].tick_params(axis='both', which='minor', labelsize=18)

# ---------- Plot 3: Projected Distance Histogram ----------
for i, bin in enumerate(bins):
    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    lenses = Table.read(f'C:/catalogs/{bin}_members_dists.fits')
    distances = lenses['R'] * 1.429
    print(f'{bin}: {len(distances)} entries')

    axs[2].hist(distances, bins=50, histtype='step',
                linewidth=2, color=colors[i], label=labels[i])

    distances = lenses['R0'] * 1.429

    axs[2].hist(distances, bins=50, histtype='step',
                linewidth=2, color=colors[i], alpha=0.5)

axs[2].set_xlim(0.1, 1.35)
axs[2].set_ylim(1, 10000)
axs[2].set_yscale('log')
axs[2].set_xlabel(r'$r_p$ [Mpc]', fontsize=22)
axs[2].legend(fontsize=16)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].tick_params(axis='both', which='minor', labelsize=18)

# ---------- Plot 4: Cluster Lambda Histogram ----------
for i, bin in enumerate(bins):
    clusters = Table.read("C:/catalogs/clusters_w_centers.fit")
    lenses_for_clusters = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (lenses_for_clusters["R"] >= lowlim) &
        (lenses_for_clusters["R"] < highlim) &
        (lenses_for_clusters["PMem"] > 0.8)
    )
    lenses_for_clusters = lenses_for_clusters[data_mask]
    IDs = np.unique(lenses_for_clusters["ID"])
    print('unique clusters: ', len(IDs))
    matched = clusters[np.isin(clusters["ID"], IDs)]
    lambda_values = matched["lambda"]

    print('mean richness:', np.mean(lambda_values))
    print('mean pmem: ', np.mean(lenses_for_clusters["PMem"]))

    axs[3].hist(lambda_values, bins=50, histtype='step',
                linewidth=2, color=colors[i])

axs[3].set_yscale('log')
axs[3].set_xlabel(r'$/lambda$ (richness)', fontsize=22)
axs[3].tick_params(axis='both', which='major', labelsize=18)
axs[3].tick_params(axis='both', which='minor', labelsize=18)

# ---------- Styling All Subplots ----------
for ax in axs:
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', which='minor', length=5, width=1)
    ax.tick_params(axis='both', which='major', length=10, width=2)

plt.tight_layout()
# plt.savefig('combined_histograms.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
for i, bin in enumerate(bins):
    clusters = Table.read("C:/catalogs/clusters_w_centers.fit")
    lenses_for_clusters = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (lenses_for_clusters["R"] >= lowlim) &
        (lenses_for_clusters["R"] < highlim) &
        (lenses_for_clusters["PMem"] > 0.8)
    )
    lenses_for_clusters = lenses_for_clusters[data_mask]
    # q16, q50, q84 = np.quantile(lenses_for_clusters['PMem'], [0.16, 0.50, 0.84])
    # print(f"PMem (median) = {q50:.4f}")
    # print(f"PMem (-1σ)     = {q50 - q16:.4f}")
    # print(f"PMem (+1σ)     = {q84 - q50:.4f}")
    print(np.mean(lenses_for_clusters["PMem"]))
    plt.hist(lenses_for_clusters['PMem'], bins=50, histtype='step',
             linewidth=2, color=colors[i])
plt.show()

# %%

richness_cat = Table.read(
    "C:/Users/romix/Documents/GitHub/DeltaSigma2023/data/remapper_cat+mem/richness_err.fit")
for i, bin in enumerate(bins):
    lenses_for_clusters = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (lenses_for_clusters["R"] >= lowlim) &
        (lenses_for_clusters["R"] < highlim) &
        (lenses_for_clusters["PMem"] > 0.8)
    )
    lenses_for_clusters = lenses_for_clusters[data_mask]
    IDs = np.unique(lenses_for_clusters["ID"])
    matched = richness_cat[np.isin(clusters["ID"], IDs)]
    lambda_values = matched["lambda"]
    lambda_errs = matched["e_lambda"]

    print('mean richness:', np.mean(lambda_errs))

    plt.hist(lambda_errs, bins=50, histtype='step',
             linewidth=2, color=colors[i])

plt.yscale('log')
plt.xlabel('$\Delta\lambda$')
plt.show()

for i, bin in enumerate(bins):
    lenses_for_clusters = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (lenses_for_clusters["R"] >= lowlim) &
        (lenses_for_clusters["R"] < highlim) &
        (lenses_for_clusters["PMem"] > 0.8)
    )
    lenses_for_clusters = lenses_for_clusters[data_mask]
    IDs = np.unique(lenses_for_clusters["ID"])
    matched = richness_cat[np.isin(clusters["ID"], IDs)]
    lambda_values = matched["lambda"]
    lambda_errs = matched["e_lambda"]

    mass = np.log10(np.exp(1.72) * (lambda_values / 60)**(1.08) * 10**(14))
    mass_top = np.log10(
        np.exp(1.72) * ((lambda_values + lambda_errs) / 60)**(1.08) * 10**(14))
    mass_bottom = np.log10(
        np.exp(1.72) * ((lambda_values - lambda_errs) / 60)**(1.08) * 10**(14))

    # print('mean richness:', np.mean(lambda_errs))
    print(bin)
    print(len(IDs))
    plt.hist(mass, bins=50, histtype='step',
             linewidth=2, color='k')
    plt.hist(mass_top, bins=50, histtype='step',
             linewidth=2, color=colors[i])
    print('<log(M+dM)> = ', np.mean(mass_top))
    plt.hist(mass_bottom, bins=50, histtype='step',
             linewidth=2, color=colors[i], alpha=0.5)
    print('real <log(M)> = ', np.mean(mass))
    print('<log(M-dM)> = ', np.mean(mass_bottom))
    print('------------------')

plt.yscale('log')
plt.xlabel('logM$_{sun}$')
plt.show()
