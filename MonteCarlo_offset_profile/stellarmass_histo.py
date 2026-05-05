import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

bins = ['0103', '0306', '0609']
colors = ['red', 'blue', 'green']
labels = [r'$0.1\leq r_p < 0.3~h^{-1}Mpc$',
          r'$0.3\leq r_p < 0.6~h^{-1}Mpc$', r'$0.6\leq r_p < 0.9~h^{-1}Mpc$']

# Create 1 row x 3 columns of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

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
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        & (lenses["PMem"] > 0.8)
    )
    another_data_mask = (
        (nomass_lens["R"] >= lowlim)
        & (nomass_lens["R"] < highlim)
        & (nomass_lens["PMem"] > 0.8)
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
axs[0].set_xlabel(r'$\log(M_\ast [M_\odot])$', fontsize=22)
axs[0].set_ylabel('Count', fontsize=22)
# axs[0].legend()
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].tick_params(axis='both', which='minor', labelsize=18)

# ---------- Plot 2: Redshift Histogram ----------
for i, bin in enumerate(bins):
    z_lenses = Table.read("C:/catalogs/redmapper_mnc_allz.fits")

    if bin == '0609':
        lowlim, highlim = 0.6, 0.9
    elif bin == '0306':
        lowlim, highlim = 0.3, 0.6
    else:
        lowlim, highlim = 0.1, 0.3

    data_mask = (
        (z_lenses["R"] >= lowlim)
        & (z_lenses["R"] < highlim)
        & (z_lenses["PMem"] > 0.8)
    )

    z_lenses = z_lenses[data_mask]
    redshifts = z_lenses['z_any']

    axs[1].hist(redshifts, bins=50, histtype='step',
                linewidth=2, color=colors[i], label=labels[i])
    # axs[1].axvline(np.mean(redshifts),
    #                color=colors[i], linestyle='--', linewidth=2)

axs[1].set_xlim(0, 0.8)
axs[1].set_ylim(1, 100000)
axs[1].set_yscale('log')
axs[1].set_xlabel(r'$z_{spec}$' + ' or ' + r'$z_{photo}$', fontsize=22)
axs[1].legend(fontsize=16)
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
                linewidth=2, color=colors[i])

    distances = lenses['R0'] * 1.429

    axs[2].hist(distances, bins=50, histtype='step',
                linewidth=2, color=colors[i], alpha=0.5)

axs[2].set_xlim(0.1, 1.35)
axs[2].set_ylim(1, 10000)
axs[2].set_yscale('log')
axs[2].set_xlabel(r'$r_p$ [Mpc]', fontsize=22)
axs[2].legend()
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].tick_params(axis='both', which='minor', labelsize=18)

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
