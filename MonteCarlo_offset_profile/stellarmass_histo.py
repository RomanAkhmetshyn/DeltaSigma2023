import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

bins = ['0103', '0306', '0609']
colors = ['red', 'blue', 'green']
labels = ['[0.1 - 0.3]', '[0.3 - 0.6]', '[0.6 - 0.9]']

plt.figure(figsize=(8, 6))

for i, bin in enumerate(bins):
    if bin == '0609':
        lowlim = 0.6
        highlim = 0.9
    elif bin == '0306':
        lowlim = 0.3
        highlim = 0.6
    elif bin == '0103':
        lowlim = 0.1
        highlim = 0.3
        
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
    original_lens = nomass_lens[another_data_mask]
    lenses['MASS_BEST'] = np.power(10, lenses['MASS_BEST'])
    
    plt.hist(np.log10(lenses['MASS_BEST']), bins=50, histtype='step', color=colors[i], label=labels[i], linewidth=2)
    plt.axvline(np.mean(np.log10(lenses['MASS_BEST'])), color=colors[i], linestyle='--', linewidth=2)

plt.xlim(8.5, 12.2)
plt.ylim(1, 100000)
plt.yscale('log')
plt.xlabel(r'$\log(M_\ast [M_\odot])$', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.legend()
plt.grid(True)

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.tick_params(axis='both', which='minor', length=5, width=1)
plt.tick_params(axis='both', which='major', length=10, width=2)


plt.tight_layout()
plt.savefig('Stellar_hist.png', bbox_inches='tight', dpi =300)
plt.show()

#%%


plt.figure(figsize=(8, 6))

for i, bin in enumerate(bins):
    z_lenses = Table.read("C:/catalogs/redmapper_mnc_allz.fits")

    if bin == '0609':
        lowlim = 0.6
        highlim = 0.9
    elif bin == '0306':
        lowlim = 0.3
        highlim = 0.6
    elif bin == '0103':
        lowlim = 0.1
        highlim = 0.3
    
    data_mask = (
        (z_lenses["R"] >= lowlim)
        & (z_lenses["R"] < highlim)
        & (z_lenses["PMem"] > 0.8)
    )
    
    z_lenses = z_lenses[data_mask]
    redshifts = z_lenses['z_any']
    
    # Plot histogram with step lines
    plt.hist(redshifts, bins=50, histtype='step', linewidth=2, color=colors[i], label=labels[i])
    plt.axvline(np.mean(redshifts), color=colors[i], linestyle='--', linewidth=2)

# Configure plot
plt.xlim(0, 0.8)
plt.ylim(1, 100000)
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.xlabel(r'$z_{spec}$'+' or '+r'$z_{photo}$', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.tick_params(axis='both', which='minor', length=5, width=1)
plt.tick_params(axis='both', which='major', length=10, width=2)

# Save and show plot
plt.savefig('redshift_histogram.png', bbox_inches='tight', dpi=300)
plt.show()

#%%

plt.figure(figsize=(8, 6))

for i, bin in enumerate(bins):
    lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits")

    if bin == '0609':
        lowlim = 0.6
        highlim = 0.9
    elif bin == '0306':
        lowlim = 0.3
        highlim = 0.6
    elif bin == '0103':
        lowlim = 0.1
        highlim = 0.3
    
    data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        & (lenses["PMem"] > 0.8)
    )
    
    lenses = Table.read(f'C:/catalogs/{bin}_members_dists.fits')
    distances = lenses['R0']*1.429
    print(len(distances))
    # Plot histogram with step lines
    plt.hist(distances, bins=50, histtype='step', linewidth=2, color=colors[i], label=labels[i])

# Configure plot
plt.xlim(0.1, 1.35)
plt.ylim(1, 100000)
plt.yscale('log')
plt.xlabel(r'R$_p$ [Mpc]', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.legend()
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.tick_params(axis='both', which='minor', length=5, width=1)
plt.tick_params(axis='both', which='major', length=10, width=2)

# Save and show plot
plt.savefig('distance_histogram.png', bbox_inches='tight', dpi=300)
plt.show()
    

