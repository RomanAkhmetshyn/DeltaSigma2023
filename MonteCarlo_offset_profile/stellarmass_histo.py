import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


bin='0609'

if bin=='0609':
    lowlim=0.6
    highlim=0.9
elif bin=='0306':
    lowlim=0.3
    highlim=0.6
elif bin=='0103':
    lowlim=0.1
    highlim=0.3 
    
lenses = Table.read("C:/catalogs/redmapper_members_n_clusters_MASS.fits")
nomass_lens=Table.read("C:/catalogs/members_n_clusters_masked.fits")
data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
       
    )

another_data_mask = (
        (nomass_lens["R"] >= lowlim)
        & (nomass_lens["R"] < highlim)
        & (nomass_lens["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
       
    )
lenses = lenses[data_mask]
original_lens=nomass_lens[another_data_mask]
# print(np.min(lenses['MASS_BEST']))
lenses['MASS_BEST']=np.power(10, lenses['MASS_BEST'])

#%%
min_mass=np.min(lenses['MASS_BEST'])
max_mass=np.amax(lenses['MASS_BEST'])

avg=np.log10(np.mean(lenses['MASS_BEST']))
percentiles=np.percentile(lenses['MASS_BEST'], [16, 50, 84])
q = np.diff(np.log10(percentiles))
print(q[0], avg, q[1])
med=np.log10(np.median(lenses['MASS_BEST']))
num=len(lenses['MASS_BEST'])
ratio=len(lenses)/len(original_lens)*100
plt.hist(np.log10(lenses['MASS_BEST']), bins=50)
plt.xlim(8, 13)
plt.ylim(0.1,50000)
plt.yscale('log')
plt.xlabel('log(mass)')
plt.ylabel('count')
plt.title(f'{bin} bin, n_sat: {num}, avg: {avg:.2f}, med: {med:.2f}, ratio: {ratio}%')
plt.show()