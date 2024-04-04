
from astropy.coordinates import SkyCoord
import astropy.cosmology
import numpy as np
from astropy.table import Table, vstack, Column
from tqdm import trange
import matplotlib.pyplot as plt

H0=100
cosmo = astropy.cosmology.FlatLambdaCDM(H0=H0, Om0=0.3, Ob0=0.049)
clusters= Table.read("C:/catalogs/clusters_w_centers.fit")


centers=[['RA0deg', 'DE0deg', 'PCen0'],
         ['RA1deg', 'DE1deg', 'PCen1']]


distances=np.zeros(len(clusters))
for row in trange(len(clusters)):
    
    z=clusters[row]['zlambda']
    
    # row_values=[cluster_ID, sats[row]['R']]
    
    # selected_rows = clusters[clusters['ID'] == cluster_ID]
    
        
    
    bcg1_RA = clusters[row][centers[0][0]]
    bcg1_Dec = clusters[row][centers[0][1]]
    bcg2_RA = clusters[row][centers[1][0]]
    bcg2_Dec = clusters[row][centers[1][1]]

    

    
    c1 = SkyCoord(bcg1_RA, bcg1_Dec, frame='icrs', unit="deg") 
    c2 = SkyCoord(bcg2_RA, bcg2_Dec, frame='icrs', unit="deg") 
    sep = c1.separation(c2)

    
    arcsec_per_kpc=cosmo.arcsec_per_kpc_proper(z)
    distances[row]=(sep.arcsecond/arcsec_per_kpc.value)
    
#%%
plt.hist(distances, bins=80)
avg=np.mean(distances)
med=np.median(distances)
plt.xlabel('distance between bcg1 and bcg2 [kpc]')
plt.ylabel('counts')
plt.title(f'mean distance: {avg:.2f}, median: {med:.2f} kpc')
plt.show()
        
    
    


