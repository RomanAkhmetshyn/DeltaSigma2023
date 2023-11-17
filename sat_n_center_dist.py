
from astropy.coordinates import SkyCoord
import astropy.cosmology
import numpy as np
from astropy.table import Table, vstack, Column
from tqdm import trange
import matplotlib.pyplot as plt

# c1 = SkyCoord('5h23m34.5s', '-69d45m22s', distance=70*u.kpc, frame='icrs')
# c2 = SkyCoord('0h52m44.8s', '-72d49m43s', distance=80*u.kpc, frame='icrs')
# sep = c1.separation_3d(c2)
# print(sep)

H0=100
cosmo = astropy.cosmology.FlatLambdaCDM(H0=H0, Om0=0.3, Ob0=0.049)
clusters= Table.read("D:/GitHub/summer-research/data/clusters_w_centers.fit")
sats = Table.read("D:/GitHub/summer-research/data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits")



bins=['0609']



for bin in bins:
    
    result_table = Table()

    # Add columns to the table
    result_table.add_column(Column(name='ID', dtype=int))
    result_table.add_column(Column(name='R', dtype=float))
    result_table.add_column(Column(name='R0', dtype=float))
    result_table.add_column(Column(name='W0', dtype=float))
    result_table.add_column(Column(name='R1', dtype=float))
    result_table.add_column(Column(name='W1', dtype=float))
    result_table.add_column(Column(name='R2', dtype=float))
    result_table.add_column(Column(name='W2', dtype=float))
    result_table.add_column(Column(name='R3', dtype=float))
    result_table.add_column(Column(name='W3', dtype=float))
    result_table.add_column(Column(name='R4', dtype=float))
    result_table.add_column(Column(name='W4', dtype=float))
    
    if bin=='0609':
        lowlim=0.6
        highlim=0.9
    elif bin=='0306':
        lowlim=0.3
        highlim=0.6
    elif bin=='0103':
        lowlim=0.1
        highlim=0.3 
    
    
    data_mask = (
            (sats["R"] >= lowlim)
            & (sats["R"] < highlim)
            # & (lenses["PMem"] > 0.8)
            # & (lenses["zspec"] > -1)
            & (sats["PMem"] > 0.8)
        )
    
    sats=sats[data_mask]
    
    # relative_diff=np.zeros(len(sats),)
    
    centers=[['RA0deg', 'DE0deg', 'PCen0'],
             ['RA1deg', 'DE1deg', 'PCen1'],
             ['RA2deg', 'DE2deg', 'PCen2'],
             ['RA3deg', 'DE3deg', 'PCen3'],
             ['RA4deg', 'DE4deg', 'PCen4']]
    
    for row in trange(len(sats)):
        cluster_ID=sats[row]['ID']
        sat_RA=sats[row]['RAJ2000']
        sat_Dec=sats[row]['DEJ2000']
        
        z=sats[row]['Z_halo']
        
        # if sats[row]['zspec']>0:
        #     z_sat=sats[row]['zspec']
        # else:
        #     z_sat=sats[row]['Z_halo']
            
        # z_centr=sats[row]['Z_halo']
        
        row_values=[cluster_ID, sats[row]['R']]
        
        selected_rows = clusters[clusters['ID'] == cluster_ID]
        
        for coords in centers:
            
        
            center_RA = selected_rows[coords[0]]
            center_Dec = selected_rows[coords[1]]
            # center_RA = selected_rows['RAJ2000']
            # center_Dec = selected_rows['DEJ2000']
            distance_weight=selected_rows[coords[2]]
            
        
            
            c1 = SkyCoord(sat_RA, sat_Dec, frame='icrs', unit="deg") 
            c2 = SkyCoord(center_RA, center_Dec, frame='icrs', unit="deg") 
            sep = c1.separation(c2)
            # c1 = SkyCoord(sat_RA, sat_Dec, distance=((z_sat*(c.to('km/s').value)/H0))*u.Mpc, frame='icrs',  unit=["deg","deg"]) #sat
            # c2 = SkyCoord(center_RA, center_Dec, distance=((z_centr*(c.to('km/s').value)/H0))*u.Mpc, frame='icrs', unit=["deg","deg" ])  #cluster central galaxy
            # sep = c1.separation_3d(c2)
            
            arcsec_per_kpc=cosmo.arcsec_per_kpc_proper(z)
            distance=(sep.arcsecond/arcsec_per_kpc.value)/1000
            
            row_values.extend([distance, distance_weight])
    
            # diff=abs((distance-sats[row]['R'])/sats[row]['R'])
            # print(diff)
            
            # distance=(sep.value)
            # diff=abs((distance-sats[row]['R'])/sats[row]['R'])
            
            # relative_diff[row]=diff[0]*100
            
        result_table.add_row(row_values)
        
    result_table.write(f'D:/GitHub/summer-research/data/{bin}_members_dists.fits', format='fits', overwrite=True)
#%%

# print(np.mean(relative_diff))
# print(np.amax(relative_diff))
# print(np.min(relative_diff))
# print(np.median(relative_diff))

# plt.hist(relative_diff, bins='auto', color='blue', alpha=0.7)
# plt.xlabel('relative difference (%)')
# plt.ylabel('count')
# plt.title(f'R bin = {bin}')
# plt.show()
