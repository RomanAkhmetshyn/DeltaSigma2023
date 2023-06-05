# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:25:38 2023

@author: Admin
"""

from multiprocessing import Pool

import numpy as np
from astropy.table import Table, vstack, Column
import math


sats = Table.read("D:/GitHub/summer-research/data/dr8_redmapper_v6.3.1_members_masked.fits")
clusters= Table.read("D:/GitHub/summer-research/data/dr8_run_redmapper_v6.3.1_lgt20_catalog.fit")

cluster_masses=[]
cluster_z=[]

IDs=[]

for match_id in sats['MEM_MATCH_ID']:
    print(match_id)
    IDs.append(match_id)
        
    for i, value in enumerate(clusters['MEM_MATCH_ID']):
        if value == match_id:
            row_number = i
            break  # Exit the loop once the value is found
    
    cluster_z.append(clusters['ZRED'][row_number])
    richness=clusters['Z_LAMBDA'][row_number]
    mass=math.exp(1.48)*(richness/60)**(1.06)*10**(14)
    cluster_masses.append(mass)
    
mass_column=Column(cluster_masses, name='M_halo')
redshift_column=Column(cluster_z, name='Z_halo')

sats.add_column(mass_column)
sats.add_column(redshift_column)
sats.write('D:/GitHub/summer-research/data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits', overwrite=True)  # Replace 'modified_table.fits' with the desired file name or path