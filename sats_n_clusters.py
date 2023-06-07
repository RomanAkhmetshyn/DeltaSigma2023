# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:25:38 2023

@author: Admin
"""

from multiprocessing import Pool

import numpy as np
from astropy.table import Table, vstack, Column
import math


sats = Table.read("D:/GitHub/summer-research/data/remapper_cat+mem/dr8_members.fit")
clusters= Table.read("D:/GitHub/summer-research/data/remapper_cat+mem/dr8_cat.fit")

cluster_masses=[]
cluster_z=[]

cluster_dic={}

for match_id in sats['ID']:
    print(match_id)
    if match_id not in cluster_dic:
        
        
        for i, value in enumerate(clusters['ID']):
            if value == match_id:
                row_number = i
                break  # Exit the loop once the value is found
        
        z=clusters['zlambda'][row_number]
        cluster_z.append(z)
        richness=clusters['lambda'][row_number]
        mass=math.exp(1.48)*(richness/60)**(1.06)*10**(14)
        cluster_masses.append(mass)
        cluster_dic[match_id]=[mass,z]
        
    elif match_id in cluster_dic:
        cluster_z.append(cluster_dic[match_id][1])
        cluster_masses.append(cluster_dic[match_id][0])
    
print('creating columns')
mass_column=Column(cluster_masses, name='M_halo')
redshift_column=Column(cluster_z, name='Z_halo')
print('adding columns')
sats.add_column(mass_column)
sats.add_column(redshift_column)
sats.write('D:/GitHub/summer-research/data/dr8_redmapper_v6.3.1_members_n_clusters_masked.fits', overwrite=True)  # Replace 'modified_table.fits' with the desired file name or path