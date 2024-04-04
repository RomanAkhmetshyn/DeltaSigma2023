import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from colossus.cosmology import cosmology
from NFW_funcs import quick_MK_profile

params = {"flat": True, "H0": 100, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81, "ns": 0.95}
cosmology.addCosmology("737", params)
cosmo = cosmology.setCosmology("737")

lenses = Table.read("C:/catalogs/members_n_clusters_masked.fits")
#Combined by myself with host halo masses and redshifts - email me if you want it

lowlim=0.3
highlim=0.6
#filter lenses that are in a distance bin. You can also filter by membership probability and redshift
data_mask = (
        (lenses["R"] >= lowlim)
        & (lenses["R"] < highlim)
        # & (lenses["PMem"] > 0.8)
        # & (lenses["zspec"] > -1)
        & (lenses["PMem"] > 0.8)
    )
lenses = lenses[data_mask] #updated table of lenses
cdf_resolution=1000 #interpolation resulotion, i.e. number of points for probability distribution func.
mass_per_point = 1098372.008822474*10000 #arbitrary number, Mass/mpp = number of points for M-C

start_bin=0.01 #first ring lens-centric disatnce Mpc
end_bin=1.5 #final ring lens-centric distance
ring_incr=0.02 #distance between rings
ring_num=round((end_bin-start_bin)/ring_incr) #number of rings
ring_radii = np.linspace(start_bin, end_bin, ring_num+1) * 1000 #Mpc*1000=kpc, radii of all rings in kps
threshold=(ring_incr/2*100)*9 #the same small width for each ring.

mdef="200m" #cosmological mass definition 

DeltaSigmas=np.zeros((1, len(ring_radii))) #np array for all delta sigma measurments

sat=lenses[100]

#%%

random_radii_x, random_radii_y = quick_MK_profile(sat['M_halo']*1.429,
                                                  #here I multiplied by 1.429 cuz I calculated
                                                  #masses for H=70 cosmology
                                                  sat['Z_halo'],
                                                  mass_per_point, 
                                                  "duffy08",
                                                  "200m",
                                                  cdf_resolution)


with plt.rc_context({"axes.grid": False}):
    fig, ax = plt.subplots(dpi=100)
    img = ax.hexbin(random_radii_x, random_radii_y, gridsize=100, bins="log")
    ax.plot(0, 0, "r+")
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('R (kpc)')
    # for radius in ring_radii:
    #     circle = plt.Circle((sat_x, sat_y), radius, edgecolor='red', facecolor='none')
    #     ax.add_patch(circle)
    
    # ax.set_ylim(-radius, radius)
    # ax.set_xlim(-radius, radius)
    fig.colorbar(img)
    ax.set_aspect("equal")
    # ax.set_title(f"{len(multi_lenses)} halos")
    plt.title(f'number of MC points: {len(random_radii_x)}, R_vir={max(random_radii_x):.2f}')
    plt.show()
    
plt.scatter(random_radii_x, random_radii_y, s=0.001)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot()

#%%

sat_x = sat['R'] * 1000 #Mpc*1000 convert coords to kpc
sat_y = 0

S=[np.pi*((r+threshold)**2-(r-threshold)**2) for r in ring_radii] #area if rings
# Calculate the distances for all random points at once
distances = np.sqrt((random_radii_x - sat_x)**2 + (random_radii_y - sat_y)**2)

# Create an empty array to store the counts for each ring
ring_counts = np.zeros(len(ring_radii), dtype=np.int64) #counts in the rings
circle_counts = np.zeros(len(ring_radii), dtype=np.int64) #counts in enclosed circles

# Iterate over each ring radius and count the points within each ring


for i in range(len(ring_radii)):
    ring_mask = np.logical_and(ring_radii[i] - threshold <= distances, 
                          distances <= ring_radii[i] + threshold) #mask points that are within ring
    
    # plt.scatter(random_radii_x, random_radii_y, s=0.001)
    # plt.scatter(random_radii_x[mask], random_radii_y[mask], s=0.001, c='red')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    
    ring_counts[i] = np.sum(ring_mask)*mass_per_point/S[i] #get surface density in rings
    

    circle_mask = np.logical_and(0 <= distances, distances <= ring_radii[i] - threshold)
    
    
    circle_counts[i] = np.sum(circle_mask)*mass_per_point/(np.pi*(ring_radii[i]- threshold)**2)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].scatter(random_radii_x, random_radii_y, s=0.001, alpha=0.4)
    axs[0].scatter(random_radii_x[circle_mask], random_radii_y[circle_mask], s=0.001, c='red')
    axs[0].scatter(random_radii_x[ring_mask], random_radii_y[ring_mask], s=0.001, c='black')
    axs[0].plot(0, 0, "b+")
    axs[0].plot(sat_x, sat_y, "k+")
    # axs[0].set_title('Scatter Plot')
    # axs[0].set_xlabel('X-axis')
    # axs[0].set_ylabel('Y-axis')
    axs[0].set_aspect('equal', adjustable='box')
    
    axs[1].plot(ring_radii, ring_counts, label=r'$\Sigma$(R)')
    axs[1].plot(ring_radii, circle_counts, label=r'$\Sigma$(<R)')
    axs[1].plot(ring_radii, circle_counts-ring_counts, label=r'Delta $\Sigma$')
    # axs[1].set_title('Cumulative Counts')
    axs[1].set_xlabel('R (kpc)')
    axs[1].set_ylabel('dsigma')
    axs[1].set_xlim([ring_radii[0], ring_radii[-1]])
    axs[1].legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()



sums=[]

for i in range(len(ring_radii)-1,-1,-1): #iterate through each radii
    DeltalessR=circle_counts[i]
    DeltaR=ring_counts[i]
    sums.append(DeltalessR-DeltaR) #Delta Sigmas for each ring-circle pair

# t=time.time() - debug_start
# print(num)
DeltaSigmas=np.add(DeltaSigmas,np.array(sums))