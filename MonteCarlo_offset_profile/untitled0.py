import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

profile_folder ='new-test/'

profiles = [
    '0103(H70)_ext(R0).txt',
    # '0103(H70)_ext(R1).txt',
    '0306(H70)_ext(R0).txt',
    # '0306(H70)_ext(R1).txt',
    '0609(H70)_ext(R0).txt',
    # '0609(H70)_ext(R1).txt',
    # '0103(H70)_(R0)colossus.txt',
    # '0306(H70)_(R0)colossus.txt',
    # '0609(H70)_(R0)colossus.txt',
    # '0103C_Xu.txt',
    # '0306C_Xu.txt',
    # '0609C_Xu.txt',
    # '0103(H70)_ext(R1).txt',
    # '0306(H70)_ext(R1).txt',
    '0103_rayleigh.txt',
    '0306_rayleigh.txt',
    '0609_rayleigh.txt',
]


plt.figure(figsize=(8, 6))
for i, file in enumerate(profiles):
    halo = np.genfromtxt(profile_folder + file)
    R = halo[1:, 0] / 1000
    DS = halo[1:, 1] / 1000000
    
    if 'rayleigh' in file:
        plt.plot(R, DS*0.3, linestyle='--', c='blue', alpha=0.6)
    else:
        plt.plot(R, DS, c='k')


# plt.text(1.3, 82, '0.1 - 0.3', fontsize=10, color='black', ha='right', va='center')
# plt.text(1.9, 50, '0.3 - 0.6', fontsize=10, color='black', ha='right', va='center')
# plt.text(1.6, -12, '0.6 - 0.9', fontsize=10, color='black', ha='right', va='center')

plt.text(3.5, 40, 'previous offset halo model', fontsize=16, color='black', ha='right', va='center')
plt.text(3.5, -10, 'halo model with rayleigh offset', fontsize=16, color='blue', ha='right', va='center')

data_path = 'C:/scp'  
df = pd.read_csv(data_path+'/roman_esd_70ShapePipe_redmapper_clusterDist0.1_randomsTrue_1.csv')
ds = (df['ds']).values
rp = (df['rp']).values
ds_err=df['ds_err']

plt.ylabel(r'$\Delta \Sigma (R) \, [M_\odot / \mathrm{pc}^2]$', fontsize=12, labelpad=0)
plt.xlabel(r'$R (\mathrm{Mpc})$', fontsize=12)
      
# plt.errorbar(rp, ds, ds_err, fmt='o',label='dsigma Data', 
#              markerfacecolor='none', markeredgecolor='k', markeredgewidth=2)
plt.xlim(0, 3.5)
plt.grid()
# plt.legend()
plt.savefig('halo_models_rayleigh.png', bbox_inches='tight', dpi =300)
plt.show()
