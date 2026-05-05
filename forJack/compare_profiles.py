# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 17:27:41 2025

@author: romix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

original = np.genfromtxt('0609_342.96_rayleigh.txt')
modified = np.genfromtxt('0609_342.96_rayleigh200m.txt')

plt.plot(original[:, 0], original[:, 1]/1000000, label='original at sigma=343')
plt.plot(modified[:, 0], modified[:, 1]/1000000, label='200m masses at sigma=343')
plt.legend()
plt.show()

plt.plot(modified[:, 0], modified[:, 1]/original[:, 1])
plt.hlines(np.power(2, 1/3), 0, 3500, ls='--', color='r')
plt.ylim(0, 2)
plt.show()
