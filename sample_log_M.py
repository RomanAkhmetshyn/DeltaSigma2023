# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 14:46:17 2025

@author: romix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
# Monte Carlo for asymmetric errors (split normal) and ratio computation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


np.random.seed(42)


def sample_split_gaussian(mu, sigma_left, sigma_right, size=1, random_state=None):
    """
    Draw random samples from a split Gaussian distribution.
    """
    rng = np.random.default_rng(random_state)
    # Probability of drawing from left side
    p_left = sigma_left / (sigma_left + sigma_right)
    # Choose side
    is_left = rng.random(size) < p_left
    samples = np.zeros(size)
    # Left side samples (reflected Gaussian)
    n_left = is_left.sum()
    if n_left > 0:
        samples[is_left] = mu - np.abs(rng.normal(0, sigma_left, n_left))
    # Right side samples
    n_right = size - n_left
    if n_right > 0:
        samples[~is_left] = mu + np.abs(rng.normal(0, sigma_right, n_right))
    return samples


# Input table transcribed from your image (log M_sub, +err, -err ; log M_bary, +err, -err)
data = [
    (11.14, 1.0650, 0.6645, 10.12, 0.2715, 0.2715),
    (11.75, 0.4002, 0.2789, 10.08, 0.2654, 0.2775),
    (11.04, 1.0688, 0.7049, 10.46, 0.2953, 0.2468),
    (12.17, 0.1648, 0.1467, 10.32, 0.2573, 0.2755),
    (12.54, 0.1367, 0.1246, 11.03, 0.2573, 0.1602),
    (12.75, 0.0784, 0.0724, 11.03, 0.2720, 0.1871),
]

rows = []
n_samples = 20000  # increase if you want better precision

for i, (logh, hm, hp, logb, bm, bp) in enumerate(data, start=1):
    samp_h = sample_split_gaussian(logh, hm, hp, n_samples)
    samp_h_med = logh
    samp_h_p16, samp_h_p84 = np.percentile(samp_h, [16, 84])
    # plt.hist(samp_h, bins=100)
    # plt.title(f'{samp_h_med:.2f} +{samp_h_p84-samp_h_med:.2f} -{samp_h_med-samp_h_p16:.2f}')
    # plt.show()
    samp_b = sample_split_gaussian(logb, bm, bp, n_samples)
    # ratio = 10^(logh - logb)
    log_diff = samp_h - samp_b
    ratio = 10**(log_diff)
    plt.hist(ratio, bins=1000)
    plt.yscale('log')
    plt.xlim(0, 100)
    plt.show()
    # Summaries
    median = np.median(ratio)
    p16, p84 = np.percentile(ratio, [16, 84])
    p2, p98 = np.percentile(ratio, [2.5, 97.5])
    # also get log-diff summaries
    med_ld = np.median(log_diff)
    l16_ld, h84_ld = np.percentile(log_diff, [16, 84])
    rows.append({
        "row": i,
        "logM_sub": logh,
        "logM_sub_+": hp,
        "logM_sub_-": hm,
        "logM_bary": logb,
        "logM_bary_+": bp,
        "logM_bary_-": bm,
        "ratio_median": median,
        "ratio_-1sigma": median - p16,
        "ratio_+1sigma": p84 - median,
        "ratio_2.5%": p2,
        "ratio_97.5%": p98,
        "logdiff_median": med_ld,
        "logdiff_16": l16_ld,
        "logdiff_84": h84_ld,
    })

df = pd.DataFrame(rows)
