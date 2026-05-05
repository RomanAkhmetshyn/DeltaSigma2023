import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

mpl.rcParams['figure.dpi'] = 300


def split_gaussian(x, mu, sigma_left, sigma_right):
    pdf = np.zeros_like(x)
    left_mask = x < mu
    right_mask = x >= mu
    A = np.sqrt(2 / np.pi) * 1 / (sigma_left + sigma_right)
    pdf[left_mask] = A * np.exp(-0.5 * ((x[left_mask] - mu) / sigma_left) ** 2)
    pdf[right_mask] = A * np.exp(-0.5 * ((x[right_mask] - mu) / sigma_right) ** 2)
    return pdf


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


# --- Demo ---
if __name__ == "__main__":
    mu = 11.75
    left = 0.44
    right = 0.24
    sigma_left = 0.4002
    sigma_right = 0.2789

    # Generate samples
    samples = sample_split_gaussian(mu, sigma_left, sigma_right, size=100000)
    samp_h_p16, samp_h_p84 = np.percentile(samples, [16, 84])
    print('median: ', mu)
    print('-', mu - samp_h_p16)
    print('+', samp_h_p84 - mu)

    # Plot histogram vs analytic PDF
    x = np.linspace(mu - 5 * sigma_left, mu + 5 * sigma_right, 10001)
    pdf = split_gaussian(x, mu, sigma_left, sigma_right)

    pdf_left_gauss = norm.pdf(x, loc=mu, scale=sigma_left)
    scale_left = np.amax(pdf_left_gauss) / np.amax(pdf)
    pdf_right_gauss = norm.pdf(x, loc=mu, scale=sigma_right)
    scale_right = np.amax(pdf_right_gauss) / np.amax(pdf)

    plt.hist(samples, bins=100, density=True, alpha=0.6, label="Samples")
    plt.vlines(mu, 0, 0.5, lw=1, ls='--', color='k')
    plt.vlines([samp_h_p16, samp_h_p84], 0, 0.5, lw=1, ls='--',
               color='g', label='16th, 84th percentile')
    plt.vlines([mu - left, mu + right], 0, 0.5, lw=1,
               ls='--', color='r', label='required +/-1σ')
    plt.plot(x, pdf, "r-", lw=2, label="PDF")

    plt.plot(x, pdf_left_gauss / scale_left, color="blue",
             alpha=0.4, ls="--", label=f"Gaussian σ={sigma_left}")
    plt.plot(x, pdf_right_gauss / scale_right, color="blue",
             alpha=0.4, ls=":", label=f"Gaussian σ={sigma_right}")

    plt.xlabel("logM")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
