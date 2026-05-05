import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# --- split gaussian PDF ---


def split_gaussian(x, mu, sigma_left, sigma_right):
    x = np.asarray(x)
    pdf = np.zeros_like(x, dtype=float)
    left_mask = x < mu
    right_mask = ~left_mask
    A = np.sqrt(2 / np.pi) * 1.0 / (sigma_left + sigma_right)
    if left_mask.any():
        pdf[left_mask] = A * np.exp(-0.5 * ((x[left_mask] - mu) / sigma_left) ** 2)
    if right_mask.any():
        pdf[right_mask] = A * np.exp(-0.5 * ((x[right_mask] - mu) / sigma_right) ** 2)
    return pdf

# --- sampler ---


def sample_split_gaussian(mu, sigma_left, sigma_right, size=1, random_state=None):
    rng = np.random.default_rng(random_state)
    p_left = sigma_left / (sigma_left + sigma_right)
    is_left = rng.random(size) < p_left
    samples = np.empty(size, dtype=float)
    n_left = int(is_left.sum())
    if n_left > 0:
        samples[is_left] = mu - np.abs(rng.normal(0, sigma_left, n_left))
    n_right = size - n_left
    if n_right > 0:
        samples[~is_left] = mu + np.abs(rng.normal(0, sigma_right, n_right))
    return samples

# --- analytic CDF ---


def cdf_split(x, mu, alpha, beta):
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    left = x < mu
    right = ~left
    if left.any():
        out[left] = (2.0 * alpha / (alpha + beta)) * norm.cdf((x[left] - mu) / alpha)
    if right.any():
        out[right] = (alpha / (alpha + beta)) + (2.0 * beta / (alpha + beta)
                                                 ) * (norm.cdf((x[right] - mu) / beta) - 0.5)
    return out

# --- objective for minimizer ---


def objective_cdf(params, mu, left, right, target_left=0.16, target_right=0.84):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return 1e6
    x_left = mu - left
    x_right = mu + right
    p_left = cdf_split(x_left, mu, alpha, beta)
    p_right = cdf_split(x_right, mu, alpha, beta)
    return (p_left - target_left)**2 + (p_right - target_right)**2


def optimize_sigmas(mu, left, right):
    inits = [
        [max(1e-3, left), max(1e-3, right)],
        [max(1e-3, left * 0.8), max(1e-3, right * 1.2)],
        [max(1e-3, left * 1.2), max(1e-3, right * 0.8)],
    ]
    best = None
    for x0 in inits:
        res = minimize(objective_cdf, x0=x0, args=(mu, left, right),
                       bounds=[(1e-8, None), (1e-8, None)], method='L-BFGS-B')
        if best is None or res.fun < best.fun:
            best = res
    return best.x[0], best.x[1], best


# --- input data ---
data = [
    (11.14, 0.54, 1.20, 10.12, 0.27, 0.27),
    (11.75, 0.24, 0.44, 10.08, 0.28, 0.26),
    (11.04, 0.59, 1.19, 10.46, 0.23, 0.31),
    (12.17, 0.14, 0.17, 10.32, 0.28, 0.25),
    (12.54, 0.12, 0.14, 11.03, 0.13, 0.29),
    (12.75, 0.07, 0.08, 11.03, 0.16, 0.30),
]

results_sub = []
results_bary = []

for row in data:
    mu_sub, plus_sub, minus_sub, mu_bary, plus_b, minus_b = row
    alpha_sub, beta_sub, res_sub = optimize_sigmas(mu_sub, minus_sub, plus_sub)
    alpha_b, beta_b, res_b = optimize_sigmas(mu_bary, minus_b, plus_b)

    results_sub.append((mu_sub, minus_sub, plus_sub, alpha_sub, beta_sub, res_sub.fun))
    results_bary.append((mu_bary, minus_b, plus_b, alpha_b, beta_b, res_b.fun))

print("Optimized sigmas for logM_sub (mu, target -left, +right → alpha, beta)")
for r in results_sub:
    print(f"mu={r[0]:.2f}, target(-{r[1]:.3f}, +{r[2]:.3f}) -> alpha={r[3]:.4f}, beta={r[4]:.4f}, obj={r[5]:.2e}")

print("\nOptimized sigmas for logM_bary (mu, target -left, +right → alpha, beta)")
for r in results_bary:
    print(f"mu={r[0]:.2f}, target(-{r[1]:.3f}, +{r[2]:.3f}) -> alpha={r[3]:.4f}, beta={r[4]:.4f}, obj={r[5]:.2e}")

# --- plot example for the first row (both sub and bary) ---
random_state = 12345
n_samples = 50000

# Sub
mu_sub, left_sub, right_sub, alpha_sub, beta_sub, _ = results_sub[0]
samples_sub = sample_split_gaussian(
    mu_sub, alpha_sub, beta_sub, size=n_samples, random_state=random_state)
p16_sub, p84_sub = np.percentile(samples_sub, [16, 84])
x_sub = np.linspace(mu_sub - 5 * alpha_sub, mu_sub + 5 * beta_sub, 2001)
pdf_sub = split_gaussian(x_sub, mu_sub, alpha_sub, beta_sub)

plt.figure(figsize=(7, 4))
plt.hist(samples_sub, bins=100, density=True, alpha=0.6, label='samples')
plt.plot(x_sub, pdf_sub, lw=2, label='pdf')
plt.vlines([p16_sub, p84_sub], 0, pdf_sub.max(),
           linestyles='--', label='sample percentiles (16/84)')
plt.vlines([mu_sub - left_sub, mu_sub + right_sub], 0,
           pdf_sub.max(), linestyles=':', label='target +/- errors')
plt.axvline(mu_sub, linestyle=':', linewidth=0.8)
plt.xlabel('logM_sub')
plt.ylabel('Density')
plt.legend()
plt.title(f"row 0: sub (mu={mu_sub:.2f})")
plt.show()

# Bary
mu_b, left_b, right_b, alpha_b, beta_b, _ = results_bary[0]
samples_b = sample_split_gaussian(
    mu_b, alpha_b, beta_b, size=n_samples, random_state=random_state)
p16_b, p84_b = np.percentile(samples_b, [16, 84])
x_b = np.linspace(mu_b - 5 * alpha_b, mu_b + 5 * beta_b, 2001)
pdf_b = split_gaussian(x_b, mu_b, alpha_b, beta_b)

plt.figure(figsize=(7, 4))
plt.hist(samples_b, bins=100, density=True, alpha=0.6, label='samples')
plt.plot(x_b, pdf_b, lw=2, label='pdf')
plt.vlines([p16_b, p84_b], 0, pdf_b.max(), linestyles='--',
           label='sample percentiles (16/84)')
plt.vlines([mu_b - left_b, mu_b + right_b], 0, pdf_b.max(),
           linestyles=':', label='target +/- errors')
plt.axvline(mu_b, linestyle=':', linewidth=0.8)
plt.xlabel('logM_bary')
plt.ylabel('Density')
plt.legend()
plt.title(f"row 0: bary (mu={mu_b:.2f})")
plt.show()
