import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import sqrt

data = [
    (0.2556854, 8), (0.198989, 7), (0.052659, 6), (0.04092, 5), (0.039334, 4),
    (0.186999, 7), (0.247336, 7), (0.043453, 6), (0.103711, 6), (0.03525, 5),
    (0.069587, 5), (0.031794, 4), (0.07955, 4), (0.2716971999, 8), (0.2417275, 8)
]

results = {}
for time, n in data:
    results.setdefault(n, []).append(time)

n_values = sorted(results.keys())
avg_times = [np.mean(results[n]) for n in n_values]
min_times = [min(results[n]) for n in n_values]
max_times = [max(results[n]) for n in n_values]

plt.figure(figsize=(8, 6))
plt.errorbar(
    n_values, avg_times,
    yerr=[np.subtract(avg_times, min_times), np.subtract(max_times, avg_times)],
    fmt='o', capsize=5, label="Average ± Range"
)

def sqrt_exp_func(n, a, b):
    return a * (n ** 2) * (b**(n/2))

params, _ = curve_fit(sqrt_exp_func, n_values, avg_times, p0=[1e-3, 2])
a_fit, b_fit = params

fit_n = np.linspace(min(n_values), max(n_values), 200)
plt.plot(
    fit_n, sqrt_exp_func(fit_n, a_fit, b_fit),
    label=f"Fit: {a_fit:.3e} * ({b_fit:.3f})^(n/2) * n^2"
)

plt.title("Time for Quantum Algorithm based on n")
plt.xlabel("n (rows)")
plt.ylabel("Time (s)")
plt.ylim(0, 0.3)
plt.xticks(n_values)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()
