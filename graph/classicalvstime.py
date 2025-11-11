import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = [
    (0.001684, 8), (0.001741, 8), (0.001707, 8),
    (0.000623, 7), (0.0007, 7), (0.000647, 7),
    (0.000262, 6), (0.000291, 6), (0.000268, 6),
    (0.0001, 5), (0.000127, 5), (0.000106, 5),
    (0.000035, 4), (0.000057, 4), (0.000038, 4)
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

def exp_func(n, a, b):
    return a * (b**n) * (n ** 2)

params, _ = curve_fit(exp_func, n_values, avg_times, p0=[1e-6, 2])
a_fit, b_fit = params

fit_n = np.linspace(min(n_values), max(n_values), 200)
plt.plot(fit_n, exp_func(fit_n, a_fit, b_fit),
         label=f"Fit: {a_fit:.3e} * ({b_fit:.3f})^n * n^2")

plt.title("Time for Classical Algorithm based on n")
plt.xlabel("n (rows)")
plt.ylabel("Time (s)")
plt.ylim(0, 0.002)
plt.xticks(n_values)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()
