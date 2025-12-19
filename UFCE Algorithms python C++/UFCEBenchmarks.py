# Copyright (C) 2025 Kyle Killian
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numba import njit, prange
import time
import psutil
import matplotlib.pyplot as plt
import json
import multiprocessing
import os

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # In GB

# UFCE core function
@njit(parallel=True)
def compute_ufce(grad_rho, h, sin_omega_t, n_r, n_t, g):
    gamma = np.zeros((n_r, n_t), dtype=np.float64)
    for i in prange(n_r):
        for j in range(n_t):
            gamma[i, j] = g * grad_rho[i] * h[j] * sin_omega_t[j]
    return gamma

# Benchmark function
def benchmark_ufce(n_r, n_t, g=10.0, omega=2 * np.pi * 252, output_dir='ufce_benchmarks'):
    # Generate test data
    grad_rho = np.sin(np.linspace(0, 2 * np.pi, n_r))  # Spatial gradient
    t = np.linspace(0, 2 * np.pi, n_t)
    h = np.cos(t)  # Temporal signal
    sin_omega_t = np.sin(omega * t)  # Oscillatory term

    # Pre-benchmark memory
    mem_start = get_memory_usage()

    # Time computation
    start_time = time.time()
    gamma = compute_ufce(grad_rho, h, sin_omega_t, n_r, n_t, g)
    end_time = time.time()
    compute_time = end_time - start_time

    # Post-benchmark memory
    mem_end = get_memory_usage()
    mem_used = mem_end - mem_start

    # Performance metrics
    total_points = n_r * n_t
    points_per_sec = total_points / compute_time if compute_time > 0 else 0
    gamma_mean = np.mean(gamma)
    gamma_variance = np.var(gamma)
    gamma_max = np.max(gamma)
    gamma_min = np.min(gamma)

    # Detailed output related to paper
    print(f"Benchmark Results for n_r={n_r}, n_t={n_t} (Total Points: {total_points:,})")
    print(f"Computation Time: {compute_time:.4f} seconds")
    print(f"Points per Second: {points_per_sec:,.2f}")
    print(f"Memory Used: {mem_used:.2f} GB")
    print(f"Gamma Mean (Paper Reference: ~4.17e-05): {gamma_mean:.2e}")
    print(f"Gamma Variance: {gamma_variance:.2e}")
    print(f"Gamma Max: {gamma_max:.2e}")
    print(f"Gamma Min: {gamma_min:.2e}")

    # Catalog results to JSON
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "n_r": n_r,
        "n_t": n_t,
        "total_points": total_points,
        "compute_time_s": compute_time,
        "points_per_sec": points_per_sec,
        "memory_used_gb": mem_used,
        "gamma_mean": gamma_mean,
        "gamma_variance": gamma_variance,
        "gamma_max": gamma_max,
        "gamma_min": gamma_min,
        "cpu_cores": multiprocessing.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{output_dir}/benchmark_{n_r}x{n_t}.json", "w") as f:
        json.dump(result, f, indent=4)

    # Generate charts if total_points > 0
    if total_points > 0:
        # Histogram of Gamma values
        plt.figure(figsize=(8, 6))
        plt.hist(gamma.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title(f"Gamma Distribution (n_r={n_r}, n_t={n_t})")
        plt.xlabel("Gamma Value")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/gamma_hist_{n_r}x{n_t}.png")
        plt.close()

        # Memory usage plot (simple bar for single test)
        plt.figure(figsize=(6, 4))
        plt.bar(["Memory Used"], [mem_used], color='green')
        plt.title("Memory Usage (GB)")
        plt.ylabel("GB")
        plt.savefig(f"{output_dir}/memory_usage_{n_r}x{n_t}.png")
        plt.close()

    return result

# Main test loop to push limits (up to 100GB RAM cap)
if __name__ == "__main__":
    # Config: Push to large grids, but cap total memory ~100GB (float64 = 8 bytes/point)
    max_ram_gb = 100
    g = 10.0
    omega = 2 * np.pi * 252
    output_dir = "ufce_cpu_benchmarks"

    # Start with small test
    benchmark_ufce(4500, 639058)  # Paper baseline: 2.9B points

    # Scale up to max: Estimate points = max_ram_gb * 1024^3 / 8
    max_points = int(max_ram_gb * (1024 ** 3) / 8)
    n_t = 10000000  # Fixed temporal size for consistency
    n_r = max_points // n_t  # Adjust spatial to fit

    print(f"Max estimated points: {max_points:,} (n_r={n_r:,}, n_t={n_t:,})")
    benchmark_ufce(n_r, n_t, g=g)