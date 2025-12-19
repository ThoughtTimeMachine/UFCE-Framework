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

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_benchmark_results():
    # Data from your actual runs
    # Run 1: 2.9B points -> 42.66 B pts/sec
    # Run 2: 10B points -> 45.25 B pts/sec
    # Run 3: 125B points -> 43.33 B pts/sec
    
    points = np.array([2.87, 10.0, 125.0]) # Billions
    throughput = np.array([42.66, 45.25, 43.33]) # Billions pts/sec
    
    plt.figure(figsize=(10, 6))
    
    # Plot bars
    bars = plt.bar(
        [str(p) + "B" for p in points], 
        throughput, 
        color=['#4c72b0', '#55a868', '#c44e52'],
        alpha=0.9,
        edgecolor='black'
    )
    
    # Add title and labels
    plt.title('UFCE Throughput Scaling (Linear Scalability Proof)', fontsize=14, pad=15)
    plt.ylabel('Throughput (Billion Points / Sec)', fontsize=12)
    plt.xlabel('Workload Size (Interaction Points)', fontsize=12)
    plt.ylim(0, 55)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f} B/s',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    # Add annotation about O(N)
    plt.text(
        1, 50, 
        "Consistent Throughput = O(N) Linear Scaling", 
        ha='center', fontsize=12, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Save
    os.makedirs('ufce_visualizations', exist_ok=True) # Ensure dir exists
    # Save to current dir so LaTeX finds it easily, or move it later
    plt.savefig('ufce_throughput.png', dpi=300, bbox_inches='tight')
    print("Throughput plot saved as 'ufce_throughput.png'")

if __name__ == "__main__":
    plot_benchmark_results()