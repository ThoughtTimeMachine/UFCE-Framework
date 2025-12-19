import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import os

# --- 1. UFCE Core Function (Returns a Matrix for Visualization) ---
# Note: Unlike the streaming benchmark, this version returns the 2D matrix
# so Matplotlib has data to draw. We limit the size to avoid OOM here.
@njit(parallel=True)
def compute_gamma_matrix(grad_rho, h_t, sin_omega_t, n_r_plot, n_t_plot, g):
    gamma = np.zeros((n_r_plot, n_t_plot), dtype=np.float64)
    for i in prange(n_r_plot):
        for j in range(n_t_plot):
            # The UFCE Product Structure: Gamma = g * grad_rho * h(t) * sin(omega*t)
            gamma[i, j] = g * grad_rho[i] * h_t[j] * sin_omega_t[j]
    return gamma

def generate_and_plot_heatmap(output_dir='ufce_visualizations'):
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Parameters for Visualization ---
    # We use a structured, representative grid to show the physics clearly.
    n_r_plot = 200    # Spatial resolution
    n_t_plot = 500    # Temporal resolution
    g = 10.0          # Coupling constant
    omega = 2 * np.pi * 5 # Frequency (5 cycles over the timeframe)

    print(f"Generating Visualization: {n_r_plot} (Space) x {n_t_plot} (Time)")

    # --- 3. Generate Structured Input Data ---
    # Spatial Coordinate (r): 0 to 10
    r_coords = np.linspace(0, 10, n_r_plot).astype(np.float64)
    # Time Coordinate (t): 0 to 1
    t_coords = np.linspace(0, 1, n_t_plot).astype(np.float64)

    # A: Spatial Gradient (grad_rho) -> Decays exponentially over space
    # This simulates a field getting weaker as you move away from a source.
    grad_rho_plot = np.exp(-r_coords / 3).astype(np.float64)
    
    # B: Temporal Signal (h_t) -> Increases linearly over time
    # This simulates a system load getting heavier.
    h_t_plot = (t_coords + 0.2).astype(np.float64)
    
    # C: Oscillatory Term -> Standard sine wave
    sin_omega_t_plot = np.sin(omega * t_coords).astype(np.float64)

    # --- 4. Compute the Matrix ---
    print("Computing Interaction Matrix...")
    gamma_matrix = compute_gamma_matrix(
        grad_rho_plot, h_t_plot, sin_omega_t_plot, 
        n_r_plot, n_t_plot, g
    )

    # --- 5. Plotting ---
    print("Rendering Heatmap...")
    plt.figure(figsize=(12, 8))
    
    # Create the Heatmap
    # extent=[xmin, xmax, ymin, ymax] sets the axis values correctly
    im = plt.imshow(
        gamma_matrix, 
        cmap='inferno',       # 'inferno' provides high contrast for hotspots
        aspect='auto', 
        origin='lower',       # (0,0) at bottom-left
        extent=[t_coords.min(), t_coords.max(), r_coords.min(), r_coords.max()]
    )
    
    # Labels and Titles
    cbar = plt.colorbar(im)
    cbar.set_label(r'Interaction Strength $\Gamma(r, t)$', rotation=270, labelpad=20)
    
    plt.title(r'UniField Coupling: Spatial Gradient $\times$ Temporal Oscillation', fontsize=16, pad=15)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Spatial Distance (r)', fontsize=12)

    # Annotate the Physics
    plt.text(
        0.05, 0.95, 
        r'$\Gamma(r,t) = g \cdot e^{-r/3} \cdot (t + 0.2) \cdot \sin(\omega t)$', 
        transform=plt.gca().transAxes, 
        fontsize=12, color='white', verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.7)
    )

    # --- 6. Save Output ---
    output_path = os.path.join(output_dir, 'ufce_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print("-" * 40)
    print(f"Visual Proof Saved: {output_path}")
    print("-" * 40)

if __name__ == "__main__":
    generate_and_plot_heatmap()