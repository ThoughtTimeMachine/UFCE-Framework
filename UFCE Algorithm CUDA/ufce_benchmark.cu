/*
 * Copyright (C) 2025 Kyle Killian
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// UFCE Configuration: 125 Billion Points
#define N_R 125000LL
#define N_T 1000000LL

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// --- TURBO KERNEL (2D Indexing) ---
// Optimization: Removes division/modulo by mapping hardware blocks to X/Y directly.
__global__ void ufce_kernel_2d(
    const float* __restrict__ grad_rho, 
    const float* __restrict__ h_t, 
    const float* __restrict__ sin_omega_t, 
    const float g, 
    const long long n_r, 
    const long long n_t,
    double* __restrict__ global_sum, 
    double* __restrict__ global_sq_sum
) {
    // 2D Grid Stride Loop
    long long idx_x = blockIdx.x * blockDim.x + threadIdx.x; // Temporal (t)
    long long idx_y = blockIdx.y * blockDim.y + threadIdx.y; // Spatial (r)
    
    long long stride_x = blockDim.x * gridDim.x;
    long long stride_y = blockDim.y * gridDim.y;

    double local_sum = 0.0;
    double local_sq = 0.0;

    // Iterate over Spatial (Y)
    for (long long r = idx_y; r < n_r; r += stride_y) {
        float spatial_term = grad_rho[r] * g;
        
        // Iterate over Temporal (X) - Coalesced Memory Access!
        for (long long t = idx_x; t < n_t; t += stride_x) {
            float val = spatial_term * h_t[t] * sin_omega_t[t];
            local_sum += val;
            local_sq += val * val;
        }
    }

    // Atomic Aggregation
    atomicAdd(global_sum, local_sum);
    atomicAdd(global_sq_sum, local_sq);
}

int main() {
    long long total_points = N_R * N_T;
    
    // Hardware Info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "UFCE Turbo Benchmark (2D Optimized): 125 Billion Points" << std::endl;

    // Host Alloc
    float *h_rho, *h_ht, *h_sin;
    cudaMallocHost(&h_rho, N_R * sizeof(float));
    cudaMallocHost(&h_ht, N_T * sizeof(float));
    cudaMallocHost(&h_sin, N_T * sizeof(float));

    // Init Data
    for(long long i=0; i<N_R; i++) h_rho[i] = sinf(i*0.01f);
    for(long long i=0; i<N_T; i++) {
        h_ht[i] = cosf(i*0.01f);
        h_sin[i] = sinf(i*0.01f);
    }

    // Device Alloc
    float *d_rho, *d_ht, *d_sin;
    double *d_sum, *d_sq, *h_sum, *h_sq;
    cudaMalloc(&d_rho, N_R * sizeof(float));
    cudaMalloc(&d_ht, N_T * sizeof(float));
    cudaMalloc(&d_sin, N_T * sizeof(float));
    cudaMalloc(&d_sum, sizeof(double));
    cudaMalloc(&d_sq, sizeof(double));
    cudaMallocHost(&h_sum, sizeof(double));
    cudaMallocHost(&h_sq, sizeof(double));

    // Copy
    cudaMemcpy(d_rho, h_rho, N_R * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ht, h_ht, N_T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin, N_T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(double));
    cudaMemset(d_sq, 0, sizeof(double));

    // --- TURBO LAUNCH CONFIG ---
    // We treat the grid as a 2D Block
    dim3 threads(256, 1);  // 256 threads per block working on Time (X)
    dim3 blocks(4096, 64); // Many blocks to cover the grid
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    ufce_kernel_2d<<<blocks, threads>>>(d_rho, d_ht, d_sin, 10.0f, N_R, N_T, d_sum, d_sq);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Results
    cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Time: " << ms/1000.0f << " s" << std::endl;
    std::cout << "Throughput: " << (total_points / (ms/1000.0f)) / 1e9 << " Billion pts/s" << std::endl;
    std::cout << "Sum Check: " << *h_sum << std::endl;
    std::cout << "--------------------------------" << std::endl;

    return 0;
}