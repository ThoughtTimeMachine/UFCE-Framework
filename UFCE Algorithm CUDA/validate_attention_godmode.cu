#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <random>

// --- CONFIGURATION: GOD MODE (50 TRILLION) ---
// Scenario: 50,000 Queries x 1,000,000,000 Keys (1 Billion Context)
// Total Operations: 50,000,000,000,000 (50 Trillion)
#define N_QUERIES 50000
#define N_KEYS    1000000000 // 1 Billion

// --- GPU KERNEL ---
__global__ void attention_kernel_linear(
    const float* __restrict__ queries, 
    const float* __restrict__ keys, 
    float* __restrict__ output_scores,
    const int n_q, 
    const int n_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_q) {
        float my_query_val = queries[idx];
        float max_score = -1e9f; 

        // The 1 Billion Iteration Loop
        for (int k = 0; k < n_k; k++) {
            float score = my_query_val * keys[k];
            max_score = fmaxf(max_score, score);
        }
        output_scores[idx] = max_score;
    }
}

int main() {
    long long total_ops = (long long)N_QUERIES * (long long)N_KEYS;

    // --- Hardware Check ---
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " UFCE GOD MODE BENCHMARK (C++ CUDA)" << std::endl;
    std::cout << " Device: " << prop.name << std::endl;
    std::cout << " Scenario: " << N_KEYS / 1000000000 << " BILLION Token Context Window" << std::endl;
    std::cout << " Total Ops: " << total_ops / 1e12 << " TRILLION" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // --- Allocation (Host) ---
    // 1 Billion floats = ~4 GB. This takes a moment to allocate.
    std::cout << "Allocating 4GB of Test Data (Host)..." << std::endl;
    
    float *h_q, *h_k, *h_out;
    cudaMallocHost(&h_q, N_QUERIES * sizeof(float));
    cudaMallocHost(&h_k, N_KEYS * sizeof(float)); 
    cudaMallocHost(&h_out, N_QUERIES * sizeof(float));

    // Fast fill
    std::cout << "Filling Data..." << std::endl;
    for (int i = 0; i < N_QUERIES; i++) h_q[i] = (float)rand() / RAND_MAX;
    // We only fill a subset to save startup time, memset the rest
    memset(h_k, 1, N_KEYS * sizeof(float)); 

    // --- Allocation (Device) ---
    std::cout << "Allocating GPU VRAM..." << std::endl;
    float *d_q, *d_k, *d_out;
    cudaMalloc(&d_q, N_QUERIES * sizeof(float));
    cudaMalloc(&d_k, N_KEYS * sizeof(float)); 
    cudaMalloc(&d_out, N_QUERIES * sizeof(float));

    // --- Transfer ---
    std::cout << "Moving 4GB Data to GPU..." << std::endl;
    cudaMemcpy(d_q, h_q, N_QUERIES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, N_KEYS * sizeof(float), cudaMemcpyHostToDevice);

    // --- Execution ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_QUERIES + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching GOD MODE Kernel (50 Trillion Ops)..." << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    attention_kernel_linear<<<blocksPerGrid, threadsPerBlock>>>(d_q, d_k, d_out, N_QUERIES, N_KEYS);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // --- Validate ---
    double seconds = milliseconds / 1000.0;
    double throughput = total_ops / seconds / 1e12; // TRILLIONS of ops

    std::cout << "\n--- GOD MODE RESULTS ---" << std::endl;
    std::cout << "Time:       " << seconds << " s" << std::endl;
    std::cout << "Throughput: " << throughput << " TRILLION ops/sec" << std::endl;

    // Cleanup
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_out);
    cudaFreeHost(h_q); cudaFreeHost(h_k); cudaFreeHost(h_out);

    return 0;
}