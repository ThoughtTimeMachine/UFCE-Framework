#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <random>

// --- CONFIGURATION: 1 TRILLION OPS ---
// Scenario: 50,000 Queries x 20,000,000 Keys
// Total Operations: 1,000,000,000,000 (1 Trillion)
#define N_QUERIES 50000
#define N_KEYS    20000000

// --- GPU KERNEL: LINEAR ATTENTION ---
// Each thread handles ONE Query and scans ALL 20 Million Keys.
// This is "Brute Force" parallelism at its finest.
__global__ void attention_kernel_linear(
    const float* __restrict__ queries, 
    const float* __restrict__ keys, 
    float* __restrict__ output_scores,
    const int n_q, 
    const int n_k
) {
    // 1. Identify which Query this thread is responsible for
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Boundary Check
    if (idx < n_q) {
        float my_query_val = queries[idx];
        float max_score = -1e9f; // Start with a very low number

        // 3. The "trillion op" inner loop
        // The thread scans the entire context window (Keys)
        for (int k = 0; k < n_k; k++) {
            float score = my_query_val * keys[k];
            
            // Optimization: fmaxf is a hardware intrinsic on NVIDIA GPUs
            max_score = fmaxf(max_score, score);
        }

        // 4. Write result to global memory
        output_scores[idx] = max_score;
    }
}

int main() {
    long long total_ops = (long long)N_QUERIES * (long long)N_KEYS;

    // --- 1. Hardware Check ---
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " UFCE ATTENTION BENCHMARK (C++ CUDA)" << std::endl;
    std::cout << " Device: " << prop.name << std::endl;
    std::cout << " Scenario: " << N_KEYS / 1000000 << " Million Token Context Window" << std::endl;
    std::cout << " Total Ops: " << total_ops / 1e12 << " TRILLION" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // --- 2. Allocation (Host) ---
    std::cout << "Generating Data (Host)..." << std::endl;
    size_t size_q = N_QUERIES * sizeof(float);
    size_t size_k = N_KEYS * sizeof(float);
    size_t size_out = N_QUERIES * sizeof(float);

    // Use pinned memory for faster transfers
    float *h_q, *h_k, *h_out;
    cudaMallocHost(&h_q, size_q);
    cudaMallocHost(&h_k, size_k);
    cudaMallocHost(&h_out, size_out);

    // Initialize with random noise (std::mt19937 is too slow for 20M items, using fast fill)
    for (int i = 0; i < N_QUERIES; i++) h_q[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N_KEYS; i++)    h_k[i] = (float)rand() / RAND_MAX;

    // --- 3. Allocation (Device) ---
    float *d_q, *d_k, *d_out;
    cudaMalloc(&d_q, size_q);
    cudaMalloc(&d_k, size_k);
    cudaMalloc(&d_out, size_out);

    // --- 4. Transfer H2D ---
    std::cout << "Moving Data to GPU VRAM..." << std::endl;
    cudaMemcpy(d_q, h_q, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, size_k, cudaMemcpyHostToDevice);

    // --- 5. Launch Configuration ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_QUERIES + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching Kernel (" << blocksPerGrid << " blocks x " << threadsPerBlock << " threads)..." << std::endl;

    // Setup Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- 6. EXECUTE ---
    cudaEventRecord(start);
    attention_kernel_linear<<<blocksPerGrid, threadsPerBlock>>>(d_q, d_k, d_out, N_QUERIES, N_KEYS);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // --- 7. Validate ---
    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
    
    double seconds = milliseconds / 1000.0;
    double throughput = total_ops / seconds / 1e9; // Billions of ops per second

    std::cout << "\n--- GPU RESULTS ---" << std::endl;
    std::cout << "Time:       " << seconds << " s" << std::endl;
    std::cout << "Throughput: " << throughput << " Billion ops/sec" << std::endl;

    if (seconds < 5.0) {
        std::cout << "\n[SUCCESS] Hyper-Speed Achieved." << std::endl;
    } else {
        std::cout << "\n[NOTE] Completed successfully." << std::endl;
    }

    // Cleanup
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_out);
    cudaFreeHost(h_q); cudaFreeHost(h_k); cudaFreeHost(h_out);

    return 0;
}