#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define N 1000000                 // Размер входного массива
#define BLOCK_SIZE 256            // Потоков в блоке

// ------------------------------------------------------------
// Проверка ошибок CUDA
// ------------------------------------------------------------
static void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ------------------------------------------------------------
// CPU prefix sum (inclusive)
// ------------------------------------------------------------
void cpu_scan(const float* in, float* out, int n) {
    out[0] = in[0];
    for (int i = 1; i < n; i++) {
        out[i] = out[i - 1] + in[i];
    }
}

// ------------------------------------------------------------
// GPU kernel: scan одного блока (Blelloch)
// ------------------------------------------------------------
__global__ void scan_block_kernel(const float* in,
                                  float* out,
                                  float* blockSums,
                                  int n)
{
    __shared__ float shmem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Загружаем данные в shared memory
    shmem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // -------- upsweep (reduce) --------
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            shmem[idx] += shmem[idx - stride];
        }
        __syncthreads();
    }

    // Сохраняем сумму блока
    if (tid == 0) {
        blockSums[blockIdx.x] = shmem[blockDim.x - 1];
        shmem[blockDim.x - 1] = 0.0f;   // подготовка к downsweep
    }
    __syncthreads();

    // -------- downsweep --------
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            float tmp = shmem[idx - stride];
            shmem[idx - stride] = shmem[idx];
            shmem[idx] += tmp;
        }
        __syncthreads();
    }

    // Inclusive scan = exclusive + исходное значение
    if (gid < n) {
        out[gid] = shmem[tid] + in[gid];
    }
}

// ------------------------------------------------------------
// Добавление смещения (offset) для каждого блока
// ------------------------------------------------------------
__global__ void add_offsets(float* data,
                            const float* offsets,
                            int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x == 0 || gid >= n) return;

    data[gid] += offsets[blockIdx.x - 1];
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main() {
    std::vector<float> h_in(N), h_cpu(N), h_gpu(N);

    srand(1);
    for (int i = 0; i < N; i++) {
        h_in[i] = float((rand() % 10) + 1);
    }

    // ---------------- CPU ----------------
    clock_t c0 = clock();
    cpu_scan(h_in.data(), h_cpu.data(), N);
    clock_t c1 = clock();
    double cpu_ms = double(c1 - c0) / CLOCKS_PER_SEC * 1000.0;

    // ---------------- GPU ----------------
    float *d_in, *d_out, *d_blockSums;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)), "malloc d_in");
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)), "malloc d_out");
    cudaCheck(cudaMalloc(&d_blockSums, blocks * sizeof(float)), "malloc d_blockSums");

    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 1. scan блоков
    scan_block_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, N);
    cudaCheck(cudaGetLastError(), "scan_block_kernel");

    // 2. scan block sums на CPU (их мало)
    std::vector<float> h_blockSums(blocks), h_blockScan(blocks);
    cudaCheck(cudaMemcpy(h_blockSums.data(), d_blockSums,
                          blocks * sizeof(float), cudaMemcpyDeviceToHost), "D2H block sums");

    if (blocks > 0) {
        cpu_scan(h_blockSums.data(), h_blockScan.data(), blocks);
        cudaCheck(cudaMemcpy(d_blockSums, h_blockScan.data(),
                              blocks * sizeof(float), cudaMemcpyHostToDevice), "H2D block scan");
    }

    // 3. добавляем офсеты
    add_offsets<<<blocks, BLOCK_SIZE>>>(d_out, d_blockSums, N);
    cudaCheck(cudaGetLastError(), "add_offsets");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaCheck(cudaMemcpy(h_gpu.data(), d_out,
                          N * sizeof(float), cudaMemcpyDeviceToHost), "D2H result");

    // ---------------- Анализ ----------------
    float diff = fabs(h_cpu[N - 1] - h_gpu[N - 1]);

    printf("CPU time: %.4f ms\n", cpu_ms);
    printf("GPU time: %.4f ms\n", gpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);
    printf("Final sum CPU: %.2f\n", h_cpu[N - 1]);
    printf("Final sum GPU: %.2f\n", h_gpu[N - 1]);
    printf("Difference: %.6f\n", diff);

    // Очистка
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockSums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}