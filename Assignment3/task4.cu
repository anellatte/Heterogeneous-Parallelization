%%writefile task4.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <iomanip>

// Проверка ошибок CUDA
#define CUDA_SAFE(x) do {                                \
    cudaError_t err = (x);                               \
    if (err != cudaSuccess) {                            \
        std::cerr << "CUDA error: "                      \
                  << cudaGetErrorString(err)             \
                  << std::endl;                          \
        exit(1);                                         \
    }                                                    \
} while (0)

// CUDA-ядро: сложение двух массивов
__global__ void add_kernel(const float* a,
                           const float* b,
                           float* c,
                           int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

// Функция замера времени для заданного block size
float test_config(const float* d_a,
                  const float* d_b,
                  float* d_c,
                  int n,
                  int block_size,
                  int warmup,
                  int runs)
{
    int grid_size = (n + block_size - 1) / block_size;

    // Прогрев
    for (int i = 0; i < warmup; i++) {
        add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CUDA_SAFE(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_SAFE(cudaEventCreate(&start));
    CUDA_SAFE(cudaEventCreate(&stop));

    CUDA_SAFE(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CUDA_SAFE(cudaEventRecord(stop));
    CUDA_SAFE(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_SAFE(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / runs;
}

// main
int main()
{
    const int N = 1'000'000;
    const size_t BYTES = N * sizeof(float);

    const int WARMUP = 5;
    const int RUNS   = 50;

    // Размеры блока для перебора
    std::vector<int> block_variants = {64, 128, 256, 512};

    // Неоптимальный вариант (фиксируем)
    const int BAD_BLOCK = 64;

    // Хост-данные
    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.1f;
        h_b[i] = i * 0.2f;
    }

    // GPU-память
    float *d_a, *d_b, *d_c;
    CUDA_SAFE(cudaMalloc(&d_a, BYTES));
    CUDA_SAFE(cudaMalloc(&d_b, BYTES));
    CUDA_SAFE(cudaMalloc(&d_c, BYTES));

    CUDA_SAFE(cudaMemcpy(d_a, h_a.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(d_b, h_b.data(), BYTES, cudaMemcpyHostToDevice));

    std::cout << "=== Подбор оптимальной конфигурации ===\n";
    std::cout << "Array size: " << N << "\n\n";

    std::cout << std::left
              << std::setw(10) << "Block"
              << std::setw(12) << "Grid"
              << std::setw(14) << "Time (ms)"
              << "\n";
    std::cout << std::string(36, '-') << "\n";

    float best_time = std::numeric_limits<float>::max();
    int best_block = -1;

    for (int block : block_variants) {
        int grid = (N + block - 1) / block;

        CUDA_SAFE(cudaMemset(d_c, 0, BYTES));
        float t = test_config(d_a, d_b, d_c, N, block, WARMUP, RUNS);

        std::cout << std::left
                  << std::setw(10) << block
                  << std::setw(12) << grid
                  << std::setw(14) << std::fixed << std::setprecision(4) << t
                  << "\n";

        if (t < best_time) {
            best_time = t;
            best_block = block;
        }
    }

    int best_grid = (N + best_block - 1) / best_block;
    int bad_grid  = (N + BAD_BLOCK  - 1) / BAD_BLOCK;

    float bad_time = test_config(d_a, d_b, d_c, N, BAD_BLOCK, WARMUP, RUNS);
    float good_time = test_config(d_a, d_b, d_c, N, best_block, WARMUP, RUNS);

    std::cout << "\n=== Сравнение ===\n";
    std::cout << "Неоптимальная: block = " << BAD_BLOCK
              << ", grid = " << bad_grid
              << ", time = " << bad_time << " ms\n";

    std::cout << "Оптимальная:   block = " << best_block
              << ", grid = " << best_grid
              << ", time = " << good_time << " ms\n";

    std::cout << "Ускорение: "
              << bad_time / good_time << "x\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}