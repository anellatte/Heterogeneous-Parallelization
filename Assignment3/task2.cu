%%writefile task2.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Макрос проверки ошибок CUDA (простой и надёжный)
#define CHECK(call) do {                               \
    cudaError_t err = call;                            \
    if (err != cudaSuccess) {                          \
        std::cerr << "CUDA error: "                    \
                  << cudaGetErrorString(err)           \
                  << std::endl;                        \
        exit(1);                                       \
    }                                                  \
} while (0)

// CUDA-ядро: поэлементное сложение массивов
__global__ void add_arrays(const float* a,
                           const float* b,
                           float* c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Функция измерения времени ядра
float run_test(int block_size,
               const float* d_a,
               const float* d_b,
               float* d_c,
               int n)
{
    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Прогрев GPU
    for (int i = 0; i < 10; i++) {
        add_arrays<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CHECK(cudaDeviceSynchronize());

    // Замер
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        add_arrays<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 100.0f; // среднее время одного запуска
}

// main
int main()
{
    const int N = 1'000'000;
    const size_t BYTES = N * sizeof(float);

    // Размеры блоков для эксперимента (минимум 3)
    std::vector<int> blocks = {128, 256, 512};

    // Хост-массивы
    std::vector<float> h_a(N), h_b(N), h_c(N);

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 1.5f;
    }

    // Память на GPU
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, BYTES));
    CHECK(cudaMalloc(&d_b, BYTES));
    CHECK(cudaMalloc(&d_c, BYTES));

    CHECK(cudaMemcpy(d_a, h_a.data(), BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), BYTES, cudaMemcpyHostToDevice));

    std::cout << "Array size: " << N << "\n";
    std::cout << "Block size | Avg kernel time (ms)\n";
    std::cout << "---------------------------------\n";

    for (int bs : blocks) {
        float time_ms = run_test(bs, d_a, d_b, d_c, N);
        std::cout << bs << "\t   | " << time_ms << "\n";
    }

    // Проверка корректности
    CHECK(cudaMemcpy(h_c.data(), d_c, BYTES, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < 10; i++) {
        if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-6) {
            ok = false;
        }
    }

    std::cout << "\nCheck: " << (ok ? "OK" : "ERROR") << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
