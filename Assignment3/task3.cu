%%writefile task3.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Макрос проверки ошибок CUDA
#define CHECK_CUDA(x) do {                               \
    cudaError_t err = (x);                               \
    if (err != cudaSuccess) {                            \
        std::cerr << "CUDA error: "                      \
                  << cudaGetErrorString(err)             \
                  << std::endl;                          \
        exit(1);                                         \
    }                                                    \
} while (0)

// Coalesced: потоки работают с соседними элементами
__global__ void kernel_linear(const float* in,
                              float* out,
                              int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f + 1.0f;
    }
}

// Non-coalesced: искусственно "ломаем" доступ
__global__ void kernel_strided(const float* in,
                               float* out,
                               int n,
                               int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int idx = (tid * stride) % n;
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

// Универсальная функция замера времени
float measure_kernel(void (*kernel)(const float*, float*, int),
                     const float* d_in,
                     float* d_out,
                     int n,
                     int block_size)
{
    int grid = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Прогрев
    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block_size>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Замер
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 200; i++) {
        kernel<<<grid, block_size>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 200.0f;
}

// Отдельный замер для non-coalesced ядра
float measure_kernel_strided(const float* d_in,
                             float* d_out,
                             int n,
                             int block_size,
                             int stride)
{
    int grid = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 10; i++) {
        kernel_strided<<<grid, block_size>>>(d_in, d_out, n, stride);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 200; i++) {
        kernel_strided<<<grid, block_size>>>(d_in, d_out, n, stride);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 200.0f;
}

// main
int main()
{
    const int N = 1'000'000;
    const size_t BYTES = N * sizeof(float);
    const int BLOCK = 256;

    // Для некоалесцированного доступа
    const int STRIDE = 999983; // простое число, почти N

    std::vector<float> h_in(N), h_out(N);
    for (int i = 0; i < N; i++) {
        h_in[i] = i * 0.01f;
    }

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, BYTES));
    CHECK_CUDA(cudaMalloc(&d_out, BYTES));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice));

    std::cout << "Array size: " << N << "\n";
    std::cout << "Block size: " << BLOCK << "\n\n";

    float t_coalesced = measure_kernel(
        kernel_linear, d_in, d_out, N, BLOCK
    );

    float t_noncoalesced = measure_kernel_strided(
        d_in, d_out, N, BLOCK, STRIDE
    );

    std::cout << "Coalesced access time:     "
              << t_coalesced << " ms\n";
    std::cout << "Non-coalesced access time: "
              << t_noncoalesced << " ms\n";
    std::cout << "Slowdown: "
              << t_noncoalesced / t_coalesced << "x\n";

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost));

    std::cout << "\nSample output values:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "out[" << i << "] = " << h_out[i] << "\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
