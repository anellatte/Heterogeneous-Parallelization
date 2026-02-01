%%writefile task1.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>


// Проверка ошибок CUDA
#define CHECK_CUDA(x) do {                             \
    cudaError_t err = (x);                             \
    if (err != cudaSuccess) {                          \
        std::cerr << "CUDA error: "                    \
                  << cudaGetErrorString(err)           \
                  << std::endl;                        \
        exit(1);                                       \
    }                                                  \
} while (0)


// ЯДРО 1 — работа ТОЛЬКО с глобальной памятью
__global__ void kernel_global(const float* input,
                              float* output,
                              float factor,
                              int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < size) {
        output[pos] = input[pos] * factor;
    }
}

// ЯДРО 2 — с использованием shared memory
__global__ void kernel_shared(const float* input,
                              float* output,
                              float factor,
                              int size)
{
    // shared-память на блок
    extern __shared__ float buffer[];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Загружаем данные из global → shared
    if (pos < size) {
        buffer[local_id] = input[pos];
    }

    __syncthreads(); // ждём все потоки блока

    // Умножаем в shared
    if (pos < size) {
        buffer[local_id] *= factor;
    }

    __syncthreads();

    // Записываем результат обратно в global
    if (pos < size) {
        output[pos] = buffer[local_id];
    }
}

// Функция измерения времени ядра
float measure_time(void (*kernel)(const float*, float*, float, int),
                   dim3 grid, dim3 block,
                   size_t shared_bytes,
                   const float* d_in,
                   float* d_out,
                   float factor,
                   int size)
{
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    // Прогрев GPU
    for (int i = 0; i < 20; i++) {
        kernel<<<grid, block, shared_bytes>>>(d_in, d_out, factor, size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Замер
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 500; i++) {
        kernel<<<grid, block, shared_bytes>>>(d_in, d_out, factor, size);
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, end));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    return ms / 500.0f; // среднее время
}

// MAIN
int main()
{
    const int SIZE = 1'000'000;
    const float MULT = 3.0f;

    std::vector<float> host_input(SIZE);
    std::vector<float> host_output1(SIZE);
    std::vector<float> host_output2(SIZE);

    // Инициализация
    for (int i = 0; i < SIZE; i++) {
        host_input[i] = i * 0.01f;
    }

    float* dev_input;
    float* dev_output;
    CHECK_CUDA(cudaMalloc(&dev_input, SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_output, SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dev_input,
                           host_input.data(),
                           SIZE * sizeof(float),
                           cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (SIZE + threads - 1) / threads;

    // ---------- GLOBAL ----------
    float time_global = measure_time(
        kernel_global,
        blocks,
        threads,
        0,
        dev_input,
        dev_output,
        MULT,
        SIZE
    );

    CHECK_CUDA(cudaMemcpy(host_output1.data(),
                           dev_output,
                           SIZE * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // ---------- SHARED ----------
    float time_shared = measure_time(
        kernel_shared,
        blocks,
        threads,
        threads * sizeof(float),
        dev_input,
        dev_output,
        MULT,
        SIZE
    );

    CHECK_CUDA(cudaMemcpy(host_output2.data(),
                           dev_output,
                           SIZE * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Проверка
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        float ref = host_input[i] * MULT;
        if (std::fabs(host_output1[i] - ref) > 1e-6 ||
            std::fabs(host_output2[i] - ref) > 1e-6) {
            correct = false;
        }
    }

    std::cout << "Array size: " << SIZE << "\n";
    std::cout << "Threads per block: " << threads << "\n\n";

    std::cout << "Global memory time: " << time_global << " ms\n";
    std::cout << "Shared memory time: " << time_shared << " ms\n";
    std::cout << "Speedup (global/shared): "
              << time_global / time_shared << "x\n";

    std::cout << "Check: " << (correct ? "OK" : "ERROR") << std::endl;

    cudaFree(dev_input);
    cudaFree(dev_output);
    return 0;
}