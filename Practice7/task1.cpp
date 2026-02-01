#include <cuda_runtime.h>     // CUDA Runtime API
#include <iostream>           // std::cout
#include <vector>             // std::vector
#include <cmath>              // std::fabs
#include <iomanip>            // std::setprecision
#include <cstdlib>            // std::exit

// ------------------------------------------------------------
// Макрос проверки ошибок CUDA
// ------------------------------------------------------------
#define CUDA_SAFE(call)                                           \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error: "                           \
                      << cudaGetErrorString(err)                  \
                      << " (" << __FILE__ << ":"                  \
                      << __LINE__ << ")\n";                       \
            std::exit(1);                                         \
        }                                                         \
    } while (0)

// ------------------------------------------------------------
// CUDA-ядро редукции суммы
// Каждый блок считает частичную сумму
// ------------------------------------------------------------
__global__ void reduce_kernel(const float* input,
                              float* partial,
                              int n)
{
    // Разделяемая память для текущего блока
    extern __shared__ float shmem[];

    int tid = threadIdx.x;                       // ID потока в блоке
    int gid = blockIdx.x * blockDim.x + tid;     // Глобальный индекс

    // Загружаем данные в shared memory
    if (gid < n)
        shmem[tid] = input[gid];
    else
        shmem[tid] = 0.0f;

    __syncthreads(); // Ждём, пока все потоки загрузят данные

    // Параллельная редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    // Поток 0 записывает сумму блока
    if (tid == 0) {
        partial[blockIdx.x] = shmem[0];
    }
}

// ------------------------------------------------------------
// Хост-функция: многошаговая редукция на GPU
// ------------------------------------------------------------
float reduce_on_gpu(const std::vector<float>& host_data)
{
    int n = static_cast<int>(host_data.size());

    float* d_input = nullptr;
    float* d_partial = nullptr;

    CUDA_SAFE(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_SAFE(cudaMemcpy(d_input, host_data.data(),
                          n * sizeof(float),
                          cudaMemcpyHostToDevice));

    int current_n = n;
    float* current_input = d_input;

    const int BLOCK = 256;

    // Повторяем редукцию, пока не останется один элемент
    while (current_n > 1) {
        int grid = (current_n + BLOCK - 1) / BLOCK;

        CUDA_SAFE(cudaMalloc(&d_partial, grid * sizeof(float)));

        reduce_kernel<<<grid, BLOCK, BLOCK * sizeof(float)>>>(
            current_input, d_partial, current_n);

        CUDA_SAFE(cudaDeviceSynchronize());

        if (current_input != d_input)
            CUDA_SAFE(cudaFree(current_input));

        current_input = d_partial;
        current_n = grid;
        d_partial = nullptr;
    }

    float result = 0.0f;
    CUDA_SAFE(cudaMemcpy(&result, current_input,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_SAFE(cudaFree(current_input));
    CUDA_SAFE(cudaFree(d_input));

    return result;
}

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
int main()
{
    const int N = 100000;          // Размер тестового массива
    std::vector<float> data(N);   // Массив на CPU

    // Заполняем массив единицами
    for (int i = 0; i < N; ++i)
        data[i] = 1.0f;

    // CPU-эталон
    double cpu_sum = 0.0;
    for (float x : data)
        cpu_sum += x;

    // GPU-редукция
    float gpu_sum = reduce_on_gpu(data);

    // Проверка
    double diff = std::fabs(cpu_sum - gpu_sum);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA Reduction Test\n";
    std::cout << "Array size      : " << N << "\n";
    std::cout << "CPU sum         : " << cpu_sum << "\n";
    std::cout << "GPU sum         : " << gpu_sum << "\n";
    std::cout << "Absolute diff   : " << diff << "\n";

    if (diff < 1e-3)
        std::cout << "Status          : OK\n";
    else
        std::cout << "Status          : ERROR\n";

    return 0;
}
