#include <cuda_runtime.h>   // CUDA Runtime API
#include <iostream>         // std::cout
#include <vector>           // std::vector
#include <random>           // генерация случайных чисел
#include <iomanip>          // форматированный вывод
#include <cstdlib>          // exit()

// ------------------------------------------------------------
// Макрос проверки ошибок CUDA
// ------------------------------------------------------------
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA ошибка: "                               \
                  << cudaGetErrorString(err)                       \
                  << " в строке " << __LINE__ << std::endl;       \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// ------------------------------------------------------------
// Таймер на GPU (через cudaEvent)
// ------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() {
        CUDA_CHECK(cudaEventRecord(start));
    }

    float toc() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ------------------------------------------------------------
// Вариант A: редукция ТОЛЬКО через глобальную память
// Каждый поток делает atomicAdd в global memory
// ------------------------------------------------------------
__global__ void reduce_global_atomic(const float* a, int n, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(out, a[idx]);
    }
}

// ------------------------------------------------------------
// Вариант B: редукция с использованием shared memory
// Сначала суммирование внутри блока, затем один atomicAdd
// ------------------------------------------------------------
__global__ void reduce_shared(const float* a, int n, float* out) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Загружаем данные в shared память
    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();

    // Параллельная редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Один поток добавляет сумму блока в глобальную память
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// ------------------------------------------------------------
// Генерация входного массива случайных чисел
// ------------------------------------------------------------
std::vector<float> generate_data(int n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    std::vector<float> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
    return data;
}

// ------------------------------------------------------------
// Один запуск редукции (возвращает время в мс)
// ------------------------------------------------------------
float run_reduce(const std::vector<float>& h,
                 bool use_shared,
                 int block_size)
{
    int n = (int)h.size();

    float *d_in = nullptr, *d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h.data(),
                           n * sizeof(float),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    int grid = (n + block_size - 1) / block_size;

    GpuTimer timer;
    timer.tic();

    if (use_shared) {
        size_t shmem = block_size * sizeof(float);
        reduce_shared<<<grid, block_size, shmem>>>(d_in, n, d_out);
    } else {
        reduce_global_atomic<<<grid, block_size>>>(d_in, n, d_out);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = timer.toc();

    cudaFree(d_in);
    cudaFree(d_out);

    return ms;
}

// ------------------------------------------------------------
// Точка входа
// ------------------------------------------------------------
int main() {
    std::cout << "=============================================\n";
    std::cout << "Задание 4: Измерение производительности CUDA\n";
    std::cout << "=============================================\n\n";

    std::vector<int> sizes = {10000, 100000, 1000000};
    const int block_size = 256;
    const int iters = 20;

    std::cout << std::left
              << std::setw(12) << "Размер n"
              << std::setw(18) << "Global (мс)"
              << std::setw(18) << "Shared (мс)"
              << std::setw(12) << "Ускорение"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int n : sizes) {
        auto h = generate_data(n);

        float t_global = 0.0f;
        float t_shared = 0.0f;

        // Усреднение времени
        for (int i = 0; i < iters; i++) {
            t_global += run_reduce(h, false, block_size);
            t_shared += run_reduce(h, true, block_size);
        }

        t_global /= iters;
        t_shared /= iters;

        float speedup = t_global / t_shared;

        std::cout << std::left
                  << std::setw(12) << n
                  << std::setw(18) << std::fixed << std::setprecision(4) << t_global
                  << std::setw(18) << std::fixed << std::setprecision(4) << t_shared
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << "\n";
    }

    return 0;
}