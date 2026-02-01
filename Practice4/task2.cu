#include <cstdio>            // printf
#include <cstdlib>           // malloc, free, rand, srand
#include <vector>            // std::vector
#include <cuda_runtime.h>    // CUDA Runtime API
#include <ctime>             // time()

#define N 1000000            // Размер массива
#define BLOCK_SIZE 256       // Количество потоков в блоке

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
// GPU таймер на CUDA events
// ------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() {
        cudaEventRecord(start);
    }

    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ------------------------------------------------------------
// (a) Редукция ТОЛЬКО через глобальную память
// Каждый поток делает atomicAdd в global memory
// ------------------------------------------------------------
__global__ void reduce_global_atomic(const float* data,
                                     float* result,
                                     int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток добавляет один элемент
    if (idx < n) {
        atomicAdd(result, data[idx]);
    }
}

// ------------------------------------------------------------
// (b) Редукция с использованием shared memory
// Сначала суммируем внутри блока,
// затем ОДИН atomicAdd на блок
// ------------------------------------------------------------
__global__ void reduce_shared(const float* data,
                              float* result,
                              int n)
{
    // Разделяемая память блока
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Загружаем данные в shared memory
    // Если вышли за границы массива — кладём 0
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    // Параллельная редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Один поток блока добавляет сумму блока в global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main() {
    // ---------------- Подготовка данных ----------------
    std::vector<float> h_data(N);

    srand((unsigned)time(nullptr));
    for (int i = 0; i < N; i++) {
        h_data[i] = float((rand() % 10) + 1);
    }

    float *d_data = nullptr;
    float *d_result = nullptr;

    cudaCheck(cudaMalloc(&d_data, N * sizeof(float)), "cudaMalloc d_data");
    cudaCheck(cudaMalloc(&d_result, sizeof(float)), "cudaMalloc d_result");

    cudaCheck(cudaMemcpy(d_data,
                          h_data.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice),
              "memcpy H2D");

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ---------------- Вариант (a): global memory ----------------
    cudaCheck(cudaMemset(d_result, 0, sizeof(float)), "memset result");

    GpuTimer timer_global;
    timer_global.tic();

    reduce_global_atomic<<<blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    cudaCheck(cudaDeviceSynchronize(), "sync global");

    float time_global = timer_global.toc();

    float sum_global = 0.0f;
    cudaCheck(cudaMemcpy(&sum_global,
                          d_result,
                          sizeof(float),
                          cudaMemcpyDeviceToHost),
              "memcpy result global");

    // ---------------- Вариант (b): shared memory ----------------
    cudaCheck(cudaMemset(d_result, 0, sizeof(float)), "memset result");

    GpuTimer timer_shared;
    timer_shared.tic();

    reduce_shared<<<blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    cudaCheck(cudaDeviceSynchronize(), "sync shared");

    float time_shared = timer_shared.toc();

    float sum_shared = 0.0f;
    cudaCheck(cudaMemcpy(&sum_shared,
                          d_result,
                          sizeof(float),
                          cudaMemcpyDeviceToHost),
              "memcpy result shared");

    // ---------------- Вывод результатов ----------------
    printf("=============================================\n");
    printf("Редукция суммы элементов массива (CUDA)\n");
    printf("Размер массива: %d\n", N);
    printf("Потоков в блоке: %d\n", BLOCK_SIZE);
    printf("=============================================\n\n");

    printf("Вариант (a): Только global memory\n");
    printf("  Время: %.4f мс\n", time_global);
    printf("  Сумма: %.2f\n\n", sum_global);

    printf("Вариант (b): Global + shared memory\n");
    printf("  Время: %.4f мс\n", time_shared);
    printf("  Сумма: %.2f\n\n", sum_shared);

    printf("Ускорение (global / shared): %.2fx\n",
           time_global / time_shared);

    // ---------------- Очистка памяти ----------------
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}