#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>

#define N 1000000          // Размер массива
#define BLOCK_SIZE 256     // Потоков в блоке
#define K 8                // Сколько элементов сортирует ОДИН поток
#define INF 1e30f          // "Бесконечность" для выхода за границы

// ------------------------------------------------------------
// Макрос проверки ошибок CUDA
// ------------------------------------------------------------
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: "                                \
                  << cudaGetErrorString(err)                       \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                   \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// ------------------------------------------------------------
// Локальная пузырьковая сортировка K элементов
// Работает ВНУТРИ ОДНОГО ПОТОКА (регистры)
// ------------------------------------------------------------
__device__ void bubble_sort_local(float v[K]) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K - 1 - i; j++) {
            if (v[j] > v[j + 1]) {
                float t = v[j];
                v[j] = v[j + 1];
                v[j + 1] = t;
            }
        }
    }
}

// ------------------------------------------------------------
// ЭТАП 1:
// Каждый поток:
//  - читает K элементов из global memory
//  - сортирует их пузырьком (локально)
//  - записывает результат в shared memory
// Далее:
//  - выполняется поэтапное слияние внутри блока (shared memory)
// ------------------------------------------------------------
__global__ void sort_tiles_and_merge(float* data, int n) {
    extern __shared__ float sh[];       // Shared memory
    int tid = threadIdx.x;
    int block = blockIdx.x;

    const int tile_size = BLOCK_SIZE * K;
    int tile_base = block * tile_size;

    float local[K];                     // Локальный массив потока

    // -------- Чтение данных из global memory --------
    int base = tile_base + tid * K;
    for (int i = 0; i < K; i++) {
        int idx = base + i;
        local[i] = (idx < n) ? data[idx] : INF;
    }

    // -------- Локальная сортировка --------
    bubble_sort_local(local);

    // -------- Запись в shared memory --------
    for (int i = 0; i < K; i++) {
        sh[tid * K + i] = local[i];
    }
    __syncthreads();

    // -------- Слияние в shared memory --------
    for (int run = K; run < tile_size; run *= 2) {
        for (int start = tid * (2 * run);
             start < tile_size;
             start += BLOCK_SIZE * (2 * run)) {

            int mid = start + run;
            int end = min(start + 2 * run, tile_size);

            int i = start;
            int j = mid;
            int t = start;

            while (i < mid && j < end) {
                sh[t++] = (sh[i] <= sh[j]) ? sh[i++] : sh[j++];
            }
            while (i < mid) sh[t++] = sh[i++];
            while (j < end) sh[t++] = sh[j++];
        }
        __syncthreads();
    }

    // -------- Запись результата обратно в global memory --------
    for (int i = tid; i < tile_size; i += BLOCK_SIZE) {
        int g = tile_base + i;
        if (g < n) data[g] = sh[i];
    }
}

// ------------------------------------------------------------
// Проверка на CPU: массив отсортирован?
// ------------------------------------------------------------
bool is_sorted_cpu(const std::vector<float>& v) {
    for (size_t i = 1; i < v.size(); i++) {
        if (v[i - 1] > v[i]) return false;
    }
    return true;
}

// ------------------------------------------------------------
// Генерация случайных данных
// ------------------------------------------------------------
std::vector<float> generate_data(int n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<float> a(n);
    for (int i = 0; i < n; i++) a[i] = dist(rng);
    return a;
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main() {
    std::cout << "=============================================\n";
    std::cout << "Задание 3: Оптимизация сортировки на GPU\n";
    std::cout << "=============================================\n\n";

    // -------- Генерация данных --------
    auto h = generate_data(N);

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h.data(),
                           N * sizeof(float),
                           cudaMemcpyHostToDevice));

    int tile_size = BLOCK_SIZE * K;
    int grid = (N + tile_size - 1) / tile_size;

    size_t shared_bytes = tile_size * sizeof(float);

    // -------- Запуск ядра --------
    sort_tiles_and_merge<<<grid, BLOCK_SIZE, shared_bytes>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------- Копирование результата --------
    CUDA_CHECK(cudaMemcpy(h.data(),
                           d_data,
                           N * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // -------- Проверка --------
    std::cout << "Проверка сортировки (CPU): "
              << (is_sorted_cpu(h) ? "ДА" : "НЕТ") << "\n";

    cudaFree(d_data);

    std::cout << "Готово.\n";
    return 0;
}