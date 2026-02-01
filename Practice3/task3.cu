#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// ------------------------------------------------------------
// ЯДРО 1: восстановление свойства кучи (heapify)
// Каждый поток работает с одним узлом
// ------------------------------------------------------------
__global__ void heapifyKernel(int* data, int n, int i) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток проверяет один элемент
    int idx = i - tid;
    if (idx < 0) return;

    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < n && data[left] > data[largest])
        largest = left;

    if (right < n && data[right] > data[largest])
        largest = right;

    if (largest != idx) {
        int tmp = data[idx];
        data[idx] = data[largest];
        data[largest] = tmp;
    }
}

// ------------------------------------------------------------
// ПОСТРОЕНИЕ КУЧИ (bottom-up)
// ------------------------------------------------------------
void buildHeapGPU(int* d_data, int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        int threads = 256;
        int blocks = (i + threads) / threads;
        heapifyKernel<<<blocks, threads>>>(d_data, n, i);
        cudaDeviceSynchronize();
    }
}

// ------------------------------------------------------------
// ОСНОВНАЯ ПИРАМИДАЛЬНАЯ СОРТИРОВКА НА GPU
// ------------------------------------------------------------
void heapSortGPU(int* d_data, int n) {

    // 1. Построение максимальной кучи
    buildHeapGPU(d_data, n);

    // 2. Последовательное извлечение максимума
    for (int i = n - 1; i > 0; i--) {

        // Меняем корень (максимум) с последним элементом
        cudaMemcpy(d_data + i, d_data, sizeof(int),
                   cudaMemcpyDeviceToDevice);

        // Восстанавливаем кучу на уменьшенном массиве
        int threads = 256;
        int blocks = (i + threads) / threads;
        heapifyKernel<<<blocks, threads>>>(d_data, i, 0);
        cudaDeviceSynchronize();
    }
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main() {
    const int N = 100000;

    // -------------------------------
    // Генерация данных на CPU
    // -------------------------------
    std::vector<int> h(N);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (int& x : h) x = dist(gen);

    // Эталон CPU
    std::vector<int> ref = h;
    std::sort(ref.begin(), ref.end());

    // -------------------------------
    // Копирование данных на GPU
    // -------------------------------
    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h.data(), N * sizeof(int),
               cudaMemcpyHostToDevice);

    // -------------------------------
    // Замер времени
    // -------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    heapSortGPU(d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // -------------------------------
    // Копирование результата
    // -------------------------------
    cudaMemcpy(h.data(), d, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    bool ok = (h == ref);

    std::cout << "GPU Heap Sort time: " << ms << " ms\n";
    std::cout << "Verification: "
              << (ok ? "PASSED" : "FAILED") << std::endl;

    cudaFree(d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
