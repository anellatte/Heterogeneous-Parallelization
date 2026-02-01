#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// ------------------------------------------------------------
// ЯДРО 1: параллельное разбиение массива по опорному элементу
// ------------------------------------------------------------
__global__ void partitionKernel(
    int* data, int* left, int* right,
    int n, int pivot)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int value = data[idx];

    // Каждый поток решает, куда пойдёт его элемент
    if (value < pivot)
        left[idx] = value;
    else
        right[idx] = value;
}

// ------------------------------------------------------------
// ЯДРО 2: локальная быстрая сортировка внутри блока
// (упрощённая, последовательная внутри блока)
// ------------------------------------------------------------
__device__ void quicksortLocal(int* arr, int left, int right) {
    if (left >= right) return;

    int i = left;
    int j = right;
    int pivot = arr[(left + right) / 2];

    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++; j--;
        }
    }

    quicksortLocal(arr, left, j);
    quicksortLocal(arr, i, right);
}

// ------------------------------------------------------------
// ЯДРО 3: каждый блок сортирует свою часть массива
// ------------------------------------------------------------
__global__ void blockQuickSort(int* data, int n, int chunk) {
    int blockId = blockIdx.x;
    int start = blockId * chunk;
    int end = min(start + chunk - 1, n - 1);

    if (start < n)
        quicksortLocal(data, start, end);
}

// ------------------------------------------------------------
// ОСНОВНАЯ ПРОГРАММА
// ------------------------------------------------------------
int main() {
    const int N = 100000;
    const int BLOCK_SIZE = 256;
    const int CHUNK_SIZE = 1024;

    // -------------------------------
    // Генерация данных на CPU
    // -------------------------------
    std::vector<int> h_data(N);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 100000);

    for (int& x : h_data)
        x = dist(gen);

    // Эталонная сортировка CPU
    std::vector<int> reference = h_data;
    std::sort(reference.begin(), reference.end());

    // -------------------------------
    // Выделение памяти на GPU
    // -------------------------------
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data.data(),
               N * sizeof(int),
               cudaMemcpyHostToDevice);

    // -------------------------------
    // Выбор опорного элемента
    // -------------------------------
    int pivot = h_data[N / 2];

    // -------------------------------
    // Параллельное разбиение
    // -------------------------------
    int* d_left;
    int* d_right;
    cudaMalloc(&d_left, N * sizeof(int));
    cudaMalloc(&d_right, N * sizeof(int));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    partitionKernel<<<blocks, BLOCK_SIZE>>>(
        d_data, d_left, d_right, N, pivot);
    cudaDeviceSynchronize();

    // -------------------------------
    // Локальная сортировка частей
    // -------------------------------
    int numBlocks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
    blockQuickSort<<<numBlocks, 1>>>(d_data, N, CHUNK_SIZE);
    cudaDeviceSynchronize();

    // -------------------------------
    // Копирование результата
    // -------------------------------
    cudaMemcpy(h_data.data(), d_data,
               N * sizeof(int),
               cudaMemcpyDeviceToHost);

    bool ok = (h_data == reference);

    std::cout << "Parallel Quick Sort (CUDA)\n";
    std::cout << "Verification: "
              << (ok ? "PASSED" : "FAILED") << std::endl;

    cudaFree(d_data);
    cudaFree(d_left);
    cudaFree(d_right);

    return 0;
}
