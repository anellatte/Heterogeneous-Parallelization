#include <stdio.h>              // Для printf
#include <stdlib.h>             // Для malloc, free, rand
#include <cuda_runtime.h>       // CUDA Runtime API
#include <math.h>               // Для fabs
#include <time.h>               // Для clock(), srand()

// -------------------- Константы --------------------
#define N_ELEMENTS 100000       // Общее количество элементов массива
#define BLOCK_SIZE 256          // Количество потоков в одном блоке CUDA

// -------------------- CUDA kernel --------------------
/*
    Этот kernel считает частичную сумму массива.
    Каждый блок считает свою сумму и сохраняет её в массив block_sums.
*/
__global__ void gpu_partial_sum(const float* data,
                                float* block_sums,
                                int n)
{
    // Shared memory для редукции внутри блока
    __shared__ float local_sum[BLOCK_SIZE];

    // Локальный индекс потока внутри блока
    int local_tid = threadIdx.x;

    // Глобальный индекс первого элемента для потока
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток будет суммировать несколько элементов
    float temp = 0.0f;

    // Grid-stride loop:
    // один поток обрабатывает несколько элементов массива
    for (int idx = global_tid; idx < n; idx += gridDim.x * blockDim.x) {
        temp += data[idx];
    }

    // Сохраняем частичную сумму потока в shared memory
    local_sum[local_tid] = temp;

    // Ждём, пока все потоки запишут данные
    __syncthreads();

    // Параллельная редукция внутри блока
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (local_tid < step) {
            local_sum[local_tid] += local_sum[local_tid + step];
        }
        __syncthreads();
    }

    // Первый поток блока записывает сумму блока
    if (local_tid == 0) {
        block_sums[blockIdx.x] = local_sum[0];
    }
}

// -------------------- CPU функция суммирования --------------------
float cpu_sum_array(const float* arr, int n)
{
    float result = 0.0f;

    for (int i = 0; i < n; i++) {
        result += arr[i];
    }

    return result;
}

// -------------------- CPU финальная редукция --------------------
float cpu_reduce_blocks(const float* block_data, int n_blocks)
{
    float total = 0.0f;

    for (int i = 0; i < n_blocks; i++) {
        total += block_data[i];
    }

    return total;
}

// -------------------- main --------------------
int main()
{
    // ---------- Подготовка данных ----------
    int n = N_ELEMENTS;
    size_t bytes = n * sizeof(float);

    float* host_data = (float*)malloc(bytes);

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        host_data[i] = (rand() % 1000) / 100.0f; // значения 0.00 – 9.99
    }

    printf("Суммирование %d элементов\n\n", n);

    // ---------- CPU ----------
    clock_t cpu_begin = clock();
    float cpu_result = cpu_sum_array(host_data, n);
    clock_t cpu_finish = clock();

    double cpu_ms =
        (double)(cpu_finish - cpu_begin) / CLOCKS_PER_SEC * 1000.0;

    printf("CPU:\n");
    printf("  Сумма = %.3f\n", cpu_result);
    printf("  Время = %.4f мс\n\n", cpu_ms);

    // ---------- GPU ----------
    float *device_data, *device_block_sums;

    int blocks =
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t block_bytes = blocks * sizeof(float);

    float* host_block_sums =
        (float*)malloc(block_bytes);

    // Выделение памяти на GPU
    cudaMalloc(&device_data, bytes);
    cudaMalloc(&device_block_sums, block_bytes);

    // Копирование данных на GPU
    cudaMemcpy(device_data,
               host_data,
               bytes,
               cudaMemcpyHostToDevice);

    // CUDA события для замера времени
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);

    // Запуск kernel
    gpu_partial_sum<<<blocks, BLOCK_SIZE>>>(
        device_data,
        device_block_sums,
        n
    );

    cudaEventRecord(ev_stop);

    // Копирование результатов блоков обратно на CPU
    cudaMemcpy(host_block_sums,
               device_block_sums,
               block_bytes,
               cudaMemcpyDeviceToHost);

    // Финальная редукция на CPU
    float gpu_result =
        cpu_reduce_blocks(host_block_sums, blocks);

    cudaEventSynchronize(ev_stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

    printf("GPU:\n");
    printf("  Сумма = %.3f\n", gpu_result);
    printf("  Время = %.4f мс\n", gpu_ms);
    printf("  Блоков = %d\n", blocks);
    printf("  Потоков в блоке = %d\n\n", BLOCK_SIZE);

    // ---------- Сравнение ----------
    float error = fabs(cpu_result - gpu_result);

    printf("Анализ:\n");
    printf("  Разница = %.6f\n", error);
    printf("  Ускорение = %.2fx\n", cpu_ms / gpu_ms);

    if (error < 0.01f) {
        printf("  ✓ Результаты совпадают\n");
    } else {
        printf("  ✗ Есть расхождение (округление)\n");
    }

    // ---------- Очистка памяти ----------
    free(host_data);
    free(host_block_sums);
    cudaFree(device_data);
    cudaFree(device_block_sums);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
