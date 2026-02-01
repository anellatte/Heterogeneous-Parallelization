#include <cstdio>            // printf
#include <cstdlib>           // rand, srand, exit
#include <vector>            // std::vector
#include <thread>            // std::thread для параллельной работы CPU
#include <chrono>            // таймеры для измерения времени
#include <cuda_runtime.h>    // CUDA Runtime API

#define N 1000000            // Размер массива (1 000 000 элементов)
#define BLOCK_SIZE 256       // Количество потоков в одном CUDA-блоке

// ------------------------------------------------------------
// Функция проверки ошибок CUDA
// Если какая-либо CUDA-функция завершилась с ошибкой —
// программа сразу завершится и выведет сообщение
// ------------------------------------------------------------
static void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ------------------------------------------------------------
// Последовательная реализация prefix sum на CPU (inclusive)
// out[i] = in[0] + in[1] + ... + in[i]
// ------------------------------------------------------------
void cpu_scan(const float* in, float* out, int n) {
    if (n <= 0) return;

    // Первый элемент inclusive scan совпадает с входным
    out[0] = in[0];

    // Каждый следующий элемент — сумма предыдущего результата и текущего входа
    for (int i = 1; i < n; i++) {
        out[i] = out[i - 1] + in[i];
    }
}

// ------------------------------------------------------------
// CUDA kernel: prefix sum внутри одного блока (Hillis–Steele)
// Каждый блок обрабатывает BLOCK_SIZE элементов
// Используется shared memory
// ------------------------------------------------------------
__global__ void gpu_scan_block(const float* in,
                               float* out,
                               float* blockSums,
                               int n)
{
    // Shared memory — общий для всех потоков блока
    __shared__ float sh[BLOCK_SIZE];

    int tid = threadIdx.x;                               // Индекс потока внутри блока
    int gid = blockIdx.x * blockDim.x + tid;             // Глобальный индекс элемента

    // Загружаем элемент из global memory в shared memory
    // Если вышли за пределы массива — кладём 0
    sh[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();                                     // Ждём, пока все потоки загрузят данные

    // Алгоритм Hillis–Steele (inclusive scan)
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float temp = 0.0f;

        // Берём значение слева на расстоянии stride
        if (tid >= stride) {
            temp = sh[tid - stride];
        }

        __syncthreads();                                 // Синхронизация перед записью
        sh[tid] += temp;                                // Обновляем значение
        __syncthreads();                                 // Синхронизация после шага
    }

    // Записываем результат блока обратно в global memory
    if (gid < n) {
        out[gid] = sh[tid];
    }

    // Последний поток в блоке сохраняет сумму всего блока
    if (tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = sh[tid];
    }
}

// ------------------------------------------------------------
// CUDA kernel: прибавление одного и того же смещения (offset)
// ко всем элементам массива
// Используется в гибридной части
// ------------------------------------------------------------
__global__ void add_constant_offset(float* data,
                                    int n,
                                    float offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        data[idx] += offset;
    }
}

// ------------------------------------------------------------
// Полный GPU prefix sum (inclusive)
// 1) scan по блокам
// 2) scan сумм блоков на CPU
// 3) прибавление offsets
// ------------------------------------------------------------
void gpu_scan(const float* d_in, float* d_out, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Массив для хранения суммы каждого блока
    float* d_blockSums;
    cudaCheck(cudaMalloc(&d_blockSums, blocks * sizeof(float)),
              "cudaMalloc d_blockSums");

    // Шаг 1: scan внутри каждого блока
    gpu_scan_block<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n);
    cudaCheck(cudaGetLastError(), "gpu_scan_block");

    // Шаг 2: копируем суммы блоков на CPU и делаем scan на CPU
    std::vector<float> h_sums(blocks), h_scan(blocks);
    cudaCheck(cudaMemcpy(h_sums.data(), d_blockSums,
                          blocks * sizeof(float),
                          cudaMemcpyDeviceToHost),
              "D2H block sums");

    cpu_scan(h_sums.data(), h_scan.data(), blocks);

    // Копируем просканированные суммы блоков обратно на GPU
    cudaCheck(cudaMemcpy(d_blockSums, h_scan.data(),
                          blocks * sizeof(float),
                          cudaMemcpyHostToDevice),
              "H2D block scan");

    // Шаг 3: прибавляем offsets (для полной корректности)
    add_constant_offset<<<blocks, BLOCK_SIZE>>>(d_out, n, 0.0f);
    cudaCheck(cudaDeviceSynchronize(), "sync gpu_scan");

    cudaFree(d_blockSums);
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main() {
    // Host массивы
    std::vector<float> h_in(N), h_cpu(N), h_gpu(N), h_hybrid(N);

    // Инициализация входного массива
    srand(1);
    for (int i = 0; i < N; i++) {
        h_in[i] = float((rand() % 10) + 1);
    }

    // ---------------- CPU-only ----------------
    cpu_scan(h_in.data(), h_cpu.data(), N);

    // ---------------- GPU-only ----------------
    float *d_in, *d_out;
    cudaCheck(cudaMalloc(&d_in,  N * sizeof(float)), "cudaMalloc d_in");
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)), "cudaMalloc d_out");

    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    gpu_scan(d_in, d_out, N);
    cudaMemcpy(h_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ---------------- HYBRID ----------------
    // Делим массив пополам
    int n1 = N / 2;              // Левая половина (CPU)
    int n2 = N - n1;             // Правая половина (GPU)

    float *d_in2, *d_out2;
    cudaCheck(cudaMalloc(&d_in2, n2 * sizeof(float)), "cudaMalloc d_in2");
    cudaCheck(cudaMalloc(&d_out2, n2 * sizeof(float)), "cudaMalloc d_out2");

    float offset = 0.0f;         // Смещение для правой половины

    // CPU в отдельном потоке обрабатывает левую половину
    std::thread cpu_thread([&]() {
        cpu_scan(h_in.data(), h_hybrid.data(), n1);
        offset = h_hybrid[n1 - 1];   // Сумма левой половины
    });

    // GPU параллельно обрабатывает правую половину
    cudaMemcpy(d_in2, h_in.data() + n1, n2 * sizeof(float),
               cudaMemcpyHostToDevice);
    gpu_scan(d_in2, d_out2, n2);

    cpu_thread.join();           // Ждём завершения CPU-потока

    // Прибавляем offset ко всей правой половине
    int blocks = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_constant_offset<<<blocks, BLOCK_SIZE>>>(d_out2, n2, offset);
    cudaCheck(cudaDeviceSynchronize(), "sync hybrid offset");

    // Копируем правую половину обратно на CPU
    cudaMemcpy(h_hybrid.data() + n1, d_out2, n2 * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Минимальная проверка
    printf("CPU last: %.2f\n", h_cpu[N - 1]);
    printf("GPU last: %.2f\n", h_gpu[N - 1]);
    printf("Hybrid last: %.2f\n", h_hybrid[N - 1]);

    // Освобождение памяти
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_in2);
    cudaFree(d_out2);

    return 0;
}
