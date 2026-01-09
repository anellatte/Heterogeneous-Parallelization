%%writefile task1.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <climits>

// ------------------------------------------------------------
// ЯДРО 1: сортировка внутри одного блока (bitonic sort)
// ------------------------------------------------------------
__global__ void blockSort(int* data, int n) {
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n)
        shared[tid] = data[gid];
    else
        shared[tid] = INT_MAX;

    __syncthreads();

    for (int size = 2; size <= blockDim.x; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int other = tid ^ stride;
            if (other > tid) {
                bool ascending = ((tid & size) == 0);
                int a = shared[tid];
                int b = shared[other];
                if ((ascending && a > b) || (!ascending && a < b)) {
                    shared[tid] = b;
                    shared[other] = a;
                }
            }
            __syncthreads();
        }
    }

    if (gid < n)
        data[gid] = shared[tid];
}

// ------------------------------------------------------------
// ЯДРО 2: параллельное слияние отсортированных участков
// ------------------------------------------------------------
__global__ void mergeKernel(const int* src, int* dst, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * 2 * width;

    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start;
    int j = mid;
    int k = start;

    while (i < mid && j < end)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];

    while (i < mid) dst[k++] = src[i++];
    while (j < end) dst[k++] = src[j++];
}

// ------------------------------------------------------------
// GPU merge sort
// ------------------------------------------------------------
void gpuMergeSort(int* d_data, int n) {
    int* d_tmp;
    cudaMalloc(&d_tmp, n * sizeof(int));

    const int BLOCK = 1024;
    const int MERGE_THREADS = 256;

    int grid = (n + BLOCK - 1) / BLOCK;

    blockSort<<<grid, BLOCK, BLOCK * sizeof(int)>>>(d_data, n);
    cudaDeviceSynchronize();

    int* src = d_data;
    int* dst = d_tmp;

    for (int width = BLOCK; width < n; width <<= 1) {
        int pairs = (n + 2 * width - 1) / (2 * width);
        int blocks = (pairs + MERGE_THREADS - 1) / MERGE_THREADS;

        mergeKernel<<<blocks, MERGE_THREADS>>>(src, dst, n, width);
        cudaDeviceSynchronize();

        std::swap(src, dst);
    }

    if (src != d_data)
        cudaMemcpy(d_data, src, n * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFree(d_tmp);
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main() {
    const int N = 100000;

    std::vector<int> h(N);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (int& x : h) x = dist(gen);

    std::vector<int> ref = h;
    std::sort(ref.begin(), ref.end());

    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuMergeSort(d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h.data(), d, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GPU merge sort time: " << ms << " ms\n";
    cudaFree(d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
