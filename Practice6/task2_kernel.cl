// OpenCL ядро умножения матриц
__kernel void matmul(__global const float* A,
                     __global const float* B,
                     __global float* C,
                     int N, int M, int K)
{
    int col = get_global_id(0); // столбец
    int row = get_global_id(1); // строка

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
