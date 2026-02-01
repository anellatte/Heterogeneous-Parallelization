#include <CL/cl.h>              // Основной заголовок OpenCL
#include <iostream>             // Ввод-вывод
#include <vector>               // Контейнер vector
#include <chrono>               // Замер времени
#include <cmath>                // fabs
#include <fstream>              // Чтение kernel-файла
#include <sstream>              // Работа со строками
#include <iomanip>              // Форматированный вывод

// ------------------------------------------------------------
// Проверка ошибок OpenCL
// ------------------------------------------------------------
static void cl_check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL ошибка (" << err << "): " << msg << std::endl;
        std::exit(1);
    }
}

// ------------------------------------------------------------
// Чтение kernel.cl в строку
// ------------------------------------------------------------
static std::string load_kernel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Не удалось открыть файл ядра: " << filename << std::endl;
        std::exit(1);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// ------------------------------------------------------------
// CPU-версия умножения матриц (для проверки корректности)
// ------------------------------------------------------------
void matmul_cpu(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int N, int M, int K)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < M; ++t) {
                sum += A[i * M + t] * B[t * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// ------------------------------------------------------------
// Главная функция
// ------------------------------------------------------------
int main() {

    // -------------------------------
    // Размеры матриц
    // -------------------------------
    const int N = 512;
    const int M = 512;
    const int K = 512;

    std::cout << "Task 2: OpenCL matrix multiplication\n";
    std::cout << "N=" << N << " M=" << M << " K=" << K << "\n\n";

    // -------------------------------
    // Инициализация данных на CPU
    // -------------------------------
    std::vector<float> A(N * M);
    std::vector<float> B(M * K);
    std::vector<float> C_gpu(N * K, 0.0f);
    std::vector<float> C_cpu(N * K, 0.0f);

    for (int i = 0; i < N * M; ++i) A[i] = 0.01f * (i % 100);
    for (int i = 0; i < M * K; ++i) B[i] = 0.02f * (i % 100);

    // -------------------------------
    // Поиск OpenCL платформы и устройства
    // -------------------------------
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    err = clGetPlatformIDs(1, &platform, nullptr);
    cl_check(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_check(err, "clGetDeviceIDs");

    // -------------------------------
    // Создание контекста и очереди
    // -------------------------------
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_check(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    cl_check(err, "clCreateCommandQueue");

    // -------------------------------
    // Загрузка и компиляция ядра
    // -------------------------------
    std::string kernel_src = load_kernel("task_2_kernel.cl");
    const char* src_ptr = kernel_src.c_str();
    size_t src_size = kernel_src.size();

    cl_program program = clCreateProgramWithSource(
        context, 1, &src_ptr, &src_size, &err);
    cl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_check(err, "clBuildProgram");

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    cl_check(err, "clCreateKernel");

    // -------------------------------
    // Буферы памяти на устройстве
    // -------------------------------
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * A.size(), nullptr, &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * B.size(), nullptr, &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * C_gpu.size(), nullptr, &err);

    cl_check(err, "clCreateBuffer");

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
                         sizeof(float) * A.size(), A.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
                         sizeof(float) * B.size(), B.data(), 0, nullptr, nullptr);

    // -------------------------------
    // Аргументы ядра
    // -------------------------------
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &M);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // -------------------------------
    // Запуск ядра
    // -------------------------------
    size_t global[2] = { (size_t)K, (size_t)N };

    auto t0 = std::chrono::high_resolution_clock::now();

    clEnqueueNDRangeKernel(queue, kernel, 2,
                           nullptr, global, nullptr,
                           0, nullptr, nullptr);
    clFinish(queue);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = t1 - t0;

    // -------------------------------
    // Чтение результата
    // -------------------------------
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                        sizeof(float) * C_gpu.size(),
                        C_gpu.data(), 0, nullptr, nullptr);

    // -------------------------------
    // CPU-проверка
    // -------------------------------
    auto c0 = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_cpu, N, M, K);
    auto c1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = c1 - c0;

    // -------------------------------
    // Проверка ошибки
    // -------------------------------
    float max_err = 0.0f;
    for (size_t i = 0; i < C_gpu.size(); ++i)
        max_err = std::max(max_err, std::fabs(C_gpu[i] - C_cpu[i]));

    // -------------------------------
    // Вывод результатов
    // -------------------------------
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "OpenCL time (ms): " << gpu_time.count() << "\n";
    std::cout << "CPU time (ms):    " << cpu_time.count() << "\n";
    std::cout << "Max error:        " << max_err << "\n";

    // -------------------------------
    // Очистка ресурсов
    // -------------------------------
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
