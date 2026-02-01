#include <omp.h>            // OpenMP (параллельные директивы, omp_get_wtime)
#include <iostream>         // cout
#include <vector>           // vector
#include <random>           // генератор случайных чисел
#include <iomanip>          // формат вывода

int main(int argc, char** argv) {

    // -------------------- Параметры --------------------
    long long N = (argc > 1) ? atoll(argv[1]) : 30000000;   // размер массива
    int num_threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    omp_set_num_threads(num_threads);                      // устанавливаем число потоков

    std::vector<double> data(N);                           // массив данных

    // -------------------- Последовательная инициализация --------------------
    double t_start = omp_get_wtime();

    std::mt19937 gen(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (long long i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    double t_after_init = omp_get_wtime();

    // -------------------- Параллельный расчёт суммы --------------------
    double sum = 0.0;
    double t_sum_start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; ++i) {
        sum += data[i];
    }

    double t_sum_end = omp_get_wtime();

    double mean = sum / (double)N;                         // среднее значение

    // -------------------- Параллельный расчёт дисперсии --------------------
    double var_acc = 0.0;
    double t_var_start = omp_get_wtime();

    #pragma omp parallel for reduction(+:var_acc)
    for (long long i = 0; i < N; ++i) {
        double diff = data[i] - mean;
        var_acc += diff * diff;
    }

    double t_var_end = omp_get_wtime();

    double variance = var_acc / (double)N;

    // -------------------- Итоговое время --------------------
    double t_end = omp_get_wtime();

    double t_init  = t_after_init - t_start;
    double t_sum   = t_sum_end - t_sum_start;
    double t_var   = t_var_end - t_var_start;
    double t_total = t_end - t_start;

    // -------------------- Вывод --------------------
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N = " << N << ", threads = " << num_threads << "\n";
    std::cout << "sum = " << sum << "\n";
    std::cout << "mean = " << mean << "\n";
    std::cout << "variance = " << variance << "\n\n";

    std::cout << "Timing (seconds):\n";
    std::cout << "init (sequential) = " << t_init << "\n";
    std::cout << "sum  (parallel)   = " << t_sum  << "\n";
    std::cout << "var  (parallel)   = " << t_var  << "\n";
    std::cout << "total             = " << t_total << "\n";

    return 0;
}
