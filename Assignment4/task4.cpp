#include <mpi.h>        // Основной заголовок MPI
#include <cstdio>       // printf
#include <cstdlib>      // rand, srand
#include <vector>       // std::vector
#include <cmath>        // fabs

#define DEFAULT_N 1000000   // Размер массива по умолчанию

// ------------------------------------------------------------
// Последовательный prefix sum (inclusive) для одного процесса
// ------------------------------------------------------------
void cpu_local_scan(const std::vector<float>& in,
                    std::vector<float>& out)
{
    int n = in.size();
    out.resize(n);

    if (n == 0) return;

    out[0] = in[0];
    for (int i = 1; i < n; i++) {
        out[i] = out[i - 1] + in[i];
    }
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank = 0;        // Номер текущего процесса
    int size = 1;        // Общее количество процессов

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер массива
    int N = DEFAULT_N;
    if (argc > 1) {
        N = std::atoi(argv[1]);
        if (N < 1) N = 1;
    }

    // --------------------------------------------------------
    // Шаг 1. Каждый процесс вычисляет свой диапазон индексов
    // --------------------------------------------------------
    int base = N / size;            // Минимальное число элементов на процесс
    int rest = N % size;            // Остаток

    int local_n = base + (rank < rest ? 1 : 0);

    // Начальный индекс текущего процесса в глобальном массиве
    int start_index = rank * base + (rank < rest ? rank : rest);

    // --------------------------------------------------------
    // Шаг 2. Формируем локальный входной массив
    // (каждый процесс сам генерирует свою часть)
    // --------------------------------------------------------
    std::vector<float> local_in(local_n);

    std::srand(42 + rank);          // Разный seed для каждого процесса
    for (int i = 0; i < local_n; i++) {
        local_in[i] = float((std::rand() % 10) + 1);
    }

    // --------------------------------------------------------
    // Синхронизация перед измерением времени
    // --------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // --------------------------------------------------------
    // Шаг 3. Локальный prefix sum
    // --------------------------------------------------------
    std::vector<float> local_scan;
    cpu_local_scan(local_in, local_scan);

    // --------------------------------------------------------
    // Шаг 4. Получаем сумму предыдущих процессов через MPI_Scan
    // --------------------------------------------------------
    float local_sum = (local_n > 0) ? local_scan.back() : 0.0f;
    float global_prefix_sum = 0.0f;

    // MPI_Scan считает inclusive prefix sum по процессам
    MPI_Scan(&local_sum,
             &global_prefix_sum,
             1,
             MPI_FLOAT,
             MPI_SUM,
             MPI_COMM_WORLD);

    // Смещение = сумма всех предыдущих процессов
    float offset = global_prefix_sum - local_sum;

    // --------------------------------------------------------
    // Шаг 5. Добавляем offset ко всем локальным элементам
    // --------------------------------------------------------
    for (int i = 0; i < local_n; i++) {
        local_scan[i] += offset;
    }

    // --------------------------------------------------------
    // Завершаем измерение времени
    // --------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    // --------------------------------------------------------
    // Берём максимальное время по всем процессам
    // --------------------------------------------------------
    double max_time = 0.0;
    MPI_Reduce(&elapsed,
               &max_time,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);

    // --------------------------------------------------------
    // Вывод результатов (только root)
    // --------------------------------------------------------
    if (rank == 0) {
        printf("============================================================\n");
        printf(" MPI Distributed Prefix Sum (alternative solution)\n");
        printf(" Array size: %d\n", N);
        printf(" Processes : %d\n", size);
        printf("------------------------------------------------------------\n");
        printf(" Execution time (max): %.6f s\n", max_time);
        printf("============================================================\n");
    }

    // Завершаем MPI
    MPI_Finalize();
    return 0;
}