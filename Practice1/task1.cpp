#include <iostream>     // Для ввода и вывода
#include <vector>       // Для работы с массивом (vector)
#include <cstdlib>      // Для rand() и srand()
#include <ctime>        // Для time()
#include <chrono>       // Для измерения времени
#include <omp.h>        // Для OpenMP

int main() {
    int N;

    // Ввод размера массива
    std::cout << "Введите размер массива: ";
    std::cin >> N;

    // Создание массива из N элементов
    std::vector<int> numbers(N);

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Заполнение массива случайными числами от 1 до 100
    for (int i = 0; i < N; i++) {
        numbers[i] = rand() % 100 + 1;
    }

    // Вывод массива
    /* std::cout << "\nМассив:\n";
    for (int i = 0; i < N; i++) {
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl; */

    // ================================
    // ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК
    // ================================

    int seq_min = numbers[0];
    int seq_max = numbers[0];

    // Начало измерения времени
    auto start_seq = std::chrono::high_resolution_clock::now();

    // Последовательный поиск минимума и максимума
    for (int i = 1; i < N; i++) {
        if (numbers[i] < seq_min) {
            seq_min = numbers[i];
        }
        if (numbers[i] > seq_max) {
            seq_max = numbers[i];
        }
    }

    // Конец измерения времени
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seq_time = end_seq - start_seq;

    // ================================
    // ПАРАЛЛЕЛЬНЫЙ ПОИСК (OpenMP)
    // ================================

    int par_min = numbers[0];
    int par_max = numbers[0];

    // Начало измерения времени
    auto start_par = std::chrono::high_resolution_clock::now();

    // Параллельный цикл с использованием reduction
    #pragma omp parallel for reduction(min:par_min) reduction(max:par_max)
    for (int i = 0; i < N; i++) {
        if (numbers[i] < par_min) {
            par_min = numbers[i];
        }
        if (numbers[i] > par_max) {
            par_max = numbers[i];
        }
    }

    // Конец измерения времени
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> par_time = end_par - start_par;

    // ================================
    // ВЫВОД РЕЗУЛЬТАТОВ
    // ================================

    std::cout << "\nПоследовательная версия:" << std::endl;
    std::cout << "Минимум = " << seq_min << std::endl;
    std::cout << "Максимум = " << seq_max << std::endl;
    std::cout << "Время выполнения = " << seq_time.count() << " секунд\n";

    std::cout << "\nПараллельная версия (OpenMP):" << std::endl;
    std::cout << "Минимум = " << par_min << std::endl;
    std::cout << "Максимум = " << par_max << std::endl;
    std::cout << "Время выполнения = " << par_time.count() << " секунд\n";

    return 0;
}
