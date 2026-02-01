#include <iostream>     // для ввода и вывода
#include <vector>       // для работы с массивом (vector)
#include <cstdlib>      // для rand() и srand()
#include <ctime>        // для time()
#include <chrono>       // для измерения времени
#include <omp.h>        // для OpenMP

int main() {
    const int N = 10000;  // размер массива

    // Создаём массив из 10 000 элементов
    std::vector<int> numbers(N);

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Заполняем массив случайными числами от 1 до 10000
    for (int i = 0; i < N; i++) {
        numbers[i] = rand() % 10000 + 1;
    }

    /* Последовательная реализация */

    int min_seq = numbers[0];
    int max_seq = numbers[0];

    // Замер времени начала
    auto start_seq = std::chrono::high_resolution_clock::now();

    // Поиск минимума и максимума
    for (int i = 1; i < N; i++) {
        if (numbers[i] < min_seq)
            min_seq = numbers[i];
        if (numbers[i] > max_seq)
            max_seq = numbers[i];
    }

    // Замер времени окончания
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_seq = end_seq - start_seq;

    /* Параллельная реализация (OpenMP) */

    int min_par = numbers[0];
    int max_par = numbers[0];

    // Замер времени начала
    auto start_par = std::chrono::high_resolution_clock::now();

    // Параллельный поиск минимума и максимума
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N; i++) {
        if (numbers[i] < min_par)
            min_par = numbers[i];
        if (numbers[i] > max_par)
            max_par = numbers[i];
    }

    // Замер времени окончания
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_par = end_par - start_par;

    /* Вывод результатов */

    std::cout << "Последовательная версия:" << std::endl;
    std::cout << "Минимум = " << min_seq << std::endl;
    std::cout << "Максимум = " << max_seq << std::endl;
    std::cout << "Время выполнения = " << time_seq.count() << " секунд" << std::endl;

    std::cout << std::endl;

    std::cout << "Параллельная версия (OpenMP):" << std::endl;
    std::cout << "Минимум = " << min_par << std::endl;
    std::cout << "Максимум = " << max_par << std::endl;
    std::cout << "Время выполнения = " << time_par.count() << " секунд" << std::endl;

    return 0;
}
