#include <iostream>
#include <cstdlib>   // rand(), srand()
#include <ctime>     // time()
#include <chrono>    // измерение времени
#include <omp.h>     // OpenMP

using namespace std;

// Функция для последовательного вычисления среднего значения
double sequentialAverage(int* arr, int size) {
    long long sum = 0;

    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    return static_cast<double>(sum) / size;
}

// Функция для параллельного вычисления среднего значения
double parallelAverage(int* arr, int size) {
    long long sum = 0;

    // Параллельное суммирование с использованием reduction
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    return static_cast<double>(sum) / size;
}

int main() {

    int N;
    cout << "Введите размер массива: ";
    cin >> N;

    // ================================
    // 1. Выделение динамической памяти
    // ================================
    int* array = new int[N];

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Заполнение массива случайными числами от 1 до 100
    for (int i = 0; i < N; i++) {
        array[i] = rand() % 100 + 1;
    }

    // ================================
    // 2. Последовательный подсчёт
    // ================================
    auto start_seq = chrono::high_resolution_clock::now();
    double avg_seq = sequentialAverage(array, N);
    auto end_seq = chrono::high_resolution_clock::now();

    chrono::duration<double> time_seq = end_seq - start_seq;

    // ================================
    // 3. Параллельный подсчёт (OpenMP)
    // ================================
    auto start_par = chrono::high_resolution_clock::now();
    double avg_par = parallelAverage(array, N);
    auto end_par = chrono::high_resolution_clock::now();

    chrono::duration<double> time_par = end_par - start_par;

    // ================================
    // 4. Вывод результатов
    // ================================
    cout << "\nПоследовательная версия:" << endl;
    cout << "Среднее значение = " << avg_seq << endl;
    cout << "Время выполнения = " << time_seq.count() << " секунд" << endl;

    cout << "\nПараллельная версия (OpenMP):" << endl;
    cout << "Среднее значение = " << avg_par << endl;
    cout << "Время выполнения = " << time_par.count() << " секунд" << endl;

    // ================================
    // 5. Освобождение памяти
    // ================================
    delete[] array;

    return 0;
}
