#include <iostream>
#include <cstdlib>   // rand(), srand()
#include <ctime>     // time()
#include <chrono>    // измерение времени
#include <omp.h>     // OpenMP

using namespace std;
using namespace chrono;

int main() {

    // Размер массива по заданию
    const int size = 5000000;

    // Динамическое выделение памяти
    int* arr = new int[size];

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Заполнение массива случайными числами от 1 до 100
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100 + 1;
    }

    /* ПОСЛЕДОВАТЕЛЬНОЕ ВЫЧИСЛЕНИЕ */

    long long seqSum = 0;

    auto startSeq = high_resolution_clock::now();

    for (int i = 0; i < size; i++) {
        seqSum += arr[i];
    }

    double seqAverage = static_cast<double>(seqSum) / size;

    auto endSeq = high_resolution_clock::now();
    duration<double> timeSeq = endSeq - startSeq;

    /* ПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ (OpenMP) */

    long long parSum = 0;

    auto startPar = high_resolution_clock::now();

    #pragma omp parallel for reduction(+:parSum)
    for (int i = 0; i < size; i++) {
        parSum += arr[i];
    }

    double parAverage = static_cast<double>(parSum) / size;

    auto endPar = high_resolution_clock::now();
    duration<double> timePar = endPar - startPar;

    /* ВЫВОД РЕЗУЛЬТАТОВ */

    cout << "Последовательная версия:" << endl;
    cout << "Среднее значение = " << seqAverage << endl;
    cout << "Время выполнения = " << timeSeq.count() << " секунд\n" << endl;

    cout << "Параллельная версия (OpenMP):" << endl;
    cout << "Среднее значение = " << parAverage << endl;
    cout << "Время выполнения = " << timePar.count() << " секунд" << endl;

    // Освобождение динамической памяти
    delete[] arr;

    return 0;
}
