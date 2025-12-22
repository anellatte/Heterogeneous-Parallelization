#include <iostream>
#include <cstdlib>   // rand(), srand()
#include <ctime>     // time()
#include <chrono>    // измерение времени
#include <omp.h>     // OpenMP

using namespace std;
using namespace chrono;

int main() {

    // Размер массива (из задания 2)
    const int size = 1000000;

    // Динамическое выделение памяти
    int* arr = new int[size];

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Заполнение массива случайными числами от 1 до 100
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100 + 1;
    }

    /* ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК */

    int seqMin = arr[0];
    int seqMax = arr[0];

    auto startSeq = high_resolution_clock::now();

    for (int i = 1; i < size; i++) {
        if (arr[i] < seqMin) {
            seqMin = arr[i];
        }
        if (arr[i] > seqMax) {
            seqMax = arr[i];
        }
    }

    auto endSeq = high_resolution_clock::now();
    duration<double> timeSeq = endSeq - startSeq;

    /* ПАРАЛЛЕЛЬНЫЙ ПОИСК (OpenMP) */

    int parMin = arr[0];
    int parMax = arr[0];

    auto startPar = high_resolution_clock::now();

    #pragma omp parallel for reduction(min:parMin) reduction(max:parMax)
    for (int i = 0; i < size; i++) {
        if (arr[i] < parMin) {
            parMin = arr[i];
        }
        if (arr[i] > parMax) {
            parMax = arr[i];
        }
    }

    auto endPar = high_resolution_clock::now();
    duration<double> timePar = endPar - startPar;

    /* ВЫВОД РЕЗУЛЬТАТОВ */

    cout << "Последовательная версия:" << endl;
    cout << "Минимум = " << seqMin << endl;
    cout << "Максимум = " << seqMax << endl;
    cout << "Время выполнения = " << timeSeq.count() << " секунд\n" << endl;

    cout << "Параллельная версия (OpenMP):" << endl;
    cout << "Минимум = " << parMin << endl;
    cout << "Максимум = " << parMax << endl;
    cout << "Время выполнения = " << timePar.count() << " секунд" << endl;

    // Освобождение памяти
    delete[] arr;

    return 0;
}
