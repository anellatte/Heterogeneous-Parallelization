#include <iostream>
#include <cstdlib>   // rand(), srand()
#include <ctime>     // time()
#include <chrono>    // измерение времени

using namespace std;
using namespace chrono;

int main() {

    // Размер массива по заданию
    const int size = 1000000;

    // Динамическое выделение памяти
    int* arr = new int[size];

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Заполнение массива случайными числами от 1 до 100
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100 + 1;
    }

    // Инициализация минимума и максимума
    int minValue = arr[0];
    int maxValue = arr[0];

    // Начало измерения времени
    auto start = high_resolution_clock::now();

    // Последовательный поиск минимума и максимума
    for (int i = 1; i < size; i++) {
        if (arr[i] < minValue) {
            minValue = arr[i];
        }
        if (arr[i] > maxValue) {
            maxValue = arr[i];
        }
    }

    // Конец измерения времени
    auto end = high_resolution_clock::now();
    duration<double> elapsedTime = end - start;

    // Вывод результатов
    cout << "Минимальный элемент: " << minValue << endl;
    cout << "Максимальный элемент: " << maxValue << endl;
    cout << "Время выполнения (последовательно): "
         << elapsedTime.count() << " секунд" << endl;

    // Освобождение памяти
    delete[] arr;

    return 0;
}
