#include <iostream>
#include <cstdlib>   // rand(), srand()
#include <ctime>     // time()

using namespace std;

int main() {

    // Размер массива по заданию
    int size = 50000;

    // Динамическое выделение памяти под массив
    int* arr = new int[size];

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Заполнение массива случайными числами от 1 до 100
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100 + 1;
    }

    // Переменная для хранения суммы элементов
    long long sum = 0;

    // Подсчёт суммы всех элементов массива
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    // Вычисление среднего значения
    double average = static_cast<double>(sum) / size;

    // Вывод результата
    cout << "Среднее значение элементов массива: " << average << endl;

    // Освобождение динамически выделенной памяти
    delete[] arr;

    return 0;
}
