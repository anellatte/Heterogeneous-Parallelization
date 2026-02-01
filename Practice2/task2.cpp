#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

// Заполнение массива
void fillArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100000;
    }
}

// Пузырьковая сортировка (OpenMP)
void bubbleSortOMP(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Сортировка выбором (OpenMP)
void selectionSortOMP(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;

        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }

        int temp = arr[i];
        arr[i] = arr[minIndex];
        arr[minIndex] = temp;
    }
}

// Сортировка вставками (OpenMP)

void insertionSortOMP(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}

// Тестирование сортировок
void testSorts(int size) {
    int* arr = new int[size];

    cout << "\nРазмер массива: " << size << endl;

    // Bubble Sort OpenMP
    fillArray(arr, size);
    auto start = high_resolution_clock::now();
    bubbleSortOMP(arr, size);
    auto end = high_resolution_clock::now();
    cout << "Bubble Sort (OpenMP): "
         << duration<double>(end - start).count()
         << " секунд" << endl;

    // Selection Sort OpenMP
    fillArray(arr, size);
    start = high_resolution_clock::now();
    selectionSortOMP(arr, size);
    end = high_resolution_clock::now();
    cout << "Selection Sort (OpenMP): "
         << duration<double>(end - start).count()
         << " секунд" << endl;

    // Insertion Sort OpenMP
    fillArray(arr, size);
    start = high_resolution_clock::now();
    insertionSortOMP(arr, size);
    end = high_resolution_clock::now();
    cout << "Insertion Sort (OpenMP): "
         << duration<double>(end - start).count()
         << " секунд" << endl;

    delete[] arr;
}

int main() {
    srand(time(nullptr));

    // тестируем на разных размерах
    testSorts(1000);
    testSorts(10000);
    testSorts(100000);

    return 0;
}
