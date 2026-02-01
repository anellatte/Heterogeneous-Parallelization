#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace chrono;

// Заполнение массива случайными числами
void fillArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100000;
    }
}

// Пузырьковая сортировка
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Сортировка выбором
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;

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

// Сортировка вставками
void insertionSort(int arr[], int n) {
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
void testSequentialSorts(int size) {
    int* arr = new int[size];

    cout << "\nРазмер массива: " << size << endl;

    // Bubble Sort
    fillArray(arr, size);
    auto start = high_resolution_clock::now();
    bubbleSort(arr, size);
    auto end = high_resolution_clock::now();
    cout << "Bubble Sort (seq): "
         << duration<double>(end - start).count()
         << " секунд" << endl;

    // Selection Sort
    fillArray(arr, size);
    start = high_resolution_clock::now();
    selectionSort(arr, size);
    end = high_resolution_clock::now();
    cout << "Selection Sort (seq): "
         << duration<double>(end - start).count()
         << " секунд" << endl;

    // Insertion Sort
    fillArray(arr, size);
    start = high_resolution_clock::now();
    insertionSort(arr, size);
    end = high_resolution_clock::now();
    cout << "Insertion Sort (seq): "
         << duration<double>(end - start).count()
         << " секунд" << endl;

    delete[] arr;
}

int main() {
    srand(time(nullptr));

    // тестирование на трёх размерах
    testSequentialSorts(1000);
    testSequentialSorts(10000);
    testSequentialSorts(100000);

    return 0;
}
