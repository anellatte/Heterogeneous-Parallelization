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

// Пузырьковая сортировка (seq)
void bubbleSortSeq(int arr[], int n) {
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

// Сортировка выбором (seq)
void selectionSortSeq(int arr[], int n) {
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

// Сортировка вставками (seq)
void insertionSortSeq(int arr[], int n) {
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


// Сравнение сортировок
void compareSorts(int size) {
    int* arr = new int[size];

    cout << "\n==============================" << endl;
    cout << "Размер массива: " << size << endl;
    cout << "==============================" << endl;

    double seqTime, parTime;

    // Bubble Sort
    fillArray(arr, size);
    auto start = high_resolution_clock::now();
    bubbleSortSeq(arr, size);
    auto end = high_resolution_clock::now();
    seqTime = duration<double>(end - start).count();

    fillArray(arr, size);
    start = high_resolution_clock::now();
    bubbleSortOMP(arr, size);
    end = high_resolution_clock::now();
    parTime = duration<double>(end - start).count();

    cout << "Bubble Sort:" << endl;
    cout << "  Последовательно: " << seqTime << " сек" << endl;
    cout << "  Параллельно:     " << parTime << " сек" << endl;
    cout << "  Ускорение:       " << seqTime / parTime << endl;

    // Selection Sort 
    fillArray(arr, size);
    start = high_resolution_clock::now();
    selectionSortSeq(arr, size);
    end = high_resolution_clock::now();
    seqTime = duration<double>(end - start).count();

    fillArray(arr, size);
    start = high_resolution_clock::now();
    selectionSortOMP(arr, size);
    end = high_resolution_clock::now();
    parTime = duration<double>(end - start).count();

    cout << "\nSelection Sort:" << endl;
    cout << "  Последовательно: " << seqTime << " сек" << endl;
    cout << "  Параллельно:     " << parTime << " сек" << endl;
    cout << "  Ускорение:       " << seqTime / parTime << endl;

    // Insertion Sort 
    fillArray(arr, size);
    start = high_resolution_clock::now();
    insertionSortSeq(arr, size);
    end = high_resolution_clock::now();
    seqTime = duration<double>(end - start).count();

    fillArray(arr, size);
    start = high_resolution_clock::now();
    insertionSortOMP(arr, size);
    end = high_resolution_clock::now();
    parTime = duration<double>(end - start).count();

    cout << "\nInsertion Sort:" << endl;
    cout << "  Последовательно: " << seqTime << " сек" << endl;
    cout << "  Параллельно:     " << parTime << " сек" << endl;
    cout << "  Ускорение:       " << seqTime / parTime << endl;

    delete[] arr;
}

int main() {
    srand(time(nullptr));

    // сравнение на разных размерах
    compareSorts(1000);
    compareSorts(10000);
    compareSorts(100000);

    return 0;
}
