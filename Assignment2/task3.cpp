#include <iostream>     // ввод и вывод
#include <vector>       // массив vector
#include <cstdlib>      // rand(), srand()
#include <ctime>        // time()
#include <chrono>       // измерение времени
#include <omp.h>        // OpenMP

// Последовательная сортировка выбором
void selectionSortSequential(std::vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        int min_index = i;

        // Поиск минимального элемента
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }

        // Меняем местами текущий элемент и минимальный
        std::swap(arr[i], arr[min_index]);
    }
}

// Параллельная сортировка выбором (OpenMP)
void selectionSortParallel(std::vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        int min_value = arr[i];

        // Параллельный поиск минимума
        #pragma omp parallel
        {
            int local_min_value = min_value;
            int local_min_index = min_index;

            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < local_min_value) {
                    local_min_value = arr[j];
                    local_min_index = j;
                }
            }

            // Критическая секция для выбора глобального минимума
            #pragma omp critical
            {
                if (local_min_value < min_value) {
                    min_value = local_min_value;
                    min_index = local_min_index;
                }
            }
        }

        // Меняем элементы местами
        std::swap(arr[i], arr[min_index]);
    }
}

int main() {
    // Размеры массивов для тестирования
    int sizes[] = {1000, 10000};

    for (int size : sizes) {
        std::cout << "\nРазмер массива: " << size << std::endl;

        std::vector<int> arr(size);
        srand(time(nullptr));

        // Заполняем массив случайными числами
        for (int i = 0; i < size; i++) {
            arr[i] = rand() % 10000;
        }

        // Копии массива для разных реализаций
        std::vector<int> arr_seq = arr;
        std::vector<int> arr_par = arr;

        /* Последовательная версия */

        auto start_seq = std::chrono::high_resolution_clock::now();
        selectionSortSequential(arr_seq);
        auto end_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_seq = end_seq - start_seq;

        /* Параллельная версия */

        auto start_par = std::chrono::high_resolution_clock::now();
        selectionSortParallel(arr_par);
        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_par = end_par - start_par;

        /* Вывод времени */

        std::cout << "Последовательная сортировка: "
                  << time_seq.count() << " секунд" << std::endl;

        std::cout << "Параллельная сортировка (OpenMP): "
                  << time_par.count() << " секунд" << std::endl;
    }

    return 0;
}
