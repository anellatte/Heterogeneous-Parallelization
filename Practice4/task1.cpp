#include <iostream>     // Для cout
#include <fstream>      // Для работы с файлами
#include <vector>       // Контейнер std::vector
#include <random>       // Генератор случайных чисел
#include <cstdint>      // uint64_t
#include <filesystem>   // Создание директорий
#include <cstdlib>      // strtoull

// ------------------------------------------------------------
// Функция сохранения массива float в бинарный файл
// Сначала записывается размер массива,
// затем все элементы массива подряд
// ------------------------------------------------------------
void save_binary(const std::string& filename,
                 const std::vector<float>& data)
{
    std::ofstream file(filename, std::ios::binary);   // Открываем файл в бинарном режиме
    if (!file) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << "\n";
        std::exit(1);
    }

    // Записываем размер массива
    uint64_t n = static_cast<uint64_t>(data.size());
    file.write(reinterpret_cast<const char*>(&n), sizeof(uint64_t));

    // Записываем сами данные
    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(float));

    file.close();
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    // ---------------- Размер массива ----------------
    size_t n = 1'000'000;   // Размер массива по умолчанию

    // Если пользователь передал размер через аргументы командной строки
    if (argc > 1) {
        n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
        if (n == 0) n = 1;  // Защита от нулевого размера
    }

    // ---------------- Создание директории ----------------
    std::filesystem::create_directories("results");

    // ---------------- Генерация данных ----------------
    std::vector<float> array(n);        // Массив данных
    double checksum = 0.0;              // Контрольная сумма

    std::mt19937 generator(42);          // Генератор с фиксированным seed
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Диапазон [0, 1]

    for (size_t i = 0; i < n; ++i) {
        array[i] = dist(generator);      // Случайное число
        checksum += array[i];            // Считаем сумму на CPU
    }

    // ---------------- Формирование имени файла ----------------
    std::string filename = "results/data_" + std::to_string(n) + ".bin";

    // ---------------- Сохранение ----------------
    save_binary(filename, array);

    // ---------------- Вывод информации ----------------
    std::cout << "=============================================\n";
    std::cout << "Задание 1: Генерация входных данных\n";
    std::cout << "=============================================\n";
    std::cout << "Размер массива: " << n << "\n";
    std::cout << "Файл: " << filename << "\n";
    std::cout << "Контрольная сумма (CPU): " << checksum << "\n";
    std::cout << "=============================================\n";

    return 0;
}