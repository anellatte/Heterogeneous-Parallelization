#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

void cpuSortTest(int n) {
    std::vector<int> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    for (int& x : data) x = dist(gen);

    auto start = std::chrono::high_resolution_clock::now();
    std::sort(data.begin(), data.end());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> time = end - start;

    std::cout << "CPU sort, N = " << n
              << ", time = " << time.count() << " ms\n";
}

int main() {
    cpuSortTest(10000);
    cpuSortTest(100000);
    cpuSortTest(1000000);
    return 0;
}
