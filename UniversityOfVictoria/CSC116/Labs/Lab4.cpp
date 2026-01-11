/*
 * Program: C++ Vectors and References (Lab4)
 * Author: Zac Matthias
 * Date: October 15, 2025
 * Description:
 *   This program contains four exercises demonstrating C++ vectors and references:
 *     1. Exercise 1 : Computes the median of a vector of numbers.
 *     2. Exercise 2 : Removes duplicate integers from a vector.
 *     3. Exercise 3 : Swaps two integers using references.
 *     4. Exercise 4 : Compares performance between passing vectors by copy and by reference.
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout)
 *   - <vector> for dynamic arrays
 *   - <algorithm> for sorting and unique operations
 *   - <chrono> for measuring execution time
 *
 *   PS E:\CSC116> g++ Lab4.cpp -o Lab4
 *   PS E:\CSC116> ./Lab4
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std;

// ------------------- Exercise 1 -------------------
void exercise1() {
    cout << "=== Exercise 1: Median of a Vector ===" << endl;

    vector<double> numbers = {71, 49, 92, 87, 0, 66, 81, 74, 0, 51, 64, 94, 79}; // input

    cout << "Numbers:" << endl;
    for (double n : numbers)
        cout << n << " ";
    cout << endl;

    sort(numbers.begin(), numbers.end());
    int n = static_cast<int>(numbers.size());

    double median;
    if (n % 2 == 0) {
        median = (numbers[n / 2 - 1] + numbers[n / 2]) / 2.0;
    } else {
        median = numbers[n / 2];
    }

    cout << "\nThe median is " << median << endl << endl;
}

// ------------------- Exercise 2 -------------------
void exercise2() {
    cout << "=== Exercise 2: Remove Duplicates ===" << endl;

    vector<int> numbers = {1, 1, 1, 2, 6, 5, 1, 1, 6};

    cout << "Numbers:" << endl;
    for (int n : numbers)
        cout << n << " ";
    cout << endl;

    sort(numbers.begin(), numbers.end());
    numbers.erase(unique(numbers.begin(), numbers.end()), numbers.end());

    cout << "\nClean:" << endl;
    for (int n : numbers)
        cout << n << " ";
    cout << endl << endl;
}

// ------------------- Exercise 3 -------------------
void swapInts(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

void exercise3() {
    cout << "=== Exercise 3: Swap Two Integers ===" << endl;

    int a = 6, b = 10;
    cout << "Before swap: " << a << " " << b << endl;
    swapInts(a, b);
    cout << "After swap:  " << a << " " << b << endl << endl;
}

// ------------------- Exercise 4 -------------------
void sum_and_print_copy(vector<int> vec) {
    long long sum = 0;
    for (int n : vec)
        sum += n;
    cout << "Sum (copy): " << sum << endl;
}

void sum_and_print_ref(const vector<int>& vec) {
    long long sum = 0;
    for (int n : vec)
        sum += n;
    cout << "Sum (reference): " << sum << endl;
}

void exercise4() {
    cout << "=== Exercise 4: Vector Sum (Copy vs Reference) ===" << endl;

    vector<int> vec;
    vec.reserve(100000);
    for (int i = 0; i < 100000; i++)
        vec.push_back(i);

    auto start = chrono::high_resolution_clock::now();
    sum_and_print_copy(vec);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time taken by copy: " << duration.count() << " ms" << endl;

    start = chrono::high_resolution_clock::now();
    sum_and_print_ref(vec);
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time taken by reference: " << duration.count() << " ms" << endl << endl;
}

// ------------------- Main -------------------
int main() {
    exercise1();
    exercise2();
    exercise3();
    exercise4();
    return 0;
}
