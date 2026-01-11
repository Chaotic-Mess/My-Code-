/*
 * Program: C++ Lab 8
 * Author: Zac Matthias
 * Date: 2025-11-21
 * Description:
 *   This program contains 4 exercises demonstrating C++ Classes, Operators, and Iterators:
 *     1. Exercise 1 : StopWatch class
 *     2. Exercise 2 : QuadraticEquation class 
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout, getline)
 *   - <chrono> for StopWatch
 *   - <cmath> for QuadraticEquation
 *   - <stdexcept> for exceptions
 * 
 *   PS E:\CSC116> g++ Lab8.cpp -o Lab8
 *   PS E:\CSC116> ./Lab8
 */
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>

using namespace std;

// Exercise 1: StopWatch
class StopWatch {
private:
    chrono::time_point<chrono::high_resolution_clock> start_time;
    chrono::time_point<chrono::high_resolution_clock> end_time;
    bool running;

public:
    StopWatch() {
        start();
    }

    void start() {
        start_time = chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = chrono::high_resolution_clock::now();
        running = false;
    }

    double get_time() {
        chrono::time_point<chrono::high_resolution_clock> end;
        if (running) {
            end = chrono::high_resolution_clock::now();
        } else {
            end = end_time;
        }
        chrono::duration<double> elapsed = end - start_time;
        return elapsed.count();
    }
};

// Exercise 2: QuadraticEquation
class QuadraticEquation {
private:
    double a, b, c;

    void check_solution(double a_in, double b_in, double c_in) {
        double discriminant = b_in * b_in - 4 * a_in * c_in;
        if (discriminant < 0) {
            throw invalid_argument("No real solution (discriminant < 0)");
        }
    }

public:
    QuadraticEquation(double a, double b, double c) {
        check_solution(a, b, c);
        this->a = a;
        this->b = b;
        this->c = c;
    }

    double get_a() const { return a; }
    double get_b() const { return b; }
    double get_c() const { return c; }

    void set_a(double a) { check_solution(a, b, c); this->a = a; }
    void set_b(double b) { check_solution(a, b, c); this->b = b; }
    void set_c(double c) { check_solution(a, b, c); this->c = c; }

    double get_discriminant() const {
        return b * b - 4 * a * c;
    }

    bool has_real_solution() const {
        return get_discriminant() >= 0;
    }

    bool is_quadratic() const {
        return a != 0;
    }

    bool has_duplicated_solution() const {
        return get_discriminant() == 0;
    }

    double get_solution1() const {
        return (-b + sqrt(get_discriminant())) / (2 * a);
    }

    double get_solution2() const {
        return (-b - sqrt(get_discriminant())) / (2 * a);
    }
};  

void Exercise1() {
    cout << "--- Exercise 1: StopWatch ---" << endl;
    StopWatch sw; 
    long long sum = 0; for (int i = 0; i < 210000000; ++i) sum += i;
    sw.stop();
    cout << "Elapsed time: " << sw.get_time() << " seconds" << endl;
}

void Exercise2() {
    cout << "\n--- Exercise 2: QuadraticEquation ---" << endl;
    try {
        QuadraticEquation qe(1, -3, 2); // x^2 - 3x + 2 = 0 -> (x-1)(x-2)=0
        cout << "Equation: " << qe.get_a() << "x^2 + " << qe.get_b() << "x + " << qe.get_c() << " = 0" << endl;
        cout << "Discriminant: " << qe.get_discriminant() << endl;
        cout << "Solution 1: " << qe.get_solution1() << endl;
        cout << "Solution 2: " << qe.get_solution2() << endl;
    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
    }
} 

int main() {
    Exercise1();
    Exercise2(); 
    return 0;
}