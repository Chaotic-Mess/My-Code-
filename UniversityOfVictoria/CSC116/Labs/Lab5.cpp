/*
 * Program: C++ References, Structs, and Exceptions (Lab5)
 * Author: Zac Matthias
 * Date: October 24, 2025
 * Description:
 *   This program contains four exercises demonstrating C++ references,
 *   structs, and exception handling:
 *     1. Exercise 1 : Solves a quadratic equation (references, exceptions)
 *     2. Exercise 2 : Implements a Matrix struct (vectors, identity, zero)
 *     3. Exercise 3 : Prompts user until valid integer is entered (exceptions)
 *     4. Exercise 4 : Draws a spiral pattern in an n x n matrix (references)
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout)
 *   - <vector> for dynamic arrays
 *   - <cmath> for sqrt()
 *   - <stdexcept> for exceptions
 *   - <limits> for input validation
 *   - <string> for integer checker
 *
 *   PS E:\CSC116> g++ Lab5.cpp -o Lab5
 *   PS E:\CSC116> ./Lab5
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>

// ===============================================================
// Exercise 1: Quadratic Solver with References and Exceptions
// ===============================================================
void solveQuadratic(double a, double b, double c, double &r1, double &r2) {
    if (a == 0) {
        if (b == 0) {
            throw std::invalid_argument("Not a valid equation (a and b are zero).");
        } else {
            r1 = r2 = -c / b;
            return;
        }
    }

    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        throw std::invalid_argument("No real roots (discriminant < 0).");
    }

    r1 = (-b + std::sqrt(discriminant)) / (2 * a);
    r2 = (-b - std::sqrt(discriminant)) / (2 * a);
}

void exercise1() {
    std::cout << "=== Exercise 1: Quadratic Equation Solver ===\n";

    double a, b, c;
    std::cout << "Enter coefficients a, b, c:\n> ";
    std::cin >> a >> b >> c;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    double r1 = 0, r2 = 0;

    try {
        solveQuadratic(a, b, c, r1, r2);
        std::cout << "Roots: " << r1 << " and " << r2 << "\n\n";
    } catch (const std::invalid_argument &e) {
        std::cout << "Error: " << e.what() << "\n\n";
    }
}

// ===============================================================
// Exercise 2: Structs and References  
// ===============================================================
struct Matrix {
    std::vector<std::vector<double>> data;

    Matrix(int n, int m, bool identity = false) {
        data.assign(n, std::vector<double>(m, 0));
        if (identity && n == m) {
            for (int i = 0; i < n; ++i)
                data[i][i] = 1;
        }
    }

    void print() const {
        for (const auto &row : data) {
            for (double val : row)
                std::cout << val << " ";
            std::cout << "\n";
        }
    }
};

void exercise2() {
    std::cout << "=== Exercise 2: Matrix Struct Demo ===\n";

    int n, m;
    std::cout << "Enter matrix dimensions (n m):\n> ";
    std::cin >> n >> m;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    Matrix zeroMatrix(n, m);
    std::cout << "\nZero matrix:\n";
    zeroMatrix.print();

    if (n == m) {
        Matrix identityMatrix(n, n, true);
        std::cout << "\nIdentity matrix:\n";
        identityMatrix.print();
    } else {
        std::cout << "\nCannot create identity matrix (not square).\n";
    }

    std::cout << "\n";
}

// ===============================================================
// Exercise 3: Exceptions and Input Validation (fixed)
// ===============================================================
std::string trim(const std::string &text) {
    const std::string whitespace = " \t\r\n\f\v";
    const std::size_t begin = text.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1);
}

int safeInput() {
    while (true) {
        std::cout << "Input an integer:\n> ";
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("Input stream closed before receiving a valid integer.");
        }

        const std::string cleaned = trim(line);
        try {
            std::size_t pos = 0;
            const int value = std::stoi(cleaned, &pos);
            if (pos != cleaned.size()) {
                throw std::invalid_argument("Extra characters detected.");
            }
            return value;
        } catch (const std::invalid_argument &) {
            std::cout << "Try again:\n";
        } catch (const std::out_of_range &) {
            std::cout << "Number too large. Try again:\n";
        }
    }
}

void exercise3() {
    std::cout << "=== Exercise 3: Safe Integer Input ===\n";
    try {
        const int number = safeInput();
        std::cout << "The number is " << number << "\n\n";
    } catch (const std::runtime_error &e) {
        std::cout << e.what() << "\n\n";
    }
}

// ===============================================================
// Exercise 4: Spiral Matrix 
// ===============================================================
 
void exercise4() {
    std::cout << "=== Exercise 4: Spiral Matrix ===\n";
    std::cout << "Enter matrix size:\n> ";
    int n;
    if (!(std::cin >> n) || n <= 0) {
        std::cout << "Invalid size. Please enter a positive integer.\n\n";
        return;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));

    const int dr[4] = {0, 1, 0, -1}; // right, down, left, up
    const int dc[4] = {1, 0, -1, 0};

    int row = 0;
    int col = 0;
    int direction = 0;

    for (int length = n; length > 0; --length) {
        for (int step = 0; step < length; ++step) {
            matrix[row][col] = length;
            if (step < length - 1) {
                row += dr[direction];
                col += dc[direction];
            }
        }
        direction = (direction + 1) % 4;
        if (length == 1) {
            break;
        }
        row += dr[direction];
        col += dc[direction];
    }

    std::cout << "\nSpiral matrix:\n";
    for (const auto &matrixRow : matrix) {
        for (size_t i = 0; i < matrixRow.size(); ++i) {
            std::cout << matrixRow[i];
            if (i + 1 < matrixRow.size()) {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ===============================================================
// Main
// ===============================================================
int main() {
    exercise1();
    exercise2();
    exercise3();
    exercise4();
    return 0;
}
