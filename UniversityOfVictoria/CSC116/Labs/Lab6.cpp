/*
 * Program: C++ Matrices and Exceptions (Lab6)
 * Author: Zac Matthias
 * Date: October 28, 2025
 * Description:
 *   This program contains 4 exercises demonstrating C++ matrices and exception handling:
 *     1. Exercise 1 : Construct an N x M random matrix
 *     2. Exercise 2 : Add two matrices with error checking
 *     3. Exercise 3 : Compute matrix trace with error checking
 *     4. Exercise 4 : Verify matrix validity with custom exception
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout, getline)
 *   - <vector> for dynamic arrays
 *   - <random> for random number generation
 *   - <stdexcept> for standard exceptions
 *   - <exception> for custom exceptions
 *   - <string> for string operations
 *
 *   PS E:\CSC116> g++ Lab6.cpp -o Lab6
 *   PS E:\CSC116> ./Lab6
 */
#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <exception>
#include <string>

// Define matrix type using 'using' statement
using matrix_type = std::vector<std::vector<double>>;

// Custom exception for Exercise 4
class InvalidMatrix : public std::exception {
private:
    std::string message;
public:
    InvalidMatrix(const std::string& msg) : message(msg) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
};

// Exercise 1: Construct an N x M random matrix
matrix_type create_random_matrix(int rows, int cols, double min_val = 0.0, double max_val = 100.0) {
    matrix_type matrix(rows, std::vector<double>(cols));
    
    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);
    
    // Fill matrix with random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dis(gen);
        }
    }
    
    return matrix;
}

// Exercise 2: Add two matrices together
matrix_type add_matrices(const matrix_type& matrix1, const matrix_type& matrix2) {
    // Check if matrices are empty
    if (matrix1.empty() || matrix2.empty()) {
        throw std::domain_error("Cannot add empty matrices");
    }
    
    // Check if matrices have the same dimensions
    if (matrix1.size() != matrix2.size()) {
        throw std::domain_error("Matrices must have the same number of rows");
    }
    
    if (matrix1[0].size() != matrix2[0].size()) {
        throw std::domain_error("Matrices must have the same number of columns");
    }
    
    int rows = matrix1.size();
    int cols = matrix1[0].size();
    matrix_type result(rows, std::vector<double>(cols));
    
    // Add corresponding elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    
    return result;
}

// Exercise 3: Compute the trace of a matrix
double trace(const matrix_type& matrix) {
    // Check if matrix is empty
    if (matrix.empty()) {
        throw std::domain_error("Cannot compute trace of an empty matrix");
    }
    
    // Check if matrix is square
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    if (rows != cols) {
        throw std::domain_error("Matrix must be square to compute trace");
    }
    
    // Compute trace (sum of diagonal elements)
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        sum += matrix[i][i];
    }
    
    return sum;
}

// Exercise 4: Verify if a matrix is valid (all rows have the same length)
bool is_valid_matrix(const matrix_type& matrix) {
    // Empty matrix is considered valid
    if (matrix.empty()) {
        return true;
    }
    
    // Check if all rows have the same length as the first row
    size_t expected_cols = matrix[0].size();
    
    for (size_t i = 1; i < matrix.size(); i++) {
        if (matrix[i].size() != expected_cols) {
            throw InvalidMatrix("Matrix is invalid: row " + std::to_string(i) + 
                              " has " + std::to_string(matrix[i].size()) + 
                              " columns, expected " + std::to_string(expected_cols));
        }
    }
    
    return true;
}

// Helper function to print a matrix
void print_matrix(const matrix_type& matrix, const std::string& name = "Matrix") {
    std::cout << name << " (" << matrix.size() << "x" << matrix[0].size() << "):\n";
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Exercise 1
void exercise1() {
    std::cout << "--- Exercise 1: Create Random Matrices ---\n";
    matrix_type mat1 = create_random_matrix(3, 3, 0.0, 10.0);
    matrix_type mat2 = create_random_matrix(3, 3, 0.0, 10.0);
    print_matrix(mat1, "Matrix 1");
    print_matrix(mat2, "Matrix 2");
}

// Exercise 2
void exercise2() {
    std::cout << "--- Exercise 2: Add Matrices ---\n";
    
    matrix_type mat1 = create_random_matrix(3, 3, 0.0, 10.0);
    matrix_type mat2 = create_random_matrix(3, 3, 0.0, 10.0);
    
    try {
        matrix_type sum = add_matrices(mat1, mat2);
        print_matrix(sum, "Sum of Matrix 1 and Matrix 2");
    } catch (const std::domain_error& e) {
        std::cout << "Error adding matrices: " << e.what() << "\n\n";
    }
    
    // Test with incompatible matrices
    std::cout << "Testing addition with incompatible matrices:\n";
    matrix_type mat3 = create_random_matrix(2, 3, 0.0, 10.0);
    try {
        matrix_type sum = add_matrices(mat1, mat3);
    } catch (const std::domain_error& e) {
        std::cout << "Caught expected error: " << e.what() << "\n\n";
    }
}

// Exercise 3
void exercise3() {
    std::cout << "--- Exercise 3: Compute Trace ---\n";
    
    matrix_type mat1 = create_random_matrix(3, 3, 0.0, 10.0);
    
    try {
        double tr = trace(mat1);
        std::cout << "Trace of Matrix 1: " << tr << "\n\n";
    } catch (const std::domain_error& e) {
        std::cout << "Error computing trace: " << e.what() << "\n\n";
    }
    
    // Test with non-square matrix
    std::cout << "Testing trace with non-square matrix:\n";
    matrix_type mat4 = create_random_matrix(3, 4, 0.0, 10.0);
    try {
        double tr = trace(mat4);
    } catch (const std::domain_error& e) {
        std::cout << "Caught expected error: " << e.what() << "\n\n";
    }
}

// Exercise 4
void exercise4() {
    std::cout << "--- Exercise 4: Verify Matrix Validity ---\n";
    
    matrix_type mat1 = create_random_matrix(3, 3, 0.0, 10.0);
    
    // Test with valid matrix
    std::cout << "Testing valid matrix:\n";
    try {
        if (is_valid_matrix(mat1)) {
            std::cout << "Matrix 1 is valid!\n\n";
        }
    } catch (const InvalidMatrix& e) {
        std::cout << "Caught error: " << e.what() << "\n\n";
    }
    
    // Test with invalid matrix (inconsistent row lengths)
    std::cout << "Testing invalid matrix (inconsistent row lengths):\n";
    matrix_type invalid_mat = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0},
        {6.0, 7.0, 8.0}
    };
    try {
        if (is_valid_matrix(invalid_mat)) {
            std::cout << "Invalid matrix is valid!\n\n";
        }
    } catch (const InvalidMatrix& e) {
        std::cout << "Caught expected error: " << e.what() << "\n\n";
    }
}

int main() {
    exercise1();
    exercise2();
    exercise3();
    exercise4();
    
    return 0;
}