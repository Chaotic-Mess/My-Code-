/**
 * @brief Lab 2 focuses on parameters and function design
 * @author Zac Matthias
 * 
 * This program demonstrates various functions: sum of two numbers, 
 * calculating distance, finding real roots of a quadratic equation, 
 * and analyzing integers.
 * 
 * Note: Add -lm to the compile command to link the math library.
 * Example: gcc -Wall -std=c18 -o TEST Lab2.c -lm
 */
#include <stdio.h>
#include <math.h>

// Function Prototypes
float print_sum(float Numb1, float Numb2);
float calculate_distance(float acceleration, float time, float velocity);
double print_real_roots(double a, double b, double c);
int integer_analysis(int a);

int main(void) {
    // Test print_sum function
    printf("Tests for print_sum:\n");
    print_sum(7.2, 1.6);               // Test with 7.2 and 1.6, should print their sum
    print_sum(2.55, 8.122);            // Test with 2.55 and 8.122
    print_sum(7.11111111111, 7.11111111111); // Test with repeating decimals for precision

    // Test calculate_distance function
    printf("\nTests for calculate_distance:\n");
    calculate_distance(3.0, 2.0, 2.0); // Test distance with given acceleration, time, and velocity

    // Test print_real_roots function
    printf("\nTests for print_real_roots:\n");
    print_real_roots(2,  3, -1);       // Test real roots for quadratic equation 2x^2 + 3x - 1 = 0
    print_real_roots(3,  12, 5);       // Test with different coefficients
    print_real_roots(-1, -4, 5);       // Test with negative leading coefficient
    print_real_roots(6,  -5, 1);       // Another test case for real roots

    // Test integer_analysis function
    printf("\nTests for integer_analysis:\n");
    integer_analysis(10);   // Analyze 10: positive and even
    integer_analysis(-6);   // Analyze -6: negative and even
    integer_analysis(17);   // Analyze 17: positive and odd
    integer_analysis(0);    // Analyze 0: special case, zero and even
    integer_analysis(-187); // Analyze -187: negative and odd

    return 0;
}

/**
 * @brief Calculates distance using the equation: d = vt + (1/2) * at^2
 * 
 * @param acceleration The acceleration of the object
 * @param time The time over which the object moves
 * @param velocity The initial velocity of the object
 * @return Returns the calculated distance (printed within the function)
 */
float calculate_distance(float acceleration, float time, float velocity) {
    // Use the kinematic equation to calculate distance
    double result = ((velocity * time) + ((acceleration * (time * time)) / 2));
    printf("Using a = %.2f, t = %.2f and v = %.2f, d = %.2f\n", acceleration, time, velocity, result);
    return 0;
}

/**
 * @brief Solves and prints the real roots of a quadratic equation ax^2 + bx + c = 0
 * 
 * @param a Coefficient of x^2
 * @param b Coefficient of x
 * @param c Constant term
 * @return Returns 0 (roots are printed within the function)
 */
double print_real_roots(double a, double b, double c) {
    // Ensure the equation is quadratic (a should not be zero)
    if (a == 0) {
        printf("ERROR: a cannot be 0\n");
        return 0;
    }

    // Calculate the discriminant (b^2 - 4ac)
    double discriminant = (b * b - 4 * a * c);

    // Check if there are real roots (discriminant must be non-negative)
    if (discriminant < 0) { 
        printf("NO REAL ROOTS\n");     // No real solutions if discriminant is negative
        return 0;
    } else {
        // Calculate and print both roots
        double root1 = ((-b + sqrt(discriminant)) / (2 * a));
        double root2 = ((-b - sqrt(discriminant)) / (2 * a));
        printf("Result is %.3f, %.3f \n", root1, root2);
    }
    return 0;
}

/**
 * @brief Analyzes if an integer is positive, negative, zero, and whether it is even or odd
 * 
 * @param a The integer to be analyzed
 * @return Returns 0 (analysis is printed within the function)
 */
int integer_analysis(int a) {
    // Output the integer being analyzed
    printf("Analyzing integer: %d", a);

    // Check if the integer is positive, negative, or zero
    if (a > 0) {
        printf("\t Value is a positive integer.");
    } else if (a < 0) {
        printf("\t Value is a negative integer.");
    } else {
        printf("\t Value is zero.");
    }

    // Check if the integer is even or odd
    if (a % 2 == 0) {
        printf("\t Value is even.");
    } else {
        printf("\t Value is odd.");
    }

    // Print results and return
    printf("\n\n");
    return 0;
}

/**
 * @brief Prints the sum of two floating-point numbers
 * 
 * @param Numb1 First number
 * @param Numb2 Second number
 * @return Returns 0 (sum is printed within the function)
 */
float print_sum(float Numb1, float Numb2) { 
    // Calculate the sum of the two numbers
    float result = Numb1 + Numb2;
    
    // Print the result with two decimal precision
    printf("\tThe sum of %.2f and %.2f is %.2f\n", Numb1, Numb2, result);
    
    return 0;
}
