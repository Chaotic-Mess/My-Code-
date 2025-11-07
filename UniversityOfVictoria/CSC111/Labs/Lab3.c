/**
 * @brief Lab 3 program that focuses on designing and using functions that return values
 * @author Zac Matthias
 */
#include <stdio.h>

/**
 * @brief determines whether the given integer n is odd
 * @param n the integer to consider
 * @return an integer representing whether n is odd (1) or not (0)
 */
int is_odd(int n) {
    return n%2 != 0;
}

/**
 * @brief Analyzes two integers and prints which are odd
 * @param num1 The first integer to evaluate
 * @param num2 The second integer to evaluate
 * @return void (no return value)
 */
void odd_analysis(int num1, int num2) {
    if (is_odd(num1) && is_odd(num2)) {
        printf("Both %d and %d are odd\n", num1, num2);
    } else if (is_odd(num1)) {
        printf("Only %d is odd\n", num1);
    } else if (is_odd(num2)) {
        printf("Only %d is odd\n", num2);
    } else {
        printf("Neither %d nor %d is odd\n", num1, num2);
    }
}

/**
 * @brief Calculates the total tax for a bill
 * @param food_bill The total amount for food purchases
 * @param alcohol_bill The total amount for alcohol purchases
 * @return The total tax for the entire bill (GST + PST where applicable)
 */
double get_tax(double food_bill, double alcohol_bill) {
    const double GST_RATE = 0.05;
    const double PST_RATE = 0.10;
    
    double total_tax = 0.0;
    
    total_tax += (food_bill + alcohol_bill) * GST_RATE;  // Apply GST to both
    total_tax += alcohol_bill * PST_RATE;                // Apply PST to alcohol only
    
    return total_tax;
}

/**
 * @brief Calculates each person's share of the bill including tax
 * @param food_bill The total amount for food purchases (pre-tax)
 * @param alcohol_bill The total amount for alcohol purchases (pre-tax)
 * @param group_size The number of people in the group
 * @return The amount each person needs to pay (total amount divided by group size)
 */
double get_bill_share(double food_bill, double alcohol_bill, int group_size) {
    double total_bill = (food_bill + alcohol_bill + get_tax(food_bill, alcohol_bill));
    return total_bill / group_size;
}

int main(void) {
    // Testing odd_analysis
    printf("Testing odd_analysis:\n");
    odd_analysis(7, 11);  // Both odd
    odd_analysis(5, 2);   // Only 5 is odd
    odd_analysis(10, 11); // Only 11 is odd
    odd_analysis(6, 12);  // Neither is odd
    printf("\n");

    // Testing get_tax
    printf("Testing get_tax:\n");
    printf("Tax for $100 food, $0 alcohol: %.2f\n", get_tax(100.00, 0.00));    // GST only on food
    printf("Tax for $0 food, $100 alcohol: %.2f\n", get_tax(0.00, 100.00));    // GST + PST on alcohol
    printf("Tax for $68.75 food, $45.98 alcohol: %.2f\n", get_tax(68.75, 45.98));  // Mixed
    printf("\n");

    // Testing get_bill_share
    printf("Testing get_bill_share:\n");
    printf("Each person pays: %.2f\n", get_bill_share(28.75, 45.98, 3));  // Split among 3 people
    printf("Each person pays: %.2f\n", get_bill_share(18.93, 0, 2));      // Split among 2 people
    printf("\n");

    return 0;
}