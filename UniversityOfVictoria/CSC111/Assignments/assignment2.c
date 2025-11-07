/*
 * Brief:
 * This program performs various calculations and analyses, including:
 * 1. Performance analysis: Compares a student's score to the average and calculates the percentage difference.
 * 2. Median calculation: Finds the median of three integers.
 * 3. Speed conversion: Converts speed from miles per hour to kilometers per hour.
 * 4. Factor checking: Determines if one number is a factor of another.
 * 5. Cost after coverage: Calculates how much an employee pays after insurance coverage, based on different tiers.
 *
 *  @Author: Zac Matthias
 */
#include <stdio.h>

/**
 * @brief Prints an analysis of a student's final grade compared to the course average.
 * @param final_grade The final grade achieved by the student.
 * @param average_grade The average grade for all students in the course.
 */
void print_performance_analysis(double final_grade, double average_grade) {
    double difference = final_grade - average_grade;

    if (average_grade == 0) {
         printf("You achieved a grade %.2f%% above the average\n", final_grade);
    }
    if (difference >= 0) {
        // above
         printf("You achieved a grade %.2f%% above the average\n", difference);
    } else {
         // below
         printf("You were below the average by %.2f%%\n", -difference);
    }
    
}

/**
 * @brief Prints the median of three integer numbers.
 * @param a The first integer.
 * @param b The second integer.
 * @param c The third integer.
 */
void print_median(int a, int b, int c) {
    int temp;
    if (a > b) {
        temp = a;
        a = b;
        b = temp;
    }
    if (b > c) {
        temp = b;
        b = c;
        c = temp;
    }
    if (a > b) {
        temp = a;
        a = b;
        b = temp;
    }
    printf("%d\n", b);
}

/**
 * @brief Prints the speed of a trip in km/h given miles traveled and time in hours.
 * @param miles The distance traveled in miles.
 * @param hours The time spent traveling in hours.
 */
void print_in_km_per_hour(int miles, int hours) {
    double distance_km = miles * 1.6;
    double speed_kmh = distance_km / hours;
    printf("%.2fkm/h\n", speed_kmh);
}
/**
 * @brief Prints whether the first integer is a factor of the second integer.
 * @param a The first integer to check if it is a factor.
 * @param b The second integer to be divided.
 */
void is_factor_of(int a, int b) {
    if (b == 0) {
        if (a == 0) {
            printf("0 is a factor of 0\n");
        } else {
            printf("%d is a factor of 0\n", a);
        }
    } else {
        if (a  == 0) {
            printf("0 is not a factor of %d\n", b);
        } else if (b % a == 0) {
            printf( "%d is a factor of %d\n", a, b);
        } else {
            printf("%d is not a factor of %d\n", a, b);
        }
    }
}

/**
 * @brief Prints the cost of a dental visit after applying coverage based on the tier.
 * @param cost The total cost of the dental visit.
 * @param tier The coverage tier (1, 2, or 3).
 */
void print_cost_after_coverage(double cost, int tier) {
    double employee_cost = 0.0;
    if (cost < 250.0) {
        // Under $250.00 range
        if (tier == 1) {
            employee_cost = 0.0; // Fully covered
        } else if (tier == 2) {
            if (cost < 10) {
                employee_cost = cost;
            } else {
              employee_cost = 10.0; // Employee pays $10.00
            }
          
        } else if (tier == 3) {
           if (cost < 25) {
                employee_cost = cost;
            } else {
              employee_cost = 25.0; // Employee pays $25.00
            }
        }
    } else if (cost >= 250.0 && cost <= 500.0) {
        // $250.00 to $500.00 range
        if (tier == 1) {
            employee_cost = cost - 400.0; // $400.00 off
        } else if (tier == 2) {
            employee_cost = cost - 300.0; // $300.00 off
        } else if (tier == 3) {
            employee_cost = cost - 200.0; // $200.00 off
        }
        if (employee_cost < 0.0) {
            employee_cost = 0.0; // Employee cost should not be negative
        }
    } else {
        // Over $500.00 range
        if (tier == 1) {
            employee_cost = cost * 0.25; // 75% covered, employee pays 25%
        } else if (tier == 2) {
            employee_cost = cost * 0.40; // 60% covered, employee pays 40%
        } else if (tier == 3) {
            employee_cost = cost * 0.60; // 40% covered, employee pays 60%
        }
    }
    
    printf("$%.2f\n", employee_cost);
}
// Function to manually test the other functions
void manual_tester() {
    // Testing print_performance_analysis
    printf("Testing print_performance_analysis:\n");
    print_performance_analysis(75.78, 75.78);
    print_performance_analysis(80, 86);
    print_performance_analysis(70, 0);
    print_performance_analysis(81, 86);
    print_performance_analysis(96, 92);
    print_performance_analysis(82.5, 82.5);

    // Testing print_median
    printf("\nTesting print_median:\n");
    print_median(0, 0, 0);
    print_median(5, 5, 5);
    print_median(5, 5, 17);
    print_median(21, 21, 5);
    print_median(5, 17, 5);
    print_median(21, 5, 21);
    print_median(17, 5, 5);
    print_median(5, 21, 21);
    print_median(5, 17, 21);
    print_median(5, 21, 17);
    print_median(17, 5, 21);
    print_median(17, 21, 5);
    print_median(21, 5, 17);
    print_median(21, 17, 5);

    // Testing print_in_km_per_hour
    printf("\nTesting print_in_km_per_hour:\n");
    print_in_km_per_hour(5, 5);
    print_in_km_per_hour(10, 11);
    print_in_km_per_hour(-3, -4);
    print_in_km_per_hour(17, 17);
    print_in_km_per_hour(13, 12);
    print_in_km_per_hour(-11, -10);

    // Testing is_factor_of
    printf("\nTesting is_factor_of:\n");
    is_factor_of(5, 17);
    is_factor_of(5, 85);
    is_factor_of(85, 5);
    is_factor_of(5, 86);
    is_factor_of(0, 0);
    // Testing print_cost_after_coverage
    printf("\nTesting print_cost_after_coverage:\n");
    print_cost_after_coverage(7.5, 2);
    print_cost_after_coverage(7.5, 3);
    print_cost_after_coverage(249.99, 1);
    print_cost_after_coverage(249.99, 2);
    print_cost_after_coverage(300.0, 2);
}

int main() {
    manual_tester();
    return 0;
}