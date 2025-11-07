/**
 * @Brief: This program contains multiple functions to perform mathematical operations such as checking for factors, counting factors, determining prime numbers, calculating factorials, computing combinations (n choose r), and summing Fibonacci numbers.
 * @Author: Zac Matthias
 */

#include <stdio.h>

/**
 * @Brief: Checks if n1 is a factor of n2.
 * @param n1: The potential factor.
 * @param n2: The number to be checked.
 * @return: 1 if n1 is a factor of n2, 0 otherwise.
 */
int is_factor(int n1, int n2) {
    
    if (n1 == 0 && n2 == 0) {
        return 1; // true, both zero
    }

    if (n1 == 0) {
        return 0; // Zero can't be a factor of any number (except 0, handled below)
    }
    
    if (n2 % n1 == 0) {
        return 1; // true
    } else {
        return 0; // false
    }
}


/**
 * @Brief: Counts the number of factors of a given number.
 * @param number_to_check: The number for which factors are counted.
 * @return: The number of factors.
 */
int num_factors(int number_to_check) {
    // Loop through all numbers from 1 to number_to_check and check if they are factors
    // Note to self: C loops are diffrent from lua loops, you need to use loop_count+=1 instead of +1 to continue the loop count (loop_count is variable, i and _ still work)

    int current_count = 0;

    for (int loop_count = 1; loop_count <= number_to_check; loop_count+=1) {
        if (is_factor(loop_count, number_to_check)) {
            current_count += 1; // Increase return count
        }
    }
    return current_count;
}

/**
 * @Brief: Checks if a number is prime.
 * @param potential_prime_numb: The number to be checked for primality.
 * @return: 1 if the number is prime, 0 otherwise.
 */
int is_prime(int potential_prime_numb) {
 // Prime numbs only has 2 factors. 1 and itself
    return (num_factors(potential_prime_numb) == 2); // A prime number has exactly two factors
}

/**
 * @Brief: Calculates the factorial of a number.
 * @param factorial_number: The number for which the factorial is calculated.
 * @return: The factorial of the given number.
 */
int factorial(int factorial_number) {
    // Multi. result by every number from 1 to factorial_number
    int result = 1;

    for (int loop_count = 1; loop_count <= factorial_number; loop_count+=1) {
        result *= loop_count;
    }
    return result;
}

/**
 * @Brief: Calculates the binomial coefficient (n choose r).
 * @param n: The total number of items.
 * @param r: The number of items to choose.
 * @return: The result of n choose r.
 */
int n_choose_r(int n, int r) {
 /*
      Note to self:
      C(n,r)= (n! / r!×(n−r)!)
      e.x.: n_choose_r(4, 2) calculates as 4! / (2! * 2!) = 24 / 4 = 6.
 */
    if (r == 0 || r == n) {
        return 1; // Special cases: C(n, 0) = C(n, n) = 1
    }
    return factorial(n) / (factorial(r) * factorial(n - r));
}

/**
 * @Brief: Sums the first n Fibonacci numbers.
 * @param n: The number of Fibonacci terms to sum.
 * @return: The sum of the first n Fibonacci numbers.
 */
int sum_fibonacci(int n) {
/*
        Note to self:
        initialize the first two Fibonacci numbers, 1 and 1, and start the sum at 2 (since those two numbers are the first in the sequence).
        then calculate the next Fibonacci number in a loop, updating first and second as we go, and add each new number to sum.
        The loop runs for n-2 iterations, because the first two numbers are already added.
    
    */
    if (n == 0) return 0; // Special case if n == 0
    if (n == 1) return 1; // Special case if n == 1
    
    int first = 1, second = 1, next, sum = 2;

    for (int i = 3; i <= n; i++) {
        next = first + second;
        sum += next;
        first = second;
        second = next;
    }

    return sum;
}

int main(){
    // Test is_factor
    printf("Factor of n1 (\"%d\") by n2 (\"%d\") return as Result [True (1) or False (0)] %d\n", 1,1,is_factor(1, 1));
    printf("Factor of n1 (\"%d\") by n2 (\"%d\") return as Result [True (1) or False (0)] %d\n", 1,12,is_factor(1, 12));
    printf("Factor of n1 (\"%d\") by n2 (\"%d\") return as Result [True (1) or False (0)] %d\n", 1,16,is_factor(1, 16));
    printf("Factor of n1 (\"%d\") by n2 (\"%d\") return as Result [True (1) or False (0)] %d\n", 0,0,is_factor(0, 0));
    printf("\n\n");

    // Test num_factors
    printf("There are %d factors in %d\n", num_factors(16), 16);
    printf("There are %d factors in %d\n", num_factors(999), 999);
    printf("\n\n");

    // Test is_prime
    printf("The number of factors in %d is %d\n", 1, num_factors(1));
    printf("The number of factors in %d is %d\n", 2, num_factors(2));
    printf("The number of factors in %d is %d\n", 3, num_factors(3));
    printf("The number of factors in %d is %d\n", 4, num_factors(4));
    printf("The number of factors in %d is %d\n", 5, num_factors(5));
    printf("The number of factors in %d is %d\n", 6, num_factors(6));
    printf("The number of factors in %d is %d\n", 7, num_factors(7));
    printf("\n\n");
        
    // Test factorial
    printf("Factorial of %d is: %d\n", 5, factorial(5));
    printf("\n\n");
        
    // Test n_choose_r
    printf("n_choose_r of %d and %d is %d\n", 4, 2, n_choose_r(4,2));
    printf("\n\n");
        
    // Test sum_fibonacci 
    printf("%d\n", sum_fibonacci(7));
    printf("\n\n");
    
    return 0;
}
