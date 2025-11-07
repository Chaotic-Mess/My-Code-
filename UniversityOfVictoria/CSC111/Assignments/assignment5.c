/**
 * @Brief This code contains functions that perform various operations 
 *        on arrays, such as calculating the sum of squares, adding values to elements,
 *        filtering elements based on conditions, checking for factors, and counting 
 *        occurrences of specific conditions across arrays. 
 * 
 * @author Zac Matthias
 */

#include <stdio.h>

// Function prototypes
void print_array(int arr[], int size);
int sum_squares(int arr[], int size);
void add_to_all(int arr[], int size, int n);
int get_all_below(int input[], int result[], int size, int threshold);
int keep_odds(int arr[], int size);
int are_all_above(int arr[], int size, int threshold);
int is_factor(int a, int b);
int does_contain_factors_of(int arr[], int element_length, int n);
int count_if_contains_factors_of(int arr1[], int arr2[], int element_length1, int element_length2);


int main(void) {
    // Test for sum_squares
    int arr1[] = {2, 4, 3};
    int arr2[] = {1, 1, 1, 1};
    int arr3[] = {0, 0, 0};
    int arr4[] = {-2, -4, -3};
    int arr5[] = {10, 20, 30};
    
    printf("\n\tsum_squares tests:\n");
    printf("%d\n", sum_squares(arr1, 3)); // 29
    printf("%d\n", sum_squares(arr2, 4)); // 4
    printf("%d\n", sum_squares(arr3, 3)); // 0
    printf("%d\n", sum_squares(arr4, 3)); // 29
    printf("%d\n", sum_squares(arr5, 3)); // 1400

    // Test for add_to_all
    int arr6[] = {8, 5, 1, 2};
    int arr7[] = {0, 0, 0};
    int arr8[] = {-1, -2, -3};
    int arr9[] = {100, 200, 300};
    int arr10[] = {5, 10, 15};
    
    printf("\n\tadd_to_all tests:\n");
    add_to_all(arr6, 4, 4); // {12, 9, 5, 6}
    add_to_all(arr7, 3, 10); // {10, 10, 10}
    add_to_all(arr8, 3, 5); // {4, 3, 2}
    add_to_all(arr9, 3, -100); // {0, 100, 200}
    add_to_all(arr10, 3, 7); // {12, 17, 22}

    // Print modified arrays
    printf("{%d, %d, %d, %d}\n", arr6[0], arr6[1], arr6[2], arr6[3]);
    printf("{%d, %d, %d}\n", arr7[0], arr7[1], arr7[2]);
    printf("{%d, %d, %d}\n", arr8[0], arr8[1], arr8[2]);
    printf("{%d, %d, %d}\n", arr9[0], arr9[1], arr9[2]);
    printf("{%d, %d, %d}\n", arr10[0], arr10[1], arr10[2]);

    // Test for get_all_below
    int result1[5], result2[3];
    int input1[] = {2, 3, 4, 6, 1};
    int input2[] = {0, 0, 0};
    
    printf("\n\tget_all_below tests:\n");
    printf("%d\n", get_all_below(input1, result1, 5, 4)); // 3
    printf("%d\n", get_all_below(input2, result2, 3, 1)); // 3
    printf("%d\n", get_all_below(input1, result1, 5, 2)); // 1
    printf("%d\n", get_all_below(input1, result1, 5, 10)); // 5
    printf("%d\n", get_all_below(input1, result1, 5, 0)); // 0

    // Test for keep_odds
    int odd_arr1[] = {2, 3, 5, 6, 1};
    int odd_arr2[] = {4, 6, 8, 10};
    int odd_arr3[] = {1, 3, 5};
    int odd_arr4[] = {7, 7, 7};
    int odd_arr5[] = {0, 0, 1};
    
    printf("\n\tkeep_odds tests:\n");
    printf("%d\n", keep_odds(odd_arr1, 5)); // 3
    printf("%d\n", keep_odds(odd_arr2, 4)); // 0
    printf("%d\n", keep_odds(odd_arr3, 3)); // 3
    printf("%d\n", keep_odds(odd_arr4, 3)); // 3
    printf("%d\n", keep_odds(odd_arr5, 3)); // 1

    // Test for are_all_above
    int check_arr1[] = {10, 20, 30};
    int check_arr2[] = {5, 10, 15};
    int check_arr3[] = {0, 0, 0};
    int check_arr4[] = {-1, -2, -3};
    int check_arr5[] = {100, 200, 300};
    
    printf("\n\tare_all_above tests:\n");
    printf("%d\n", are_all_above(check_arr1, 3, 5)); // 1
    printf("%d\n", are_all_above(check_arr2, 3, 5)); // 0
    printf("%d\n", are_all_above(check_arr3, 3, 1)); // 0
    printf("%d\n", are_all_above(check_arr4, 3, -5)); // 1
    printf("%d\n", are_all_above(check_arr5, 3, 50)); // 1

    // Test for does_contain_factors_of
    int factor_arr1[] = {9, 7, 18, 12, 21};
    int factor_arr2[] = {12, 5, 14, 3, 31};
    
    printf("\n\tdoes_contain_factors_of tests:\n");
    printf("%d\n", does_contain_factors_of(factor_arr1, 5, 3)); // 1
    printf("%d\n", does_contain_factors_of(factor_arr1, 5, 4)); // 0
    printf("%d\n", does_contain_factors_of(factor_arr1, 5, 7)); // 1
    printf("%d\n", does_contain_factors_of(factor_arr2, 5, 12)); // 1
    printf("%d\n", does_contain_factors_of(factor_arr2, 5, 2)); // 0

    // Test for count_if_contains_factors_of
    int count_arr1[] = {9, 7, 18, 12, 21};
    int count_arr2[] = {12, 5, 14, 3, 31};
    int count_arr3[] = {1, 2, 3};
    int count_arr4[] = {7, 14, 21};
    int count_arr5[] = {6, 12, 18};
    
    printf("\n\tcount_if_contains_factors_of tests:\n");
    printf("%d\n", count_if_contains_factors_of(count_arr1, count_arr2, 5, 5)); // 2
    printf("%d\n", count_if_contains_factors_of(count_arr1, count_arr3, 5, 3)); // 1
    printf("%d\n", count_if_contains_factors_of(count_arr1, count_arr4, 5, 3)); // 2
    printf("%d\n", count_if_contains_factors_of(count_arr1, count_arr5, 5, 3)); // 3
    printf("%d\n", count_if_contains_factors_of(count_arr2, count_arr5, 5, 3)); // 2

    return 0;
}

/**
 * @brief Checks if the first value is a factor of the second value.
 * 
 * @param a The potential factor.
 * @param b The value to check against.
 * @return 1 if a is a factor of b, 0 otherwise.
 */
int is_factor(int a, int b) {
    if (a == 0) {
        return 0;  // If 'a' is zero, it can't be a factor
    }
    return b % a == 0;
}

/**
 * @brief Calculates the sum of squares of all elements in the array.
 * 
 * @param arr The array of integers.
 * @param element_length The number of elements in the array.
 * @return The sum of squares of the array elements.
 */
int sum_squares(int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i] * arr[i];
    }
    return sum;
}


/**
 * @brief Adds a given value to all elements in the array.
 * 
 * @param arr The array of integers.
 * @param element_length The number of elements in the array.
 * @param value The integer value to add to each element.
 */
void add_to_all(int arr[], int size, int n) {
    for (int i = 0; i < size; i++) {
        arr[i] += n;
    }
}

/**
 * @brief Copies all elements in the input array below a threshold to the result array.
 * 
 * @param input_arr The input array of integers.
 * @param result_arr The result array to store elements below the threshold.
 * @param element_length The number of elements in the arrays.
 * @param threshold The threshold value.
 * @return The number of elements copied to the result array.
 */
int get_all_below(int input[], int result[], int size, int threshold) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (input[i] < threshold) {
            result[count++] = input[i];
        }
    }
    return count;
}

/**
 * @brief Filters the array to keep only odd values, shifting them to the front.
 * 
 * @param arr The array of integers.
 * @param element_length The number of elements in the array.
 * @return The number of odd elements in the array.
 */
int keep_odds(int arr[], int size) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] % 2 != 0) {
            arr[count++] = arr[i];
        }
    }
    return count;
}

/**
 * @brief Checks if all elements in the array are greater than the given threshold.
 * 
 * @param arr The array of integers.
 * @param element_length The number of elements in the array.
 * @param threshold The threshold value.
 * @return 1 if all elements are above the threshold, 0 otherwise.
 */
int are_all_above(int arr[], int size, int threshold) {
    for (int i = 0; i < size; i++) {
        if (arr[i] <= threshold) {
            return 0;
        }
    }
    return 1;
}


/**
 * @brief Checks if any element in the array has the given value as a factor.
 * 
 * @param arr The array of integers.
 * @param element_length The number of elements in the array.
 * @param n The integer value to check as a factor.
 * @return 1 if the n is a factor of any element, 0 otherwise.
 */
int does_contain_factors_of(int arr[], int element_length, int n) {
    if (n == 0) {
        return 0;  // Prevent division by zero
    }
    for (int i = 0; i < element_length; i++) {
        if (is_factor(n, arr[i])) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Counts the number of elements in the second array that are factors of any elements in the first array.
 * 
 * @param arr1 The first array of integers.
 * @param element_length1 The number of elements in the first array.
 * @param arr2 The second array of integers.
 * @param element_length2 The number of elements in the second array.
 * @return The count of elements in the second array that are factors of elements in the first array.
 */
int count_if_contains_factors_of(int arr1[], int arr2[], int element_length1, int element_length2) {
    int count = 0;
    for (int i = 0; i < element_length2; i++) {
        if (does_contain_factors_of(arr1, element_length1, arr2[i])) {
            count++;
        }
    }
    return count;
}