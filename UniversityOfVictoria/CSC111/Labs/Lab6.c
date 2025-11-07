#include <stdio.h>

void checkpoint1_tests(void);
void checkpoint2_tests(void);
void checkpoint3_tests(void);
void checkpoint4_tests(void);

void print_array(int array[], int length);
void print_alternating(int array[], int length);
void print_reverse(int array[], int length);
int get_min(int array[], int length);
int are_all_odd(int array[], int length);
int contains_odd(int array[], int length);
void multiply_by(int array[], int length, int n);
void PartIV_multiply_by(int *array, int length, int n);
void clamp(int array[], int length, int min_value, int max_value);

int main(void) {
    checkpoint1_tests();
    checkpoint2_tests();
    checkpoint3_tests();
    checkpoint4_tests();
}

void checkpoint1_tests(void) {
    printf("\n\tTests for CHECKPOINT 1\n");
    int arr1[6] = {5, 7, 6, 2, 8, 4};
    int arr2[3] = {99, 48, 52};
    int arr3[1] = {4};
    int arr4[0];
    int a1_length = 6, a2_length = 3, a3_length = 1, a4_length = 0;

    printf("\n\ttesting print_array:\n");
    print_array(arr1, a1_length);
    print_array(arr2, a2_length);
    print_array(arr3, a3_length);
    print_array(arr4, a4_length);
    
    printf("\n\ttesting print_alternating:\n");
    print_alternating(arr1, a1_length);
    print_alternating(arr2, a2_length);
    print_alternating(arr3, a3_length);
    print_alternating(arr4, a4_length);
    
    printf("\n\ttesting print_reverse:\n");
    print_reverse(arr1, a1_length);
    print_reverse(arr2, a2_length);
    print_reverse(arr3, a3_length);
    print_reverse(arr4, a4_length);
    
}

void checkpoint2_tests(void) {
    printf("\n\tTests for CHECKPOINT 2\n");
    int arr1[1] = {4};
    int arr2[3] = {99, 47, 51};
    int arr3[6] = {5, 7, 9, 11, 13, 14};
    int arr4[9] = {5, 7, 6, 2, 8, 1, 9, 3, 4};
    int arr5[0];
    int arr6[1] = {3};
    int a1_length = 1, a2_length = 3, a3_length = 6;
    int a4_length = 9, a5_length = 0, a6_length = 1;

    printf("\n\ttesting get_min:\n");
    printf("get_min([4]): %d\n", get_min(arr1, a1_length));
    printf("get_min([99, 47, 51]): %d\n", get_min(arr2, a2_length));
    printf("get_min([5, 7, 9, 11, 13, 14]): %d\n", get_min(arr3, a3_length));
    printf("get_min([5, 7, 6, 2, 8, 1, 5, 3, 4]): %d\n", get_min(arr4, a4_length));
    
    printf("\n\ttesting are_all_odd:\n");
    printf("are_all_odd with [4]: %d\n", are_all_odd(arr1, a1_length));
    printf("are_all_odd with [99, 47, 51): %d\n", are_all_odd(arr2, a2_length));
    printf("are_all_odd with [5, 7, 9, 11, 13, 14]: %d\n", are_all_odd(arr3, a3_length));
    printf("are_all_odd with [5, 7, 6, 2, 8, 1, 5, 3, 4): %d\n", are_all_odd(arr4, a4_length));
    printf("are_all_odd with []: %d\n", are_all_odd(arr5, a5_length));
    printf("are_all_odd with [3]: %d\n", are_all_odd(arr6, a6_length));
    
    int arr7[4] = {2, 4, 6, 8};
    int arr8[5] = {2, 4, 6, 8, 9};
    int a7_length = 4, a8_length = 5;
    printf("\n\ttesting contains_odd:\n");
    printf("contains_odd with [4]: %d\n", contains_odd(arr1, a1_length));
    printf("contains_odd with [99, 47, 51): %d\n", contains_odd(arr2, a2_length));
    printf("contains_odd with [5, 7, 9, 11, 13, 14]: %d\n", contains_odd(arr3, a3_length));
    printf("contains_odd with [5, 7, 6, 2, 8, 1, 5, 3, 4): %d\n", contains_odd(arr4, a4_length));
    printf("contains_odd with []: %d\n", contains_odd(arr5, a5_length));
    printf("contains_odd with [3]: %d\n", contains_odd(arr6, a6_length));
    printf("contains_odd with [2, 4, 6, 8]: %d\n", contains_odd(arr7, a7_length));
    printf("contains_odd with [2, 4, 6, 8, 9]: %d\n", contains_odd(arr8, a8_length));
    
}

void checkpoint3_tests(void) {
    printf("\n\tTests for CHECKPOINT 3\n");
    int arr1[1] = {4};
    int arr2[3] = {99, 47, 51};
    int arr3[6] = {5, 7, 9, 11, 13, 14};
    int arr4[9] = {5, 7, 6, 2, 8, 1, 9, 3, 4};
    int a1_length = 1, a2_length = 3, a3_length = 6, a4_length = 9;

    printf("\n\tTesting multiply_by:\n");
    int multiplier1 = 3, multiplier2 = 7;
    multiply_by(arr1, a1_length, multiplier1);
    multiply_by(arr2, a2_length, multiplier1);
    multiply_by(arr3, a3_length, multiplier2);
    multiply_by(arr4, a4_length, multiplier2);
    print_array(arr1, a1_length);
    print_array(arr2, a2_length);
    print_array(arr3, a3_length);
    print_array(arr4, a4_length);
    
    printf("\n\tTesting clamp:\n");    
    int arr5[10] = {8, 135, 76, 22, 2, 124, 40, 36, 101, 2};
    int a5_length = 10;
    printf("original array:\t\t\t");
    print_array(arr5, a5_length);
    
    clamp(arr5, a5_length, 5, 100);
    printf("after clamping with 5, 100:\t");
    print_array(arr5, a5_length);
    
    clamp(arr5, a5_length, 20, 50);
    printf("after clamping with 20, 50:\t");
    print_array(arr5, a5_length); 
}

void checkpoint4_tests(void){
   // Testing PartIV_multiply_by from Part IV: Arrays and Pointers Exercise 3
    printf("\n\tTesting PartIV_multiply_by\n");
    int arr1[1] = {4};
    int arr2[3] = {99, 47, 51};
    int arr3[6] = {5, 7, 9, 11, 13, 14};
    int arr4[9] = {5, 7, 6, 2, 8, 1, 9, 3, 4};
    int a1_length = 1, a2_length = 3, a3_length = 6, a4_length = 9;
    int multiplier1 = 3, multiplier2 = 7;
    
    PartIV_multiply_by(arr1, a1_length, multiplier1);
    PartIV_multiply_by(arr2, a2_length, multiplier1);
    PartIV_multiply_by(arr3, a3_length, multiplier2);
    PartIV_multiply_by(arr4, a4_length, multiplier2);
    print_array(arr1, a1_length);
    print_array(arr2, a2_length);
    print_array(arr3, a3_length);
    print_array(arr4, a4_length);
}

/** (Part I)
 * @brief Outputs the contents of the given array
 * @param array the array of integers to output
 * @param length the length of the given array
 */
void print_array(int array[], int length) {
    if (length > 0) {
        int i;
        printf("%d", array[0]);
        for (i = 1; i < length; i++) {
            printf(", %d", array[i]);
        }
        printf("\n");
    } else {
        printf("no output\n");
    }
}

/** (Part I)
 * @brief Outputs every second element in the given array
 * @param array the array of integers to output
 * @param length the length of the given array
 */
void print_alternating(int array[], int length) {
    for (int i = 0; i < length; i += 2) {
        printf("%d", array[i]);
        if (i <= length - 3) {
            printf(", ");
        }
    }
    printf("\n");
}

/** (Part I) 
 * @brief Outputs the contents of the array in reverse order (back to front)
 * @param array the array of integers to output
 * @param length the length of the given array
 */
void print_reverse(int array[], int length) {
    for (int i = length - 1; i >= 0; i--) {
        printf("%d", array[i]);
        if (i != 0) {
            printf(", ");
        }
        
    }
    printf("\n");
}

/** (Part II)
 * @brief Gets the minimum value found in the array
 * @param array the array of integers to output
 * @param length the length of the given array
 * @return the smallest value found in the array as an integer
 * @pre the array has at least one element (length > 0)
 */
int get_min(int array[], int length) {
    int min_value = array[0];
    for (int i = 1; i < length; i++) {
        if (array[i] < min_value) {
            min_value = array[i];
        }
    }
    return min_value;
}

/** (Part II)
 * @brief Determines if every value in the array is odd
 * @param array the array of integers to output
 * @param length the length of the given array
 * @return 1 if all elements are odd, 0 otherwise
 * @note if the array is empty 1 is returned
 *       (since no even values are found)
 */
int are_all_odd(int array[], int length) {
    for (int i = 0; i < length; i++) {
        if (array[i] % 2 == 0) {
            return 0; // Found an even number
        }
    }
    return 1; // All numbers are odd
}

/** (Part II)
 * @brief Determines if the array contains any odd values
 * @param array the array of integers to output
 * @param length the length of the given array
 * @return 1 if there is at least one odd integer, 0 otherwise
 * @note if the array is empty 0 is returned
 *       (since no odd value is found)
 */
int contains_odd(int array[], int length) {
    for (int i = 0; i < length; i++) {
        if (array[i] % 2 != 0) {
            return 1; // Found an odd number
        }
    }
    return 0; // No odd number found
}

/** (Part III)
 * @brief Modifies the array by multiplying each value by multiplier
 * @param array the array of integers to output
 * @param length the length of the given array
 * @param n the integer to multiple every element by
 */
void multiply_by(int array[], int length, int n) {
    for (int i = 0; i < length; i++) {
        array[i] *= n;
    }
}

/** (Part III)
 * @brief Updates the array so that all elements are >= min_value and <= max_value
 *        by modifying the value of all elements that are too small to be 
 *        min_value and all values that are too big to be max_value
 * @param array the array of integers to output
 * @param length the length of the given array
 * @param min_value the minimum value all elements must be as an integer
 * @param max_value the maximum value all elements must be as an integer
 */
void clamp(int array[], int length, int min_value, int max_value) {
    for (int i = 0; i < length; i++) {
        if (array[i] < min_value) {
            array[i] = min_value;
        } else if (array[i] > max_value) {
            array[i] = max_value;
        }
    }
}


/** (Part IV)
 * @brief Modifies the array by multiplying each value by multiplier
 *        using pointer arithmetic instead of array indexing
 * @param array the array of integers to output
 * @param length the length of the given array
 * @param n the integer to multiply every element by
 */
void PartIV_multiply_by(int *array, int length, int n) {
    int *ptr = array; // Pointer to traverse the array
    for (int i = 0; i < length; i++) {
        *ptr *= n;    // Multiply the value pointed to by ptr by n
        ptr++;        // Move the pointer to the next element
    }
}