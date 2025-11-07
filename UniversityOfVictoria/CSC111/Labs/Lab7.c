#include <stdio.h>

#define MAX_NUMS 100

// function definitions provided for you
void print_array(int array[], int len);
void print_2D_array(int rows, int cols, int array[rows][cols]);
void print_2D_ptrs(int rows, int cols, int* array[cols]);
void multiply_by(int array[], int len, int multiplier);

// functions for you to implement
void multiply_2D_by(int rows, int cols, int array[rows][cols], int multiplier);
void multiply_2D_ptrs_by(int rows, int cols, int* array[cols], int multiplier);
int get_max_2D(int rows, int cols, int array[rows][cols]);
int get_max_ptrs(int rows, int cols, int array[rows][cols]);
void get_min_position(int rows, int cols, int array[rows][cols], int* row, int* col);
int* get_min_location_ptr(int rows, int cols, int array[rows][cols]);
int read_into_array(FILE* file_handle, int array[]);
void write_array_to_file(FILE* file_handle, int array[], int length);

int main(void){
    int a3[3] = {6, 5, 4};
    int b3[3] = {2, 1, 5};
    int a4[4] = {1, 6, 0, 2};
    int b4[4] = {2, 3, -6, 2};
    int c4[4] = {1, 2, -3, 5};
    int* ptrs_array1[]= {a3, b3};
    int* ptrs_array2[]= {a4, b4, c4};

    int array_2Da[6][5] = {{37, 83,   75,  23,   71},
                           {3,  14,   15,   3,    2},
                           {65,  3,   58,  79,    3},
                           {2,  38,   86,  26,    4},
                           {3,   3,    8,   3,    2},
                           {88, 10,   36,  11,    6}};

    int array_2Db[2][4] = {{3,  3,  8,  3},
                          {88, 10, 36, 11}};

    // Sample tests for provided functions:
    
    printf("printing a3:\n");
    print_array(a3, 3);
    printf("multiplying by 2 and printing again:\n");
    multiply_by(a3, 3, 2);
    print_array(a3, 3);
    printf("\nprinting int* array ptrs_array1:\n");
    print_2D_ptrs(2, 3, ptrs_array1);
    printf("\nprinting int* array ptrs_array2:\n");
    print_2D_ptrs(3, 4, ptrs_array2);
    printf("\nprinting array_2Da:\n");
    print_2D_array(6, 5, array_2Da);
    printf("\nprinting array_2Db:\n");
    print_2D_array(2, 4, array_2Db);
    
    
    // CHECKPOINT 1 TESTS
    printf("\nmultiplying array_2Da by 3\n");
    multiply_2D_by(6, 5, array_2Da, 3);
    print_2D_array(6, 5, array_2Da);

    printf("\nmultiplying array_2Db by 10\n");
    multiply_2D_by(2, 4, array_2Db, 10);
    print_2D_array(2, 4, array_2Db);

    printf("\nmultiplying ptrs_array1 by 7\n");
    multiply_2D_ptrs_by(2, 3, ptrs_array1, 7);
    print_2D_ptrs(2, 3, ptrs_array1);

    printf("\nmultiplying ptrs_array2 by 5\n");
    multiply_2D_ptrs_by(3, 4, ptrs_array2, 5);
    print_2D_ptrs(3, 4, ptrs_array2);
    
    // CHECKPOINT 2 TESTS:
    int max;
    // Note: if you do not comment out the multiply_by function calls above,
    // the values will be different from in the original declarations
    printf("\ngetting max value in array_2Da\n");
    max = get_max_2D(6, 5, array_2Da);
    printf("max found: %d\n", max);

    printf("\ngetting max value in array_2Db\n");
    max = get_max_2D(2, 4, array_2Db);
    printf("max found: %d\n", max);

    printf("\ngetting max value in array_2Da with pointer arithmetic\n");
    max = get_max_ptrs(6, 5, array_2Da);
    printf("max found: %d\n", max);

    printf("\ngetting max value in array_2db with pointer arithmetic\n");
    max = get_max_ptrs(2, 4, array_2Db);
    printf("max found: %d\n", max);
    
    
    // CHECKPOINT 3 TESTS:
    int min_rnum = 0;
    int min_cnum = 0;
    get_min_position(6, 5, array_2Da, &min_rnum, &min_cnum);
    printf("\nMin value in array_2Da: %d\n", array_2Da[min_rnum][min_cnum]);
    get_min_position(2, 4, array_2Db, &min_rnum, &min_cnum);
    printf("Min value in array_2Db: %d\n", array_2Db[min_rnum][min_cnum]);

    int *min_val_ptr;
    min_val_ptr = get_min_location_ptr(6, 5, array_2Da);
    printf("\nMin value in array_2Da returning ptr: %d\n", *min_val_ptr);

    min_val_ptr = get_min_location_ptr(2, 4, array_2Db);
    printf("Min value in array_2Db returning ptr: %d\n", *min_val_ptr);
    

    // CHECKPOINT 4 TESTS:
    FILE* file_handle1 = fopen("input.txt", "r");
    if (file_handle1 == NULL) {
        printf("could not open file\n");
        return 1; // non-zero return from main indicates an error
    } else {
        int array[MAX_NUMS] = {0}; // array initialized to 0s
        int num_elements;
        num_elements = read_into_array(file_handle1, array);
        fclose(file_handle1);
        printf("Printing contents read from file:\n");
        print_array(array, num_elements);

        FILE* file_handle2 = fopen("output.txt", "w");
        if (file_handle1 == NULL) {
            printf("could not open file\n");
            return 1; // non-zero return from main indicates an error
        } else {
            write_array_to_file(file_handle2 ,array, num_elements);
            fclose(file_handle2);
        }
    }

    // TODO: 
    // 1) Open a file named output.txt for writing
    // 2) Verify the file is opened 
    // 3) Call write_array_to_file with your file pointer, an array, and its size
    // 4) Close the file
    // 4) Compile and run the program
    // 5) Refresh the File Browser to the left and open up the output.txt file
    
    
    return 0;
}

/**
 * @brief Prints the elements in the array separated by commas
 * @param array the array of integer elements
 * @param length the number of elements in the array
 */
void print_array (int array[], int length) {
    int i = 0;
    printf("[");
    for (i = 0; i < length; i++){
        printf("%3d", array[i]);
        if (i != length-1) {
            printf(", ");
        }
    }
    printf("]\n");
}

/**
 * @brief Prints the elements in the given 2D array in a grid of rows and columns
 * @param table the 2d array of elements with dimensions num_row by num_cols
 * @param num_rows the number of rows in the 2D array
 * @param int the number columns in the 2D array (>=0 and <=NUM_COLS)
 */
void print_2D_array(int rows, int cols, int array[rows][cols]) {
    int row = 0;
    for (row = 0; row < rows; row++){
        print_array(array[row], cols);
    }
}

/**
 * @brief Prints the elements pointed to by the array pointer
 * @param array an array of num_rows arrays that are each num_cols long
 * @param num_rows the number of rows in the array
 * @param num_cols the number columns, always >=0
 */
void print_2D_ptrs(int rows, int cols, int* array[cols]) {
    int i;
    int* ptr = &array[0][0];
    for (i = 0; i < rows; i++){
        print_array(ptr+(i*cols), cols);
    }
}

/**
 * @brief Multiply every element in the 2D array by muliplier
 * @param array the array of integer elements
 * @param length the number of elements in the array
 * @param multiplier each element is multiplied by this value
 */
void multiply_by(int array[], int length, int multiplier){
    int i;
    for (i = 0; i < length; i++){
        array[i] *= multiplier;
    }
}

/**
 * @brief multiplies every value in the 2D array by the multiplier
 * @param rows the number of rows in the 2D array
 * @param cols the number of cols on each row in the 2D array
 * @param array the 2D array of integers
 * @param multiplier the value to multiply each array value by
 */
void multiply_2D_by(int rows, int cols, int array[rows][cols], int multiplier) {
    for (int i = 0; i < rows; i++) {
        multiply_by(array[i], cols, multiplier);
    }
}

/**
 * @brief multiplies every value in the 2D array by the multiplier
 * @param rows the number of rows in the 2D array
 * @param cols the number of cols on each row in the 2D array
 * @param array an array of int pointers (pointing to arrays)
 * @param multiplier the value to multiply each array value by
 */
void multiply_2D_ptrs_by(int rows, int cols, int* array[rows], int multiplier) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(array[i] + j) *= multiplier; 
        }
    }
}

/**
 * @brief Returns the maximum value found in the 2D array
 * @param rows the number of rows in the 2D array
 * @param cols the number of cols on each row in the 2D array
 * @param array the 2D array of integers
 * @return the maximum value found in the array
 */
int get_max_2D(int rows, int cols, int array[rows][cols]) {
    int max = array[0][0];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (array[i][j] > max) {
                max = array[i][j]; 
            }
        }
    }
    return max; 

}

/**
 * @brief gets the maximum value in the array using pointer arithmetic
 * @param rows the number of rows in the 2D array
 * @param cols the number of colums per row in the 2D array
 * @param array the 2D array of integers
 * @return the maximum value found in the array
 */
int get_max_ptrs(int rows, int cols, int array[rows][cols]) {
  int* ptr = &array[0][0];
    int max = *ptr; 

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (*(ptr + (i * cols) + j) > max) { 
                max = *(ptr + (i * cols) + j);
            }
        }
    }

    return max; 
}

/**
 * @brief Identifies the row and column where the minimum value is located
 * @param rows the number of rows in the 2D array
 * @param cols the number of colums per row in the 2D array
 * @param array the 2D array of integers
 * @param row a pointer to the integer storing the row number of the min value
 * @param col a pointer to the integer storing the column number of the min value
 */
void get_min_position(int rows, int cols, int array[rows][cols], int* row, int* col) {
    int min = array[0][0]; 

    *row = 0;
    *col = 0; // Initialize row and col pointers to (0, 0) location

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (array[i][j] < min) { 
                min = array[i][j];
                *row = i; 
                *col = j;
            }
        }
    }
}

/**
 * @brief Returns a pointer to the minimum value in the array
 * @param rows the number of rows in the 2D array
 * @param cols the number of colums per row in the 2D array
 * @param array the 2D array of integers
 * @return a pointer to the memory address storing the minimum value in the array
 */
int* get_min_location_ptr(int rows, int cols, int array[rows][cols]) {
    int* ptr = &array[0][0]; // Points to the location of the first element
    int* min_ptr = ptr; // Initialize min_ptr to point to the first element

    for (int i = 0; i < rows * cols; i++) {
        if (*(ptr + i) < *min_ptr) {
            min_ptr = (ptr + i); 
        }
    }

    return min_ptr; 
}

/**
 * @brief Reads the contents of the file and puts them into the given array
 *        At most MAX_NUMS values are copied from the input file
 * @param file_handle a pointer to a valid file opened for writing
 * @param array the array of integers to write to the file
 * @return the number of elements successfully placed in the array
 */
int read_into_array(FILE *file_handle, int array[]) {
    int count = 0;
    while (count < MAX_NUMS && fscanf(file_handle, "%d", &array[count]) == 1) {
        count++;
    }
    return count; 
}

/**
 * @brief Writes the contents of the array into the file
 *        one integer per line.
 * @param file_handle a pointer to a valid file opened for writing
 * @param array the array of integers to write to the file
 * @param length the number of elements in the array
 */
void write_array_to_file(FILE *file_handle, int array[], int length) {
    int i;
    for (i = 0; i < length; i++) {
        fprintf(file_handle, "%d\n", array[i]);
    }
}