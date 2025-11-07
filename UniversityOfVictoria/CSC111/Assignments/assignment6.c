/** 
 * This program processes weather data from a file, storing dates, station IDs, and temperatures.
 * It includes functions to retrieve the highest temperature, filter high temperatures, compute 
 * statistics for specific weather stations, and calculate the average temperature before a given date.
 * The main function runs test cases to validate each function’s accuracy.
 * 
 * @author Zac Matthias
 */


#include <stdio.h>

#define MAX_LINES_INPUT 200   
#define ROW_SIZE 4

int read_data(FILE *file, int data_array[MAX_LINES_INPUT][ROW_SIZE], double temp_array[MAX_LINES_INPUT]);
double get_highest_temp(const double temp_array[], int length);
int filter_high_temps(const int input_2D[][ROW_SIZE], const double temp_1D[], int num_rows, int result_2D[][ROW_SIZE]);
int get_station_stats(const int input_2D[][ROW_SIZE], const double temp_1D[], int num_rows, double result_1D[3], int station_number);
int is_before(int first_year, int first_month, int first_day, int second_year, int second_month, int second_day);
int get_avg_temp_before_date(const int input_2D[][ROW_SIZE], const double temp_1D[], int num_rows, int year, int month, int day, double *average);

void TEST_read_data();
void TEST_get_highest_temp();
void TEST_filter_high_temps();
void TEST_get_station_stats();
void TEST_is_before();
void TEST_get_avg_temp_before_date();

int main(void) {
   // TESTING READ DATA
    TEST_read_data();
    TEST_get_highest_temp();
    TEST_filter_high_temps();
    TEST_get_station_stats();
    TEST_is_before();
    TEST_get_avg_temp_before_date();
    return 0;
}

/**
 * @brief Reads data from a file and stores date and station info in a 2D array,
 *        while temperatures are stored in a separate array.
 * @param input_file Pointer to the file to read from.
 * @param data 2D array to store year, month, day, and station values.
 * @param temperatures Array to store temperature values.
 * @return Number of rows read successfully, or 0 if an error occurs.
 */    
int read_data(FILE *file, int data_array[MAX_LINES_INPUT][ROW_SIZE], double temp_array[MAX_LINES_INPUT]) {
    int year, month, day, hour, minute, station;
    double temperature;
    int rows = 0;

    while (rows < MAX_LINES_INPUT && fscanf(file, "%d %d %d %d %d %d %lf", &year, &month, &day, &hour, &minute, &station, &temperature) == 7) {
        data_array[rows][0] = year;
        data_array[rows][1] = month;
        data_array[rows][2] = day;
        data_array[rows][3] = station;
        
        temp_array[rows] = temperature;
        
        rows++;
    }

    return rows;
}

/**
 * @brief Retrieves the highest temperature from an array of temperatures.
 * 
 * @param temperatures Array of temperature readings.
 * @param length The number of elements in the temperatures array.
 * @return double The highest temperature found in the array.
 */
double get_highest_temp(const double temperatures[], int length) {
    double highest_temp = temperatures[0];
    int i = 1;

    while (i < length) {
        if (temperatures[i] > highest_temp) {
            highest_temp = temperatures[i];
        }
        i++;
    }
    return highest_temp;
}
/**
 * @brief Filters high temperatures from the input data.
 * 
 * @param input_2D 2D array containing weather data.
 * @param temperatures Array of temperature readings.
 * @param rows Number of rows in the input data.
 * @param result_2D 2D array to store the filtered results.
 * @return int The number of rows that met the filtering criteria.
 */
int filter_high_temps(const int input_2D[MAX_LINES_INPUT][ROW_SIZE], const double temperatures[], 
                      int rows, int result_2D[MAX_LINES_INPUT][ROW_SIZE]) {
    double highest_temp = get_highest_temp(temperatures, rows);
    int result_count = 0;
    int i = 0;

    while (i < rows) {
        if (temperatures[i] >= highest_temp - 0.1) {
            result_2D[result_count][0] = input_2D[i][0];
            result_2D[result_count][1] = input_2D[i][1];
            result_2D[result_count][2] = input_2D[i][2];
            result_2D[result_count][3] = input_2D[i][3];
            result_count++;
        }
        i++;
    }
    return result_count;
}

/**
 * @brief Gets statistics for a specific weather station.
 * 
 * @param input_2D 2D array containing weather data.
 * @param temperatures Array of temperature readings.
 * @param rows Number of rows in the input data.
 * @param stats Array to store min, max, and average temperatures for the station.
 * @param station_number The station number to filter statistics by.
 * @return int 1 if data was found for the station, 0 otherwise.
 */
int get_station_stats(const int input_2D[MAX_LINES_INPUT][ROW_SIZE], const double temperatures[], 
                      int rows, double stats[3], int station_number) {
    int found = 0;
    double min_temp = temperatures[0], max_temp = temperatures[0], sum_temp = 0.0;
    int temp_count = 0;
    int i = 0;

    while (i < rows) {
        if (input_2D[i][3] == station_number) {
            double temp = temperatures[i];
            if (!found || temp < min_temp) min_temp = temp;
            if (!found || temp > max_temp) max_temp = temp;
            sum_temp += temp;
            temp_count++;
            found = 1;
        }
        i++;
    }

    if (found) {
        stats[0] = min_temp;
        stats[1] = max_temp;
        stats[2] = sum_temp / temp_count;
    }

    return found;
}

/**
 * @brief Checks if the first date is before the second date.
 * 
 * @param first_year The year of the first date.
 * @param first_month The month of the first date.
 * @param first_day The day of the first date.
 * @param second_year The year of the second date.
 * @param second_month The month of the second date.
 * @param second_day The day of the second date.
 * @return int 1 if the first date is before the second date, 0 otherwise.
 */
int is_before(int first_year, int first_month, int first_day, int second_year, int second_month, int second_day) {
 if (first_year < second_year) { return 1; } 
 else if (first_year == second_year) {
        if (first_month < second_month) { return 1; }
        else if (first_month == second_month) {
            if (first_day < second_day) { return 1; } }
    }
    return 0; 
}
/**
 * @brief Calculates the average temperature before a specified date.
 * 
 * @param input_2D 2D array containing weather data.
 * @param temperatures Array of temperature readings.
 * @param rows Number of rows in the input data.
 * @param year The year to compare against.
 * @param month The month to compare against.
 * @param day The day to compare against.
 * @param avg_temp Pointer to store the resulting average temperature.
 * @return int 1 if average temperature was calculated, 0 otherwise.
 */
int get_avg_temp_before_date(const int input_2D[MAX_LINES_INPUT][ROW_SIZE], const double temperatures[], int rows, int year, int month, int day, double *avg_temp) {
    double sum_temp = 0.0;
    int count = 0;
    int i = 0;
    
    while (i < rows) {
        if (is_before(input_2D[i][0], input_2D[i][1], input_2D[i][2], year, month, day)) {
            sum_temp += temperatures[i];
            count++;
        }
        i++;
    }

    if (count > 0) {
        *avg_temp = sum_temp / count;
        return 1;
    }

    return 0;
}


//////  TESTS
//////  TESTS
//////  TESTS
//////  TESTS
//////  TESTS

void TEST_read_data() {
    printf("\t\t \n\n TESTING READ_DATA \t\t \n\n");
    int data[MAX_LINES_INPUT][ROW_SIZE];
    double temperatures[MAX_LINES_INPUT];

    // I made 5 test files 
    const char *test_files[] = {
        "test_data_1.txt", // Normal data input
        "test_data_2.txt", // Empty file
        "test_data_3.txt", // More rows than MAX_LINES_INPUT
        "test_data_4.txt", // Incorrectly formatted file
        "test_data_5.txt"  // Valid data with mixed formats
    };

    int num_tests = sizeof(test_files) / sizeof(test_files[0]);
    for (int i = 0; i < num_tests; i++) {
        FILE *file = fopen(test_files[i], "r");
        if (file) {
            int rows_read = read_data(file, data, temperatures);
            printf("Test %d - Rows read: %d\n", i + 1, rows_read);
            fclose(file);
        } else {
            printf("Test %d - Failed to open file: %s\n", i + 1, test_files[i]);
        }
    }
}

void TEST_get_highest_temp() {
    printf("\t\t \n\n TESTING get_highest_temp \t\t \n\n");
    double temp_array1[] = {14.4, 9.0, 5.5};
    printf("Test 1: %f\n", get_highest_temp(temp_array1, 3)); // Expected: 14.4

    double temp_array2[] = {20.1, 20.5, 20.2, 21.0, 19.8};
    printf("Test 2: %f\n", get_highest_temp(temp_array2, 5)); // Expected: 21.0

    double temp_array3[] = {-5.0, -3.0, -10.0, -1.0};
    printf("Test 3: %f\n", get_highest_temp(temp_array3, 4)); // Expected: -1.0

    double temp_array4[] = {32.5, 32.5, 32.5, 32.5};
    printf("Test ROW_SIZE: %f\n", get_highest_temp(temp_array4, 4)); // Expected: 32.5

    double temp_array5[] = {0.0};
    printf("Test 5: %f\n", get_highest_temp(temp_array5, 1)); // Expected: 0.0
}

void TEST_filter_high_temps() {
    printf("\t\t \n\n TESTING filter_high_temps \t\t \n\n");
    
    double temperatures[] = {14.4, 9.0, 5.5};
    int input_2D[][ROW_SIZE] = {
        {2019, 5, 14, 56},
        {2020, 4, 5, 36},
        {2020, 4, 4, 71}
    };
  
    int result_2D[MAX_LINES_INPUT][ROW_SIZE];
    
    int rows_read = filter_high_temps(input_2D, temperatures, 3, result_2D);
    printf("Test 1: Rows returned: %d\n", rows_read); // Expected: 1 (only 14.4 is >= 14.4 - 0.1)

    double temperatures2[] = {15.0, 20.0, 10.0, 19.5, 18.0};
    int input_2D2[][ROW_SIZE] = {
        {2019, 5, 14, 1},
        {2020, ROW_SIZE, 5, 2},
        {2020, ROW_SIZE, ROW_SIZE, 3},
        {2020, ROW_SIZE, 6, ROW_SIZE},
        {2020, ROW_SIZE, 7, 5}
    };
    rows_read = filter_high_temps(input_2D2, temperatures2, 5, result_2D);
    printf("Test 2: Rows returned: %d\n", rows_read); // Expected: 3 (15.0, 20.0, 19.5)

    double temperatures3[] = {10.0, 12.0, 8.0, 5.0};
    int input_2D3[][ROW_SIZE] = {
        {2021, 1, 1, 1},
        {2021, 1, 2, 2},
        {2021, 1, 3, 3}
    };
    rows_read = filter_high_temps(input_2D3, temperatures3, 3, result_2D);
    printf("Test 3: Rows returned: %d\n", rows_read); // Expected: 0 (none >= 10.0 - 0.1)

    double temperatures4[] = {30.0, 29.0, 32.0, 31.5};
    int input_2D4[][ROW_SIZE] = {
        {2018, 12, 1, 4},
        {2019, 6, 15, 5},
        {2020, 3, 20, 6}
    };
    rows_read = filter_high_temps(input_2D4, temperatures4, 3, result_2D);
    printf("Test ROW_SIZE: Rows returned: %d\n", rows_read); // Expected: 3 (all temps are high)

    double temperatures5[] = {14.0, 14.1, 14.2, 14.3};
    int input_2D5[][ROW_SIZE] = {
        {2020, 1, 1, 10},
        {2020, 1, 2, 11},
        {2020, 1, 3, 12}
    };
    rows_read = filter_high_temps(input_2D5, temperatures5, 3, result_2D);
    printf("Test 5: Rows returned: %d\n", rows_read); // Expected: 3 (all temps are high)
}

void TEST_get_station_stats() {
    printf("\t\t \n\n TESTING get_station_stats \t\t \n\n");
    int input_2D[][ROW_SIZE] = {
        {2019, 5, 14, 56},
        {2020, 4, 5, 36},
        {2020, 4, 4, 56},
        {2020, 4, 6, 56}
    };
    double temperatures[] = {14.4, 9.0, 15.0, 20.0};
    double stats[3];

    int result = get_station_stats(input_2D, temperatures, ROW_SIZE, stats, 56);
    printf("Test 1: Found: %d, Min: %f, Max: %f, Avg: %f\n", result, stats[0], stats[1], stats[2]); // Expected: Found: 3, Min: 14.ROW_SIZE, Max: 20.0, Avg: 17.8

    result = get_station_stats(input_2D, temperatures, ROW_SIZE, stats, 36);
    printf("Test 2: Found: %d, Min: %f, Max: %f, Avg: %f\n", result, stats[0], stats[1], stats[2]); // Expected: Found: 1, Min: 9.0, Max: 9.0, Avg: 9.0

    result = get_station_stats(input_2D, temperatures, ROW_SIZE, stats, 100);
    printf("Test 3: Found: %d\n", result); // Expected: Found: 0 (not found)

    int input_2D2[][ROW_SIZE] = {
        {2020, 1, 1, 1},
        {2020, 1, 2, 1},
        {2020, 1, 3, 1}
    };
    double temperatures2[] = {15.0, 16.0, 17.0};

    result = get_station_stats(input_2D2, temperatures2, 3, stats, 1);
    printf("Test ROW_SIZE: Found: %d, Min: %f, Max: %f, Avg: %f\n", result, stats[0], stats[1], stats[2]); // Expected: Found: 3, Min: 15.0, Max: 17.0, Avg: 16.0

    double temperatures3[] = {10.0, 12.0, 8.0, 5.0};
    int input_2D3[][ROW_SIZE] = {
        {2021, 1, 1, 1},
        {2021, 1, 2, 1},
        {2021, 1, 3, 2}
    };

    result = get_station_stats(input_2D3, temperatures3, 3, stats, 2);
    printf("Test 5: Found: %d, Min: %f, Max: %f, Avg: %f\n", result, stats[0], stats[1], stats[2]); // Expected: Found: 1, Min: 8.0, Max: 8.0, Avg: 8.0
}

void TEST_is_before() {
    printf("\t\t \n\n TESTING is_before \t\t \n\n");
    printf("Test 1: %d\n", is_before(2020, 1, 1, 2021, 1, 1)); // Expected: 1 (true)
    printf("Test 2: %d\n", is_before(2021, 1, 1, 2020, 1, 1)); // Expected: 0 (false)
    printf("Test 3: %d\n", is_before(2020, 1, 1, 2020, 1, 2)); // Expected: 1 (true)
    printf("Test ROW_SIZE: %d\n", is_before(2020, 1, 2, 2020, 1, 1)); // Expected: 0 (false)
    printf("Test 5: %d\n", is_before(2020, 1, 1, 2020, 2, 1)); // Expected: 1 (true)
}

void TEST_get_avg_temp_before_date() {
    printf("\t\t \n\n TESTING get_avg_temp_before_date \t\t \n\n");
    int input_2D[][ROW_SIZE] = {
        {2019, 5, 14, 56},
        {2020, 4, 5, 36},
        {2020, 4, 4, 56},
        {2020, 4, 6, 56}
    };

    double temperatures[] = {14.4, 9.0, 15.0, 20.0};
    double avg_temp;

    int result = get_avg_temp_before_date(input_2D, temperatures, ROW_SIZE, 2020, ROW_SIZE, 5, &avg_temp);
    printf("Test 1: Result: %d, Avg Temp: %f\n", result, avg_temp); // Expected: Result: 1, Avg Temp: 14.ROW_SIZE

    result = get_avg_temp_before_date(input_2D, temperatures, ROW_SIZE, 2019, 6, 1, &avg_temp);
    printf("Test 2: Result: %d, Avg Temp: %f\n", result, avg_temp); // Expected: Result: 1, Avg Temp: 14.ROW_SIZE

    result = get_avg_temp_before_date(input_2D, temperatures, ROW_SIZE, 2020, ROW_SIZE, ROW_SIZE, &avg_temp);
    printf("Test 3: Result: %d, Avg Temp: %f\n", result, avg_temp); // Expected: Result: 1, Avg Temp: 11.7

    result = get_avg_temp_before_date(input_2D, temperatures, ROW_SIZE, 2020, ROW_SIZE, 6, &avg_temp);
    printf("Test ROW_SIZE: Result: %d, Avg Temp: %f\n", result, avg_temp); // Expected: Result: 1, Avg Temp: 11.7

    result = get_avg_temp_before_date(input_2D, temperatures, ROW_SIZE, 2021, 1, 1, &avg_temp);
    printf("Test 5: Result: %d\n", result); // Expected: Result: 0 (no temps)
}






