/*
 * @Brief This program manages city data, performing operations like sorting, inserting, and filtering based on criteria such as name or population. 
 * It reads city data from a CSV file, identifies the highest population, and retrieves details of cities by name. Includes built-in tests for core functionalities.
 * 
 * @Author Zac Mattias
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_STR_LEN 20
#define MAX_CHAR_ARRAY (MAX_STR_LEN +1)
#define MAX_CITIES 200
#define FILENAME "CountryCityDataSmall.csv"

typedef struct {
    char name[MAX_CHAR_ARRAY]; // name of the city
    int population;            // population of the city
} City;

// Helper functions:
void print_city(City* c);
void print_city_array(City array[], int num_cities);

// Assigment exercises:
void insert_city(City sorted_cities[], int num_cities, City city_data);
int read_data(FILE* file_handle, City cities[]);
int get_highest_city_population(City cities[], int num_cities);
int get_names_that_end_with(City cities[], int num_cities, City destination[], char letter);
int get_cities_with_above_avg_population(City cities[], int num_cities, char dest_names[][MAX_CHAR_ARRAY]);
int find_city(City cities[], int num_cities, char name[MAX_CHAR_ARRAY]); 
int get_population_of_city(City cities[], int num_cities, char name[MAX_CHAR_ARRAY]);

void test_insert_city() {
    printf("\nTest 0 - insert_city (STANDBY) \n");
    
    City cities[MAX_CITIES] = {{"Cairo", 7734602}, {"Tokyo", 31480498}};
    int num_cities = 2;

    City c1 = {"Sao Paulo", 10021437};
    City c2 = {"Tokyo", 42931275};

    printf("cities array before insert:\n");
    print_city_array(cities, num_cities);

    printf("\ninserting: ");
    print_city(&c1);
    insert_city(cities, num_cities, c1);
    num_cities++; 
    printf("\ncities array after insert:\n");
    print_city_array(cities, num_cities);

    printf("\ninserting: ");
    print_city(&c2);
    insert_city(cities, num_cities, c2);
    num_cities++; 
    printf("\ncities array after insert:\n");
    print_city_array(cities, num_cities);

}


void test_get_highest_city_population() {
    // Test 1: Normal case
    City cities[] = {{"Cairo", 7734602}, {"Sao Paulo", 10021437}, {"Toronto", 4612187}};
    int num_cities = 3;
    int result = get_highest_city_population(cities, num_cities);
    
    printf("Test 1 - get_highest_city_population: ");
    if (result == 10021437) {
        printf("Passed\n");
        printf("Expected: 10021437, Got: %d\n", result);
    } else {
        printf("Failed (Expected: 10021437, Got: %d)\n", result);
    }
}

void test_get_names_that_end_with() {
    // Test 2: Normal case
    City cities[] = {{"Cairo", 7734602}, {"Rome", 36332}, {"Pune", 2935968}};
    int num_cities = 3;
    City destination[MAX_CITIES];
    char letter = 'e';
    
    int result = get_names_that_end_with(cities, num_cities, destination, letter);
    
    printf("Test 2 - get_names_that_end_with: ");
    if (result == 2) {
        printf("Passed\n");
        printf("Expected: 2, Got: %d\n", result);
    } else {
        printf("Failed (Expected: 2, Got: %d)\n", result);
    }
}

void test_get_cities_with_above_avg_population() {
    // Test 3: Normal case
    City cities[] = {{"Cairo", 7734602}, {"Madrid", 3802644}, {"Rome", 36332}, {"Rome", 2643736}};
    int num_cities = 4;
    char dest_names[MAX_CITIES][MAX_CHAR_ARRAY];
    
    int result = get_cities_with_above_avg_population(cities, num_cities, dest_names);
    
    printf("Test 3 - get_cities_with_above_avg_population: ");
    if (result == 2) {
        printf("Passed\n");
         printf("Expected: 2, Got: %d\n", result);
    } else {
        printf("Failed (Expected: 2, Got: %d)\n", result);
    }
}

void test_find_city() {
    // Test 4: Normal case
    City cities[] = {{"Cairo", 7734602}, {"Rome", 2643736}, {"Madrid", 3802644}, {"Rome", 36332}};
    int num_cities = 4;
    char city_name[MAX_CHAR_ARRAY] = "Rome";
    
    int result = find_city(cities, num_cities, city_name);
    
    printf("Test 4 - find_city: ");
    if (result == 1) {
        printf("Passed\n");
        printf("Expected: 1, Got: %d\n", result);
    } else {
        printf("Failed (Expected: 1, Got: %d)\n", result);
    }
}

void test_get_population_of_city() {
    // Test 5: Normal case
    City cities[] = {{"Cairo", 7734602}, {"Madrid", 3802644}, {"Rome", 36332}, {"Rome", 2643736}};
    int num_cities = 4;
    char city_name[MAX_CHAR_ARRAY] = "rome";
    
    int result = get_population_of_city(cities, num_cities, city_name);
    
    printf("Test 5 - get_population_of_city: ");
    if (result == 36332) {
        printf("Passed\n");
        printf("Expected: 36332, Got: %d\n", result);
    } else {
        printf("Failed (Expected: 36332, Got: %d)\n", result);
    }
}

int main(void) {
    // Exercise 1 (insert_city) tests:
    City cities[MAX_CITIES] = {{"Cairo", 7734602}, {"Tokyo", 31480498}};
    City c1 = {"Sao Paulo", 10021437};
    City c2 = {"Tokyo", 42931275};
    printf("cities array before insert:\n");
    print_city_array(cities, 2);
    printf("\ninserting: ");
    print_city(&c1);
    insert_city(cities, 2, c1);
    printf("\ncities array after insert:\n");
    print_city_array(cities, 3);
    printf("\ninserting: ");
    print_city(&c2);
    insert_city(cities, 3, c2);
    printf("\ncities array after insert:\n");
    print_city_array(cities, 4);
    printf("\n");


    // Exercise 2 (read_data) tests:
    FILE* in_file; 
    City array[MAX_CITIES];
    int num_cities;

    in_file = fopen(FILENAME, "r");
    if (in_file == NULL) {
        printf("Could not read file\n");
        return 1;
    }
    printf("reading cities from %s:\n", FILENAME);
    num_cities = read_data(in_file, array);
    print_city_array(array, num_cities);

    printf("\t\t \n\n");
    

    // Write tests for the remaining functions here:
    test_insert_city();
    printf("\t\t \n\n");
    
    test_get_highest_city_population();
    printf("\t\t \n\n");
    
    test_get_names_that_end_with();
    printf("\t\t \n\n");
    
    test_get_cities_with_above_avg_population();
    printf("\t\t \n\n");
    
    test_find_city();
    printf("\t\t \n\n");
    
    test_get_population_of_city();
    printf("\t\t \n\n");
    return 0;
}

/**
 * @brief Outputs information about the given City instance
 * @param c A pointer to the City instance to output information about
 */
void print_city(City* c) {
    printf("%s, population: %d\n", c->name, c->population);
}

/**
 * @brief Outputs information about City instances in the given array
 * @param array the array of City instances
 * @param num_cities the integer number of cities to output information about
 */
void print_city_array(City array[], int num_cities) {
    int i;
    for (i = 0; i < num_cities; i++) {
        print_city(&array[i]);
    }
}

/**
 * @brief Inserts city_data into the sorted array of City instances
 * @param sorted_cities the sorted array of City instances
 * @param num_cities the integer number of cities to output information about
 * @param city_data the City instance to insert into the correct position of the array
 */
void insert_city(City sorted_cities[], int num_cities, City city_data) {
  int i;
    for (i = num_cities; i > 0; i--) {
        if (strcmp(city_data.name, sorted_cities[i - 1].name) < 0 ||
            (strcmp(city_data.name, sorted_cities[i - 1].name) == 0 && city_data.population < sorted_cities[i - 1].population)) {
            sorted_cities[i] = sorted_cities[i - 1];
        } else {
            break;
        }
    }
    sorted_cities[i] = city_data;
}

/**
 * @brief Reads data from the linked input file into the array of City instances
 * @param file_handle a valid file opened for reading
 * @param cities the array of City instances to store city information from from the file
 */
int read_data(FILE* file_handle, City cities[]) {
    char country[MAX_CHAR_ARRAY];
    char city_name[MAX_CHAR_ARRAY];
    int population;
    int count = 0;

    while (count < MAX_CITIES && fscanf(file_handle, "%[^,],%[^,],%d\n", country, city_name, &population) == 3) {
        City new_city;
        strncpy(new_city.name, city_name, MAX_CHAR_ARRAY);
        new_city.name[MAX_CHAR_ARRAY - 1] = '\0';
        new_city.population = population;
        insert_city(cities, count, new_city);
        count++;
    }
    return count;
}

/**
 * @brief Finds the highest population among an array of cities.
 * @param cities Array of City instances.
 * @param num_cities Number of valid elements in the cities array.
 * @return The highest population in the array.
 */
int get_highest_city_population(City cities[], int num_cities) {
    int max_population = cities[0].population;

    for (int i = 1; i < num_cities; i++) {
        if (cities[i].population > max_population) {
            max_population = cities[i].population;
        }
    }
    return max_population;
}

/**
 * @brief Copies cities with names ending with a given character into a destination array.
 * @param cities Array of City instances.
 * @param num_cities Number of valid elements in the cities array.
 * @param destination Array to store cities with matching names.
 * @param letter Character to match at the end of city names (case insensitive).
 * @return Number of cities copied to the destination array.
 */
int get_names_that_end_with(City cities[], int num_cities, City destination[], char letter) {
    int count = 0;

    for (int i = 0; i < num_cities; i++) {
        int len = strlen(cities[i].name);
        if (tolower(cities[i].name[len - 1]) == tolower(letter)) {
            destination[count++] = cities[i];
        }
    }
    return count;
}

/**
 * @brief Copies names of cities with above-average populations to a destination array.
 * @param cities Array of City instances.
 * @param num_cities Number of valid elements in the cities array.
 * @param dest_names Array to store names of cities with above-average populations.
 * @return Number of cities with above-average populations.
 */
int get_cities_with_above_avg_population(City cities[], int num_cities, char dest_names[][MAX_CHAR_ARRAY]) {
    double total_population = 0;

    for (int i = 0; i < num_cities; i++) {
        total_population += cities[i].population;
    }

    double avg_population = total_population / num_cities;
    int count = 0;

    for (int i = 0; i < num_cities; i++) {
        if (cities[i].population > avg_population) {
            strncpy(dest_names[count++], cities[i].name, MAX_CHAR_ARRAY);
        }
    }
    return count;
}

/**
 * @brief Finds the first occurrence of a city name in the array.
 * @param cities Array of City instances.
 * @param num_cities Number of valid elements in the cities array.
 * @param name Name of the city to search for (case insensitive).
 * @return The index of the first occurrence, or -1 if not found.
 */
int find_city(City cities[], int num_cities, char name[MAX_CHAR_ARRAY]) {
    for (int i = 0; i < num_cities; i++) {
        if (strcasecmp(cities[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Retrieves the population of the first occurrence of a given city name.
 * @param cities Array of City instances.
 * @param num_cities Number of valid elements in the cities array.
 * @param name Name of the city to search for (case insensitive).
 * @return The population of the city, or -1 if not found.
 */
int get_population_of_city(City cities[], int num_cities, char name[MAX_CHAR_ARRAY]) {
    for (int i = 0; i < num_cities; i++) {
        if (strcasecmp(cities[i].name, name) == 0) {
            return cities[i].population;
        }
    }
    return -1;
}
