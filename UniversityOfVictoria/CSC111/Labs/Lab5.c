/**
 * @brief This program includes several exercises showing the use of condition-controlled loops, 
 * pointers, and user input handling with the `scanf` function.
 * @author Zac Matthias
 */
#include <stdio.h>

/* 
 * @Brief: This function continuously prompts the user for positive integers and sums them.
 * The function terminates when the user enters -1 and returns the sum of all valid inputs.
 * @Returns: The added sum of the inputted numbers that isn't -1.
 */
int sum_integers() {
    int num, sum = 0;
    printf("\tEnter a number to be added to the sum. (currently 0) \n");
    printf("\tEnter an integer or -1 to stop: \n");
    scanf("%d", &num);

    while (num != -1) {
        sum += num;
        printf("\tEnter an integer or -1 to stop: \n");
        scanf("%d", &num);
    }

    return sum;
}

/*
 * @Brief: This function asks the user to input an integer within a specified range.
 * It keeps prompting the user until a valid integer is entered and returns the valid input.
 * @Param int min - The minimum acceptable integer.
 * @Param int max - The maximum acceptable integer.
 * @Return: The first valid integer entered within the specified range (int)
 */
int get_number(int min, int max) {
    int input;
    
    printf("\tEnter an integer between %d and %d:\n ", min, max);
    scanf("%d", &input);

    while (input < min || input > max) {
        printf("\tNo, an integer between %d and %d: \n", min, max);
        scanf("%d", &input);
    }

    return input;
}

/*
 * @Brief: This function calculates the average of non-negative integers between 0 and 100.
 * The function keeps prompting the user for inputs, stopping when -1 is entered.
 * It excludes values not within the range (0 to 100) and computes the average of valid inputs.
 * @Return: The average of all valid integers entered (double). If no valid numbers are entered, returns 0.0.
 */
double get_average() {
    int num, count = 0;
    double sum = 0;

    printf("\tEnter an integer between 0 and 100 to be averaged out, or -1 to stop: \n");
    scanf("%d", &num);

    while (num != -1) {
        if (num >= 0 && num <= 100) {
            sum += num;
            count++;
        } else {
            printf("No, an integer between 0 and 100: ");
        }
        scanf("%d", &num);
    }

    if (count == 0) {
        return 0.0;  // Return 0 if no valid numbers are entered
    } else {
        return sum / count;  // Return the average of valid inputs
    }
}

int main() {
    // Example usage of sum_integers function
    printf("Value returned from sum_integers: %d\n",  sum_integers());

    // Example usage of get_number function
    printf("Value returned from get_number: %d\n", get_number(-8, 22));

    // Example usage of get_average function
    printf("Value returned from get_average: %.1f\n", get_average());

    return 0;
}
