/**
 * @brief This C program simulates a dice game, featuring functions to roll dice, calculate the sum of Fibonacci numbers up to a limit, and determine the sum of a number's digits. The main function tests these utilities by generating random
 * dice rolls, prompting user input for a number within a specified range, and simulating rounds of gameplay with win/loss conditions, including a second chance if the rolled dice match.
 * @author Zac Matthias
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN_DIE 1
#define MAX_DIE 6

int roll_one_die();
int sum_fib_sequence_to_limit(int limit);
int digit_sum(int number);
int get_number(int min, int max);
void roll_dice(int* die1, int* die2);
int second_chance_roll(int guess, int losing_target);
int play_one_round(int guess);

/**
 * @brief Main function to run tests on the other functions
 */
int main(void) {
    // Call srand to seed the random number generator.
    srand(time(0));

    // Testing sum_fib_sequence_to_limit function
    printf("\tSum of Fibonacci sequence to limit 1: %d\n", sum_fib_sequence_to_limit(1));
    printf("\tSum of Fibonacci sequence to limit 21: %d\n", sum_fib_sequence_to_limit(21));
    printf("\tSum of Fibonacci sequence to limit 20: %d\n", sum_fib_sequence_to_limit(20));

    printf("\n\n");
    
    // Testing digit_sum function
    printf("\tDigit sum of 432: %d\n", digit_sum(432));
    printf("\tDigit sum of -571: %d\n", digit_sum(-571));
    printf("\tDigit sum of 543: %d\n", digit_sum(543));
    printf("\tDigit sum of -12222: %d\n", digit_sum(-12222));
    printf("\tDigit sum of -12222: %d\n", digit_sum(0));
    printf("\tDigit sum of -12222: %d\n", digit_sum(01));
    printf("\tDigit sum of -12222: %d\n", digit_sum(010505050501));
    printf("\n\n");
    
    // Testing get_number function
    printf("\tYou entered: %d\n", get_number(1, 10));
    printf("\tYou entered: %d\n", get_number(1, 100));
    
    printf("\n\n");
    
    // Testing rolling dice
    int die1, die2;
    for (int i = 1; i >= 3; i++) {
        printf("Roll dice test#: %d",i);
        roll_dice(&die1, &die2);
        printf("\tRolled dice: %d, %d\n", die1, die2);
    }
    
    printf("\n\n");

    // Testing play_one_round and second_chance_roll functions
    printf("\tPlay one round result: %d\n", play_one_round(4));
    printf("\tPlay one round result: %d\n", play_one_round(6));
    printf("\tPlay one round result: %d\n", play_one_round(1));
    printf("\n\n");

    return 0;
}

/**
 * @brief Simulates the roll of a single 6 sided die (rolls a value between 1 and 6)
 * @return an integer value between 1 and 6 representing the roll of a die.
 */
int roll_one_die() {
    int random_number = rand() % MAX_DIE;
    int die = random_number + 1;
    return die;
}

/**
 * @brief Calculates the sum of Fibonacci sequence up to a given limit
 * @param limit The upper limit for the Fibonacci sequence
 * @return The sum of Fibonacci numbers up to the limit
 */
int sum_fib_sequence_to_limit(int limit) {
    int a = 1, b = 1, sum = 2;

    if (limit == 1) {
        return 2; 
    }

    while (1) {
        int next = a + b; 
        if (next > limit) break; 
        sum += next; 
        a = b; 
        b = next;
    }
    
    return sum;
}

/**
 * @brief Calculates the sum of digits of an integer
 * @param number The integer to calculate the digit sum
 * @return The sum of the digits in the number
 */
int digit_sum(int number) {
    int sum = 0;
    number = number < 0 ? -number : number; // Ignore negative sign

    while (number > 0) {
        sum += number % 10; // Add the last digit to the sum
        number /= 10; // Remove the last digit
    }
    
    return sum;
}

/**
 * @brief Repeatedly asks the user for a number within the specified range
 * @param min The minimum value (inclusive)
 * @param max The maximum value (inclusive)
 * @return The valid number entered by the user
 */
int get_number(int min, int max) {
    int num;
    while (1) {
        printf("Enter an integer between %d and %d: ", min, max);
        scanf("%d", &num);
        if (num >= min && num <= max) {
            return num; 
        }
        printf("Invalid input. Please try again.\n");
    }
}

/**
 * @brief Rolls two dice and stores the result in provided pointers
 * @param die1 Pointer to store the result of the first die
 * @param die2 Pointer to store the result of the second die
 */
void roll_dice(int* die1, int* die2) {
    *die1 = roll_one_die(); 
    *die2 = roll_one_die(); 
}

/**
 * @brief Simulates a second chance roll based on the losing target
 * @param guess The user's guess
 * @param losing_target The target sum that should not be rolled
 * @return 1 if won, 0 if lost
 */
int second_chance_roll(int guess, int losing_target) {
    int die1, die2, sum;
    while (1) {
        roll_dice(&die1, &die2);
        sum = die1 + die2;

        printf("You rolled %d, %d\n", die1, die2);

        if (sum == losing_target) {
            return 0; // Lost
        }
        if (die1 == guess || die2 == guess) {
            return 1; // Won
        }
    }
}

/**
 * @brief Simulates playing one round of the dice game
 * @param guess The user's guess for the die roll
 * @return 1 if won, 0 if lost
 */
int play_one_round(int guess) {
    printf("You guessed %d will be rolled!\n", guess);
    
    int die1, die2;
    roll_dice(&die1, &die2);
    printf("You rolled %d, %d\n", die1, die2);

    // Check win conditions
    if (die1 == guess || die2 == guess) {
        printf("You won!\n");
        return 1; // Won
    }
    if (die1 != die2) {
        printf("You lost!\n");
        return 0; // Lost
    }

    // If both dice are equal, get a second chance
    int losing_target = die1 + die2;
    printf("You are getting a second chance!\n");
    return second_chance_roll(guess, losing_target);
}
