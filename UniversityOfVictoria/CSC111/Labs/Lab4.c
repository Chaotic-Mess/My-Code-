/**
 * @brief Lab 4 program that focuses on loops
 * @author Zac Matthias
 */
#include <stdio.h>

void print_n_chars(int n, char c);
void print_right_triangle(int height, char c);
void print_number_right_triangle(int height);
void print_triangle(int height, char Filler_char, char Background_char);
void print_number_triangle(int height);


int main(void) {
    // Test print_n_chars
    //print_n_chars(3, 'h');
    printf("\n");

    // Test print_right_triangle
    print_right_triangle(3, '$');
    print_right_triangle(4, '*');

    // Tests print_triangle
    print_triangle(3, '$', '*');
    print_triangle(4, '*', '-');

    // Tests print_number_triangle
    print_number_triangle(3);
    print_number_triangle(5);

    // Test print_number_right_triangle
    print_number_right_triangle(3);
    print_number_right_triangle(5);
    
    return 0;
}

/**
 * @brief Prints out the character c exactly n times on a line
 * @param n the number of times to print the character as an integer
 * @param c the character to be repeatedly printed
 * note to self:
 * c needs to be marked with ' ' NOT " ". " " is passing the string "h" (which is of type char *) as the second argument, but the function expects a char (' ').
 */
void print_n_chars(int n, char c) {
    for (int i = 0; i < n; i++) {
        printf("%c", c);
    }
}

/**
 * @brief Prints a right-aligned triangle of a specified height using a given character.
 * 
 * The triangle is printed with spaces to the left, and the character `c` to form the triangle. 
 * Each row contains more of the character, starting from one character at the top and increasing
 * by one character per row until the base is reached.
 * 
 * @param height The height of the triangle as an integer.
 * @param c The character to construct the triangle, denoted with single quotes (' ') NOT double quotes (" "), as this is a character  * and not a string.
 */
void print_right_triangle(int height, char c){
   for (int i = 1; i <= height; ++i) {
        print_n_chars(height - i, ' '); // make the space
        print_n_chars(i, c); // make the char
        printf("\n"); // next line
    }
    printf("\n\n");
}

/**
 * @brief Prints a right-aligned triangle of a specified height using the current int.
 * 
 * The triangle is printed with spaces to the left, and the i int to form the triangle. 
 * Each row contains more of the character, starting from one number at the top and increasing
 * by one per row until the base is reached.
 * 
 * @param height The height of the triangle as an integer.
 * 
 */
void print_number_right_triangle(int height){
   for (int i = 1; i <= height; ++i) {
        print_n_chars(height - i, ' '); // make the space
        // print the number
        for (int j = 1; j <= i; ++j) {
            printf("%d", j);
        }
        printf("\n"); // next line
    }
    printf("\n\n");
}


/**
 * @brief Prints a centered triangle of a specified height using two characters.
 * 
 * The triangle is printed using the `Background_char` to pad the sides, and the `Filler_char`
 * to form the body of the triangle. The number of characters in the body increases by 2 on each row,
 * starting from one character at the top and increasing symmetrically on both sides until the base is reached.
 * 
 * @param height The height of the triangle as an integer. Each level of the triangle has 1 more row.
 * @param Background_char The character to use for padding the sides of the triangle, typically spaces or another symbol.
 * @param Filler_char The character to construct the body of the triangle.
 */
void print_triangle(int height, char Filler_char, char Background_char) {
    for (int i = 1; i <= height; ++i) {
        print_n_chars(height - i, Background_char);  // Print leading spaces
        print_n_chars(2 * i - 1, Filler_char);    // Print inner characters (odd numbers starting at 1, 3, 5...)
        print_n_chars(height - i, Background_char);
        
        printf("\n");
    }
    printf("\n\n");
}
/**
 * @brief Prints a centered number triangle of a specified height.
 * 
 * The triangle is printed with leading spaces for proper alignment, and each row contains
 * numbers starting from 1, incrementing up to the current row number, and then decrementing back to 1.
 * 
 * @param height The height of the triangle as an integer. Each level of the triangle has one more row than the previous level.
 */
void print_number_triangle(int height) {
    for (int i = 1; i <= height; ++i) {
        print_n_chars(height - i, ' ');   // leading spaces

        // Print increasing numbers from 1 to i
        for (int j = 1; j <= i; ++j) { 
            printf("%d", j); 
        } 

        // Print decreasing numbers from (i-1) to 1
        for (int j = i - 1; j >= 1; --j) { 
            printf("%d", j);
        }
        
        printf("\n"); // Move to the next row
    }
    printf("\n\n");
}



