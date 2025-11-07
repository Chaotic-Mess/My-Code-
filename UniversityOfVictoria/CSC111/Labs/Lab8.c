#include <stdio.h>
#include <string.h>
#include <ctype.h>

/*
    @Author Zac Matthias
    @Brief This program provides text manipulation and analysis functions across four parts. Part one modifies strings by replacing spaces and changing specific characters. Part two handles controlled user input through read_line and read_word. Part three counts
    characters and words and capitalizes the first letter of each word. Part four trims spaces for cleaner text output, with testing functions demonstrating each feature.
    
    gcc -Wall -std=c18 -o TEST Lab8.c
    ./TEST
*/

void Test_PartOne();
void mystery1(char str[], int length);
void mystery2(char str[], int index, char n);

void Test_PartTwo();
void read_line(char output_array[], int size);
char read_word(char output_array[], int size);

void Test_PartThree();
int count_characters(char str[]);
int count_words(char str[]);
void capitalize(char str[]);

void Test_PartFour();
void remove_leading_spaces(char str[], char output[]);
void remove_trailing_spaces(char str[], char output[]);
void condense_spaces(char str[], char output[]);

int main(void) {
    Test_PartOne();
	Test_PartTwo();
    Test_PartThree();
    Test_PartFour();
    return 0;
}

//////////// PART ONE
void mystery1(char str[], int length) {
    for (int i = 0; i < 26; i++) {
        if (isspace(str[i])) {
            str[i] = '^';
        }
    }
}

void mystery2(char str[], int index, char n) {
    str[index] = n;
}





//////////// PART 2
/**
 * @brief Reads characters from the user and stores them in output_array until
 *        either a \n is detected or the array fills up (size-1 characters entered)
 *        This function also writes a null terminator ('\0') after the last character
 *        in output_array -- and should not write any \n characters.
 * @param output_array the array of characters to write to
 * @param size the maximum capacity of the output_array
 */
void read_line(char output_array[], int size) {
    int i = 0;
    char ch;

    while (i < size - 1) {
        ch = getchar();
        if (ch == '\n') {
            break; // Stop reading on newline
        }
        output_array[i++] = ch;
    }
    
    output_array[i] = '\0'; 
}


/**
 * @brief reads characters from the user into the output_array until a non-alphabetic 
 *        character is entered (indicating the end of a word). A null terminator 
 *        is written into the index after the last character written to output_array
 * @param output_array the array of characters to write to
 * @param size the maximum capacity of the output_array
 * @return a char representing the last character read from the user, which is either
 *         the first non-alphabetic character entered or the last alphabetic character
 *         entered (if the array reached its capacity before reading a non-alphabetic char)
 * @pre size-1 alphabetic characters OR LESS are entered before a non-alphabetic character
 */
char read_word(char output_array[], int size) {
    int i = 0;
    char ch;

    while (i < size - 1) { 
        ch = getchar();
        
        if (!isalpha(ch)) { // Check if the character is non-alphabetic
            break;
        }
        
        output_array[i++] = ch;
    }
    
    output_array[i] = '\0'; // Null terminate the string
    return ch; // Return the non-alphabetic character
}






//////////// PART 3
/**
 * @brief Returns a count of total the number of characters in str,
 *        not including the null terminator
 * @param str a null terminated string
 * @return the integer number of characters
 */
int count_characters(char str[]) {
    int count = 0;

    while (str[count] != '\0') {
        count++;
    }

    return count; // Return the total count of characters
}

/**
 * @brief Returns a count of the number of words in str
 * @param str a null terminated string
 * @return the integer number of words in str
 * @note for this exercise assume all non-whitespace characters
 *       are a word (ie. "08", "-->", "8*4" are considered words)
 */
int count_words(char str[]){
    int count = 0;
    int inWord = 0;
    for (int i = 0; str[i] != '\0'; i++) {
        if (!isspace(str[i])) {
            if (!inWord) {
                count++;
                inWord = 1;
            }
        } else {
            inWord = 0;
        }
    }
    return count;
}

/**
 * @brief Updates the given str by capitalizing the 
 *        first letter of each word
 * @param str a null terminated string
 */
void capitalize(char str[]){
    int inWord = 0;
    for (int i = 0; str[i] != '\0'; i++) {
        if (!isspace(str[i]) && !inWord) {
            str[i] = toupper(str[i]);
            inWord = 1;
        } else if (isspace(str[i])) {
            inWord = 0;
        }
    }
}





//////////// Part four
/**
 * @brief copy all characters from str to output except
 *        for any white space charactes at the start of str
 * @param str the char array to copy characters from
 * @param output the char array to copy characters to
 */
void remove_leading_spaces(char str[], char output[]){
    int i = 0;
    
    while (isspace(str[i])) {
        i++;
    }
    
    int j = 0;
    while (str[i] != '\0') {
        output[j++] = str[i++];
    }
    output[j] = '\0'; 

}

/**
 * @brief copy all characters from str to output except
 *        for any white space charactes at the end of str
 * @param str the char array to copy characters from
 * @param output the char array to copy characters to
 */
void remove_trailing_spaces(char str[], char output[]){
    int len = strlen(str);
    
    int end = len - 1;
    while (end >= 0 && isspace(str[end])) {
        end--;
    }
    
    int i;
    for (i = 0; i <= end; i++) {
        output[i] = str[i];
    }
    output[i] = '\0'; 
}


/**
 * @brief copy all characters from str to output except
 *        all consecutive spaces are treated as a single space
 * @param str the char array to copy characters from
 * @param output the char array to copy characters to
 */
void condense_spaces(char str[], char output[]){
    int i = 0, j = 0;
    int in_space = 0;
    
    while (str[i] != '\0') {
        if (isspace(str[i])) {
            if (!in_space) {
                output[j++] = ' ';
                in_space = 1;
            }
        } else {
            output[j++] = str[i];
            in_space = 0;
        }
        i++;
    }
    
    if (j > 0 && isspace(output[j - 1])) {
        j--;
    }
    
    output[j] = '\0'; 
    
}









//////////// TEST FUNCTIONS
void Test_PartOne() {
    printf("\n\n\t\tTESTING PART ONE\t\t\n\n");
    
    // Exercise 1
    char str1[] = "this is part 1 of lab 8";
    mystery1(str1, 27);
    printf("updated string: %s\n", str1);
    
    // Exercise 2
    char str2[] = "Please read the instructions";
    mystery2(str2, 10, '\0');
    printf("updated string: %s\n", str2);
    mystery2(str2, 10, 'X');
    printf("updated string: %s\n", str2);

    printf("\n\n\n\n");
}

void Test_PartTwo() {
    printf("\n\n\t\tTESTING PART TWO\t\t\n\n");

    // Exercise 1 Tests: Uncomment the code below to test.
    char small_line[5];
    char big_line[1000];

    printf("Enter a line of input:\n");
    read_line(big_line, 1000);
    printf("You entered:-%s-\n", big_line);

    printf("Enter another line (try adding the characters: 12345678):\n");
    read_line(small_line, 5);
    printf("You entered: -%s-\n", small_line);

    printf("this next call to read_line line will read any remaining characters not read by previous call:\n");
    read_line(big_line, 1000);
    printf("You entered: -%s-\n", big_line);
    printf("What do you notice about the output?\n");
    

    // Exercise 2 Tests: Uncomment the code below to test.
    char word[100];
    char terminating_character;
    printf("Enter a word of input:\n");
    terminating_character = read_word(word, 100);
    printf("You entered: \"%s\"\n", word);
    printf("Terminating: %c\n", terminating_character);

    printf("Enter a word of input:\n");
    terminating_character = read_word(word, 100);
    printf("You entered: \"%s\"\n", word);
    printf("Terminating: %c\n", terminating_character);

    printf("Enter a word of input:\n");
    terminating_character = read_word(word, 100);
    printf("You entered: \"%s\"\n", word);
    printf("Terminating: %c\n", terminating_character);

    printf("\n\n\n\n");
}

void Test_PartThree() {
    printf("\n\n\t\tTESTING PART THREE\t\t\n\n");
    
    char S1[] = "Raspberry";
    char S2[] = "      "; // Contains 0 words, 6 characters
    char S3[] = "CSc 111 strINGS LaB";
    char S4[] = "   raspberry pear pineapple banana";
    char S5[] = "   <-- spaces at the beginning, spaces at the end -->  ";

    //Make a new array to use as temporary storage.
    char W[1000];

    // Exercise 1 Tests: Uncomment the code below to test.
    printf("\n\nTesting count_characters\n\n");
    int num_chars;
    num_chars = count_characters(S1);
    printf("Characters, should be 9: %d\n", num_chars );
    num_chars = count_characters(S2);
    printf("Characters, should be 6: %d\n", num_chars );
    num_chars = count_characters(S3);
    printf("Characters, should be 19: %d\n", num_chars );
    num_chars = count_characters(S4);
    printf("Characters, should be 34: %d\n", num_chars );
    num_chars = count_characters(S5);
    printf("Characters, should be 55: %d\n", num_chars );
    

    // Exercise 2 Tests: Uncomment the code below to test.
    printf("\n\nTesting count_words\n\n");
    int num_words;
    num_words = count_words(S1);
    printf("Words, should be 1: %d\n", num_words );
    num_words = count_words(S2);
    printf("Words, should be 0: %d\n", num_words );
    num_words = count_words(S3);
    printf("Words, should be 4: %d\n", num_words );
    num_words = count_words(S4);
    printf("Words, should be 4: %d\n", num_words );
    num_words = count_words(S5);
    printf("Words, should be 10: %d\n", num_words );
    
    // Exercise 3 Tests: Uncomment the code below to test.
    printf("\n\nTesting capitalize\n\n");
    strcpy(W, S1);
    capitalize(W);
    printf("Capitalized, should be \"Raspberry\": \"%s\"\n", W );
    strcpy(W, S2);
    capitalize(W);
    printf("Capitalized, should be \"      \": \"%s\"\n", W );
    strcpy(W, S3);
    capitalize(W);
    printf("Capitalized, should be \"CSc 111 StrINGS LaB\": \"%s\"\n", W );
    strcpy(W, S4);
    capitalize(W);
    printf("Capitalized, should be \"   Raspberry Pear Pineapple Banana\": \"%s\"\n", W );
    strcpy(W, S5);
    capitalize(W);
    printf("Capitalized, should be \"   <-- Spaces At The Beginning, Spaces At The End --> \": \"%s\"\n", W );

    printf("\n\n\n\n");
}

void Test_PartFour() {
    printf("\n\n\t\tTESTING PART FOUR\t\t\n\n");
    char S1[] = "Hello         World ";
    char S2[] = "      "; //Contains 0 words, 6 characters
    char S3[] = "CSC    111    strINGS           LaB";
    char S4[] = "   raspberry    pear pineapple    banana  ";
    char S5[] = "   <-- spaces at the beginning, spaces at the end -->  ";

    //Make a new array to use as temporary storage.
    char W[1000];

    // Exercise 1 Tests: Uncomment the code below to test.
    printf("\n\nTesting remove_leading_spaces\n\n");
    remove_leading_spaces(S1, W);
    printf("With leading spaces removed, should be \"Hello         World \": \"%s\"\n", W);
    remove_leading_spaces(S2, W);
    printf("With leading spaces removed, should be \"\": \"%s\"\n", W);
    remove_leading_spaces(S3, W);
    printf("With leading spaces removed, should be \"CSC    111    strINGS           LaB\": \"%s\"\n", W);
    remove_leading_spaces(S4, W);
    printf("With leading spaces removed, should be \"raspberry    pear pineapple    banana  \": \"%s\"\n", W);
    remove_leading_spaces(S5, W);
    printf("With leading spaces removed, should be \"<-- spaces at the beginning, spaces at the end --> \": \"%s\"\n", W);

    // Exercise 2 Tests: Uncomment the code below to test.
    printf("\n\nTesting remove_trailing_spaces\n\n");
    remove_trailing_spaces(S1, W);
    printf("With trailing spaces removed, should be \"Hello         World\": \"%s\"\n", W);
    remove_trailing_spaces(S2, W);
    printf("With trailing spaces removed, should be \"\": \"%s\"\n", W);
    remove_trailing_spaces(S3, W);
    printf("With trailing spaces removed, should be \"CSC    111    strINGS           LaB\": \"%s\"\n", W);
    remove_trailing_spaces(S4, W);
    printf("With trailing spaces removed, should be \"   raspberry    pear pineapple    banana\": \"%s\"\n", W);
    remove_trailing_spaces(S5, W);
    printf("With trailing spaces removed, should be \"   <-- spaces at the beginning, spaces at the end -->\": \"%s\"\n", W);

    // Exercise 3 Tests: Uncomment the code below to test.
    printf("\n\nTesting condense_spaces\n\n");
    condense_spaces(S1, W);
    printf("With spaces condensed, should be \"Hello World\": \"%s\"\n", W);
    condense_spaces(S2, W);
    printf("With spaces condensed, should be \" \": \"%s\"\n", W);
    condense_spaces(S3, W);
    printf("With spaces condensed, should be \"CSC 111 strINGS LaB\": \"%s\"\n", W);
    condense_spaces(S4, W);
    printf("With spaces condensed, should be \" raspberry pear pineapple banana \": \"%s\"\n", W);
    condense_spaces(S5, W);
    printf("With spaces condensed, should be \" <-- spaces at the beginning, spaces at the end --> \": \"%s\"\n", W);

    printf("\n\n\n\n");
}