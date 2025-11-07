/** 
 * This program performs text censorship in an input file by replacing specified words with asterisks ("***"). The main functions include checking if characters are valid word characters, converting 
 *  words to lowercase for case-insensitive comparisons, searching for words in a list of forbidden terms, and censoring matched words in the output file.
 * 
 * @author Zac Matthias
 */

#include <stdio.h>
#include <string.h> // For strlen function ( SIMPLE )
#include <strings.h> // For strlen function ( ADVANCED )
#include <ctype.h> // for the tolower

#define MAX_WORD_LEN        50
#define CHAR_ARRAY_WIDTH    MAX_WORD_LEN+1 // allow for null terminator character

int is_word_character(char ch);
void to_lowercase(char dest[], char word[]);
int is_word_in_terms(char word[CHAR_ARRAY_WIDTH], char terms[][CHAR_ARRAY_WIDTH], int len_terms);
void censor(char infilename[], char outfilename[], char terms[][CHAR_ARRAY_WIDTH], int len_terms);

int main() {
    // TESTING is_word_character
    printf("\t\t\n\nTesting is_word_character function:\n\n");
    printf("is_word_character('a'): %d (Expected: 1)\n", is_word_character('a'));
    printf("is_word_character('-'): %d (Expected: 1)\n", is_word_character('-'));
    printf("is_word_character(' '): %d (Expected: 0)\n", is_word_character(' '));
    printf("is_word_character('Z'): %d (Expected: 1)\n", is_word_character('Z'));
    printf("is_word_character(','): %d (Expected: 0)\n", is_word_character(','));

    // TESTING to_lowercase
    printf("\t\t\n\nTesting to_lowercase function:\n\n");
    char dest[CHAR_ARRAY_WIDTH];
    
    to_lowercase(dest, "Hello");
    printf("to_lowercase('Hello'): %s (Expected: hello)\n", dest);
    
    to_lowercase(dest, "WORLD");
    printf("to_lowercase('WORLD'): %s (Expected: world)\n", dest);
    
    to_lowercase(dest, "MixedCASE");
    printf("to_lowercase('MixedCASE'): %s (Expected: mixedcase)\n", dest);
    
    to_lowercase(dest, "12345");
    printf("to_lowercase('12345'): %s (Expected: 12345)\n", dest);
    
    to_lowercase(dest, "HELLO-WORLD");
    printf("to_lowercase('HELLO-WORLD'): %s (Expected: hello-world)\n", dest);


    // TESTING is_word_in_terms
    printf("\t\t\n\nTesting is_word_in_terms function:\n\n");

    char is_word_in_terms_terms1[][CHAR_ARRAY_WIDTH] = {"apple", "banana", "orange"};
    char is_word_in_terms_terms2[][CHAR_ARRAY_WIDTH] = {"cat", "dog", "fish"};
    char is_word_in_terms_terms3[][CHAR_ARRAY_WIDTH] = {"HELLO", "WORLD", "TEST"};
    char single_term_array[][CHAR_ARRAY_WIDTH] = {"unique"};
    
    printf("is_word_in_terms(\"banana\", terms1, 3): %d (Expected: 1)\n", is_word_in_terms("banana", is_word_in_terms_terms1, 3));
    printf("is_word_in_terms(\"grape\", terms1, 3): %d (Expected: 0)\n", is_word_in_terms("grape", is_word_in_terms_terms1, 3));
    printf("is_word_in_terms(\"APPLE\", terms1, 3): %d (Expected: 1)\n", is_word_in_terms("APPLE", is_word_in_terms_terms1, 3));
    printf("is_word_in_terms(\"world\", terms3, 3): %d (Expected: 1)\n", is_word_in_terms("world", is_word_in_terms_terms3, 3));
    printf("is_word_in_terms(\"dog\", terms2, 0): %d (Expected: 0)\n", is_word_in_terms("dog", is_word_in_terms_terms2, 0));
    printf("is_word_in_terms(\"unique\", single_term_array, 1): %d (Expected: 1)\n", is_word_in_terms("unique", single_term_array, 1));
    printf("is_word_in_terms(\"notfound\", single_term_array, 1): %d (Expected: 0)\n", is_word_in_terms("notfound", single_term_array, 1));
    printf("is_word_in_terms(\"Cat\", terms2, 3): %d (Expected: 1)\n", is_word_in_terms("Cat", is_word_in_terms_terms2, 3));

    
    // TESTING CENSOR 
    printf("\t\t\n\n Testing censor \n\n");
   // Test 1: Censoring in happy_small.txt with "air" and "with"
    censor("happy_small.txt", "happy_small_out_terms1.txt", (char[][CHAR_ARRAY_WIDTH]){"air", "with"}, 2);
    
    // Test 2: No trailing newline in happy_small.txt with "air" and "with"
    censor("happy_small.txt", "happy_small_no_trailing_newline_out_terms3.txt", (char[][CHAR_ARRAY_WIDTH]){"air", "with"}, 2);
    
    // Test 3: Censoring in happy.txt with "air" and "with"
    censor("happy.txt", "happy_out_terms2.txt", (char[][CHAR_ARRAY_WIDTH]){"air", "with"}, 2);

    // Test 4: Censoring in happy.txt with "cat" and "dog"
    censor("happy.txt", "happy_out_terms3.txt", (char[][CHAR_ARRAY_WIDTH]){"cat", "dog"}, 2);

    // Test 5: Censoring in happy_small.txt with "air", "with", "dog"
    censor("happy_small.txt", "happy_small_out_terms4.txt", (char[][CHAR_ARRAY_WIDTH]){"air", "with", "dog"}, 3);

    // Test 6: Censoring in happy.txt with "world", "happy", "day"
    censor("happy.txt", "happy_out_terms1.txt", (char[][CHAR_ARRAY_WIDTH]){"world", "happy", "day"}, 3);

    return 0;
}

/**
 * @brief Determines if the given character is considered a word character.
 * @param ch The character to analyze.
 * @return 1 if the character is a letter, hyphen, or apostrophe; 0 otherwise.
 */
int is_word_character(char ch) {
    return (isalpha(ch) || ch == '-' || ch == '\'');
}

/**
 * @brief Converts a word to lowercase and stores it in the destination array.
 * @param dest The destination array where the lowercase word will be stored.
 * @param word The input word to convert.
 */
void to_lowercase(char dest[], char word[]) {
    // Loop through each character in the input word
    for (int i = 0; word[i] != '\0'; i++) {
        // Convert to lowercase and copy to dest
        dest[i] = (char)tolower((unsigned char)word[i]);
    }
    // Null terminate the destination string
    dest[strlen(word)] = '\0';
}

/**
 * @brief Checks if a given word matches any term in the terms array.
 * @param word The word to search for in the terms array.
 * @param terms The array of terms to check against.
 * @param len_terms The number of terms in the array.
 * @return 1 if the word is found in the terms array; 0 otherwise.
 */
int is_word_in_terms(char word[CHAR_ARRAY_WIDTH], char terms[][CHAR_ARRAY_WIDTH], int len_terms) {
    int i = 0;
    while (i < len_terms) {
        if (strcasecmp(word, terms[i])==0){
           return 1; // Match found
        }
        i++;
    }
    return 0; // No match found
}

/**
 * @brief Censors words in the input file by replacing forbidden words with "***".
 * @param infilename The name of the input file.
 * @param outfilename The name of the output file.
 * @param terms The array of forbidden words.
 * @param len_terms The number of forbidden words in the array.
 */
void censor(char infilename[], char outfilename[], char terms[][CHAR_ARRAY_WIDTH], int len_terms) {
    FILE *inputFile = fopen(infilename, "r");
    FILE *outputFile = fopen(outfilename, "w");

    if (inputFile == NULL || outputFile == NULL) {
        return; // Exit if file opening fails
    }

    char word[CHAR_ARRAY_WIDTH];
    int index = 0;
    int ch;

    // Read the input file character by character
    while ((ch = fgetc(inputFile)) != EOF) {
        if (isalnum(ch) || ch == '-' || ch == '\'') { // Check for word characters
            if (index < MAX_WORD_LEN) {
                word[index++] = (char)ch; // Collect characters into the word
            }
        } else {
            if (index > 0) { // Process completed word
                word[index] = '\0'; // Null terminate the word

                if (is_word_in_terms(word, terms, len_terms)) {
                    fprintf(outputFile, "***"); // Censor matched words
                } else {
                    fprintf(outputFile, "%s", word); // Write the original word
                }
                index = 0; // Reset for the next word
            }
            fputc(ch, outputFile); // Write non-word characters
        }
    }

    // Handle the last word if necessary
    if (index > 0) {
        word[index] = '\0'; // Null terminate
        if (is_word_in_terms(word, terms, len_terms)) {
            fprintf(outputFile, "***");
        } else {
            fprintf(outputFile, "%s", word);
        }
    }

    fclose(inputFile);
    fclose(outputFile);
}
