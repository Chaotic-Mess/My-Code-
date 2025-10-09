/*
 * Program: C++ String Exercises Collection (Lab2)
 * Author: Zac Matthias
 * Date: 2025-09-25
 * Description:
 *   This program contains four exercises demonstrating basic C++ string manipulation:
 *     1. Exercise 1 : Repeats a given string n times through concatenation.
 *     2. Exercise 2 : Prompts user for input and displays the string in reverse order.
 *     3. Exercise 3 : Checks if a user-entered string is a palindrome,
 *        ignoring spaces and capitalization.
 *     4. Exercise 4 : Reads multiple lines of text and counts
 *        the frequency of each word, displaying results in alphabetical order.
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout, getline)
 *   - <string> for string manipulation and operations
 *   - <algorithm> for string transformation functions (reverse, transform)
 *   - <map> for storing and organizing word frequency data
 *   - <sstream> for string stream processing and word parsing
 * 
 *   PS E:\CSC116> g++ Lab2.cpp -o Lab2
 *   PS E:\CSC116> ./Lab2
 */

#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
using namespace std;

// Exercise 1: Repeat a string n times
void exercise1() {
    cout << "=== Exercise 1: String Repetition ===" << endl;
    
    string str = "hello";
    int times = 5;
    string output = "";
     
    for (int i = 0; i < times; i++) {
        output += str;
    }
    
    cout << "string = \"" << str << "\"" << endl;
    cout << "times = " << times << endl;
    cout << "output = \"" << output << "\"" << endl;
    cout << endl;
}

// Exercise 2: Reverse a string
void exercise2() {
    cout << "=== Exercise 2: String Reversal ===" << endl;
    
    string input;
    cout << "Enter string:" << endl;
    cout << "> ";
    getline(cin, input);
    
    string reversed = "";

    for (int i = input.length() - 1; i >= 0; i--) {
        reversed += input[i];
    }
    
    cout << "Reverse is string: " << reversed << endl;
    cout << endl;
}

// Exercise 3: Check if string is palindrome
void exercise3() {
    cout << "=== Exercise 3: Palindrome Check ===" << endl;
    
    string input;
    cout << "Enter string:" << endl;
    cout << "> ";
    getline(cin, input);
    
    string cleaned = "";
    for (char c : input) {
        if (c != ' ') {
            cleaned += tolower(c);
        }
    }
    
    string reversed = cleaned; 
    reverse(reversed.begin(), reversed.end());
    
    if (cleaned == reversed) {
        cout << input << " is a palindrome" << endl;
    } else {
        cout << input << " is not a palindrome" << endl;
    }
    cout << endl;
}

// Exercise 4: Word frequencies
void exercise4() {
    cout << "=== Exercise 4: Word Frequencies ===" << endl;
    
    cout << "Input" << endl;
    cout << "> ";
    
    string line;
    string allText = "";
    
    
    while (getline(cin, line) && !line.empty()) {// Read multiple lines until empty line
        allText += line + " ";
        cout << "> ";
    }
    
    const int MAX_WORDS = 100;
    string words[MAX_WORDS];
    int counts[MAX_WORDS];
    int numUniqueWords = 0;
    
    stringstream ss(allText); // Convert to lowercase and split into words
    string word;
    
    while (ss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower); // Convert to lowercase
        
        // Check if word already exists in our array
        bool found = false;
        for (int i = 0; i < numUniqueWords; i++) {
            if (words[i] == word) {
                counts[i]++; 
                found = true;
                break;
            }
        }
        
        // If word not found and we have space, add it as new word
        if (!found && numUniqueWords < MAX_WORDS) {
            words[numUniqueWords] = word;
            counts[numUniqueWords] = 1;
            numUniqueWords++;
        } else if (!found) {
            cout << "Warning: Maximum word limit reached!" << endl;
        }
    }
    
    cout << "Word Frequencies:" << endl;
    for (int i = 0; i < numUniqueWords; i++) {
        cout << words[i] << " " << counts[i] << endl;
    }
}

int main() {
    exercise1();   
    exercise2();   
    exercise3();       
    exercise4();

    return 0;
}