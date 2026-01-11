/*
 * Program: C++ Sets (Lab7)
 * Author: Zac Matthias
 * Date: 
 * Description:
 *   This program contains 4 exercises demonstrating C++ sets:
 *     1. Exercise 1 : Create a set of all punctuation characters used in a vector of input strings
 *     2. Exercise 2 : Reverse an input string and print it
 *     3. Exercise 3 : Remove all duplicates from a vector of integers using a set
 *     4. Exercise 4 : Reads A String And Prints The Word Frequency For Each Word VIA Map
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout, getline)
 *   - <vector> for using the vector container
 *   - <set> for using the set container
 *   - <cctype> for character handling functions (ispunct)
 *   - <string> for using the string class
 *   - <map> for storing and organizing word frequency data
 * 
 *   PS E:\CSC116> g++ Lab7.cpp -o Lab7
 *   PS E:\CSC116> ./Lab7
 */
#include <iostream>
#include <vector>
#include <set>
#include <cctype>
#include <string>
#include <map>
 
// ========== Exercise 1 : Find Punctuation Characters ==========
void Exercise1(std::vector<std::string> &inputLines) {
    std::set<char> punctuationSet;
    for (const auto &line : inputLines) {
        for (const auto &ch : line) {
            if (std::ispunct(static_cast<unsigned char>(ch))) {
                punctuationSet.insert(ch);
            }
        }
    }

    std::cout << "Punctuation characters found: ";
    for (const auto &punct : punctuationSet) {
        std::cout << punct << ' ';
    }
    std::cout << std::endl;
}

// ========== Exercise 1 : Reverse Input String ==========
void Exercise2() {
    std::cout << "Enter string: ";
    
    std::string input;
    std::getline(std::cin, input);

    std::cout << "Reversed is string: "; 
    for(int i = input.length() - 1; i >= 0; i--) {
        std::cout << input[i];
    }
    std::cout << std::endl;
}

// ========== Exercise 3 : Removes All Duplicates VIA Set ==========
void Exercise3(std::vector<int> numbers) {
    std::set<int> uniqueNumbers;
    for (const auto &num : numbers) {
        uniqueNumbers.insert(num);
    }

    std::cout << "Unique numbers: ";
    for (const auto &num : uniqueNumbers) {
        std::cout << num << ' ';
    }
    std::cout << std::endl;
}

// ========== Exercise 4: Reads A String And Prints The Word Frequency For Each Word VIA Map ==========
void Exercise4() {
    std::cout << "Enter string(s)\n";
    std::string line, text;
    while (std::getline(std::cin, line) && !line.empty()) {
        if (!text.empty()) text += ' ';
        text += line;
    }

    for (char &c : text) c = std::tolower(static_cast<unsigned char>(c));

    std::map<std::string,int> freq;
    std::string w;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) || std::ispunct(static_cast<unsigned char>(c))) {
            if (!w.empty()) { ++freq[w]; w.clear(); }
        } else w.push_back(c);
    }
    if (!w.empty()) ++freq[w];

    std::cout << "Word Frequencies:\n";
    for (const auto &p : freq) std::cout << p.first << ": " << p.second << '\n';
}

int main() {
    std::vector<std::string> inputLines = {"Sea", "Shore,", "sea", "shell's.", "SHORE.", "line!"};
    Exercise1(inputLines);

    Exercise2();

    std::vector<int> nums = {1, 1, 1, 2, 6, 5, 1, 1, 6};
    Exercise3(nums);

    Exercise4();
    return 0;
}