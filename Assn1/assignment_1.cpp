/*
 * Program: C++ Text Justifier (Assignment 1)
 * Author: Zac Matthias
 * Date: 2025-09-30
 * Description:
 *   Reads words from standard input and outputs a justified representation
 *   of the text with a fixed line width of 30 characters. Justification is
 *   achieved by varying the number of spaces between words (no hyphenation).
 *
 *   Exercises implemented:
 *     1. Input      : read words from std::cin until EOF
 *     2. Storing    : build and STORE each output line in a vector, then print
 *     3. Justifying : compute spaces exactly as spec (integer division per gap)
 *
 * Libraries used:
 *   - <iostream> for input/output (std::cin, std::cout)
 *   - <vector>   for dynamic storage of words and lines
 *   - <string>   for text manipulation
 *
 *   PS E:\CSC116> g++ assignment_1.cpp -o assignment_1
 *   PS E:\CSC116> ./assignment_1
 * 
 *  (Paste text into Terminal; press Ctrl-Z then Enter to signal EOF on Windows.)
 */

#include <iostream>
#include <vector>
#include <string>

//// Forward Declaration (for organizations sake)
std::string justifyLine(const std::vector<std::string>& line_words, std::size_t width); // Exercise 3: Justification 

//// Exercise 1: Input 
std::vector<std::string> read_words_from_stdin() {
    std::vector<std::string> words;
    std::string w;
    while (std::cin >> w) {
        words.push_back(w);
    }
    return words;
}

// Helper for E2: join words with single spaces (used for the last line / left align)
std::string leftAlign(const std::vector<std::string>& line_words) {
    std::string line;
    for (std::size_t i = 0; i < line_words.size(); ++i) {
        if (i) line.push_back(' ');
        line += line_words[i];
    }
    return line;
}

//// Exercise 2: Storing & Building Lines 
// Build ALL output lines first (store in a vector), then print. The last line
// is left-aligned with single spaces. Very long words (> width) are placed on
// their own line, unbroken (no hyphenation), as per the suggested algorithm.
std::vector<std::string> build_output_lines(const std::vector<std::string>& words, std::size_t width) {
    std::vector<std::string> output;             // stores every line to print later
    std::vector<std::string> L;                  // current line words
    std::size_t letters_in_L = 0;                // total letters in L (no spaces)

    for (const auto& w : words) {
        const std::size_t n_words = L.size();
        const bool fits = (n_words + letters_in_L + w.size()) <= width; // spec test

        if (fits) {
            L.push_back(w);
            letters_in_L += w.size();
        } else {
            // Finalize current line: fully justify (except single-word case handled inside)
            if (!L.empty()) {
                output.push_back(justifyLine(L, width));
            }
            L.clear();
            letters_in_L = 0;

            // Handle the new word according to the spec
            if (w.size() > width) {
                // Output this long word as its own line immediately (store it)
                output.push_back(w);
            } else {
                L.push_back(w);
                letters_in_L = w.size();
            }
        }
    }

    // After processing all words, output the last line left-aligned
    if (!L.empty()) {
        output.push_back(leftAlign(L));
    }

    return output;
}

//// Exercise 3: Justification 
// Build a fully-justified line exactly `width` characters long using the
// algorithm specified in Exercise 3 (integer division per remaining gap).
std::string justifyLine(const std::vector<std::string>& line_words, std::size_t width) {
    if (line_words.empty()) return std::string();
    if (line_words.size() == 1) return line_words.front();

    // Total letters
    std::size_t letters = 0;
    for (const auto& s : line_words) letters += s.size();

    // Spaces to distribute
    std::size_t spaces_remaining = (width > letters) ? (width - letters) : 0;

    std::string line;
    line.reserve(width);

    std::size_t words_remaining = line_words.size();
    for (std::size_t i = 0; i < line_words.size(); ++i) {
        const std::string& w = line_words[i];
        line += w;
        --words_remaining; // words remaining AFTER w
        if (words_remaining == 0) break; // last word: no spaces after
        std::size_t s = (words_remaining > 0) ? (spaces_remaining / words_remaining) : 0; // integer division
        line.append(s, ' ');
        if (spaces_remaining >= s) spaces_remaining -= s; else spaces_remaining = 0; // safety
    }
    return line;
}

int main() {
    const std::size_t LINE_WIDTH = 30; // fixed line width as per spec
    const std::vector<std::string> words = read_words_from_stdin();
    const std::vector<std::string> lines = build_output_lines(words, LINE_WIDTH);

    for (const auto& line : lines) {
        std::cout << line << '\n';
    }

    return 0;
}
