/*
 * Program: C++ Vectors, Strings, Random, and IO (Lab3)
 * Author: Zac Matthias
 * Date: October 8, 2025
 * Description:
 *   This program contains 4 exercises demonstrating C++ vectors, strings,
 *   randomness, and console I/O — separated by functions:
 *     1. Exercise 1 : Mode of user-entered positive integers (vectors, maps)
 *     2. Exercise 2 : Month lookup by number (strings/arrays)
 *     3. Exercise 3 : Remove duplicates from a space-separated list (vectors & strings)
 *     4. Exercise 4 : Bean machine paths and final disposition (vectors, random, I/O)
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout, getline)
 *   - <vector> for dynamic arrays
 *   - <string> and <sstream> for string handling and parsing
 *   - <unordered_map>, <unordered_set> for frequency counting and deduplication
 *   - <random> and <chrono> for randomness
 *   - <limits> to safely clear input buffers
 * 
 *   PS E:\CSC116> g++ lab3.cpp -o lab3
 *   PS E:\CSC116> ./lab3
 */
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <chrono>
#include <limits>
 
void discard_rest_of_line() {
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

// ========== Exercise 1 (Vectors): Mode ==========
void exercise1() {
    using std::cin;
    using std::cout;

    cout << "Enter positive ints:\n";
    std::vector<int> inputs;
    int x;

    while (true) {
        cout << "> ";
        if (!(cin >> x)) {
            // Bad input -> clear and retry   (¬_¬)
            cin.clear();
            discard_rest_of_line();
            continue;
        }
        if (x <= 0) break;           // stop on non-positive    ( ﾟдﾟ)つ Bye
        inputs.push_back(x);
    }

    int mode = 0; // default if no valid positives were entered
    if (!inputs.empty()) {
        std::unordered_map<int, int> freq;
        std::unordered_map<int, int> firstIndex;
        for (size_t i = 0; i < inputs.size(); ++i) {
            int v = inputs[i];
            ++freq[v];
            if (!firstIndex.count(v)) firstIndex[v] = static_cast<int>(i);
        }
        int bestCount = 0;
        int bestFirst = static_cast<int>(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            int v = inputs[i];
            int c = freq[v];
            int f = firstIndex[v];
            if (c > bestCount || (c == bestCount && f < bestFirst)) {
                bestCount = c;
                bestFirst = f;
                mode = v;
            }
        }
    }

    std::cout << "\nThe mode is " << mode << "\n\n";
}

// ========== Exercise 2 (Strings): Month by number ==========
void exercise2() {
    using std::cin;
    using std::cout;

    cout << "Enter a month number (1-12):\n> ";
    int m;
    while (!(cin >> m)) {
        cin.clear();
        discard_rest_of_line();
        cout << "> ";
    }

    const char* months[12] = {
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    };

    if (m >= 1 && m <= 12) {
        cout << months[m - 1] << "\n\n";
    } else {
        cout << "Invalid month number\n\n";
    }

    // prepare for next getline-based input
    discard_rest_of_line();
}

// ========== Exercise 3 (Vectors & Strings): Remove duplicates ==========
void exercise3() {
    using std::cin;
    using std::cout;
    using std::string;

    cout << "Numbers:\n> ";
    string line;
    std::getline(cin, line);

    std::stringstream ss(line);
    std::unordered_set<int> seen;
    std::vector<int> output; // keep first-seen order (order not important because thats what the instructions say 🤔)

    int n;
    while (ss >> n) {
        if (!seen.count(n)) {
            seen.insert(n);
            output.push_back(n);
        }
    }

    cout << "Without duplicates";
    if (!output.empty()) cout << " ";
    for (size_t i = 0; i < output.size(); ++i) {
        cout << output[i];
        if (i + 1 < output.size()) cout << " ";
    }
    cout << "\n\n";
}

// ========== Exercise 4 (Vectors & IO): Bean machine ==========
void exercise4() {
    using std::cin;
    using std::cout;

    cout << "Number of balls to drop\n> ";
    int balls;
    while (!(cin >> balls)) {
        cin.clear();
        discard_rest_of_line();
        cout << "> ";
    }

    cout << "Number of slots:\n> ";
    int slots;
    while (!(cin >> slots)) {
        cin.clear();
        discard_rest_of_line();
        cout << "> ";
    }

    if (balls < 0) balls = 0;
    if (slots < 1) slots = 1; // ensure at least one slot (otherwise infinite loop) (╯°□°）╯︵ ┻━┻

    // RNG setup (ಥ _ ಥ)
    std::mt19937 rng(static_cast<unsigned int>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> bit(0, 1); // 0 = L, 1 = R

    cout << "\nPaths:\n";
    std::vector<int> bins(static_cast<size_t>(slots), 0);

    for (int b = 0; b < balls; ++b) {
        int rights = 0;
        std::string path;
        int steps = slots - 1; // number of decisions (pins) 
        for (int s = 0; s < steps; ++s) {
            int r = bit(rng);
            if (r == 0) {
                path.push_back('L');
            } else {
                path.push_back('R');
                ++rights;
            }
        }
        cout << path << "\n";
        if (rights >= 0 && rights < slots) {
            ++bins[static_cast<size_t>(rights)];
        }
    }

    cout << "\nDisposition:\n\n";
    // Horizontal histogram: one line per slot, 'o' repeated by count in that slot  (this was ridiculously hard to figure out for some reason ಠ_ಠ)
    for (int i = 0; i < slots; ++i) {
        for (int k = 0; k < bins[static_cast<size_t>(i)]; ++k) {
            cout << 'o';
        }
        cout << "\n";
    }
}

int main() {
    exercise1();
    exercise2();
    exercise3();
    exercise4();
    return 0;
}