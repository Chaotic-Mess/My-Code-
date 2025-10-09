/*
 * Program: C++ Exercises Collection
 * Author: Zac Matthias
 * Date: 2025-09-15
 * Description:
 *   This program contains four exercises demonstrating basic C++ concepts:
 *     1. Exercise 1: Finds the maximum of user-entered positive doubles.
 *     2. Exercise 2: Plays a Rock-Paper-Scissors game against the computer
 *        using ASCII art for visualization.
 *     3. Exercise 3: Calculates the standard deviation of a user-provided set of numbers.
 *     4. Exercise 4: Displays a simple horizontal bar chart using predefined values.
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout)
 *   - <vector> for dynamic arrays
 *   - <cmath> for mathematical functions (sqrt)
 *   - <cstdlib> for random number generation (rand, srand)
 *   - <ctime> for seeding random numbers (time)
 *   - <cctype> for character handling (toupper)
 * 
 *   PS E:\CSC116> g++ Lab1.cpp -o Lab1
 *   PS E:\CSC116> ./Lab1
 */


#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cctype>

using namespace std;

// ------------------- Exercise 1 -------------------
void exercise1() {
    double num;
    double max = 0.0;

    cout << "\n--- Exercise 1: Maximum of positive doubles ---\n";
    cout << "Enter positive doubles:" << endl;
    while (true) {
        cout << "> ";
        if (!(cin >> num)) {
            cin.clear();  // clear the error state
            cin.ignore(10000, '\n');  // discard invalid input
            cout << "Invalid input. Please enter a number.\n";
            continue;
        }
        if (num <= 0) break;
        if (num > max) max = num;
    }
    cout << "\nThe maximum is " << max << endl;
}

// ------------------- Exercise 2 -------------------
void printRock() {
    cout << " __.--.--._\n"
         << "/  | _|  | `|\n"
         << "|  |` |  |  |\n"
         << "| /'--:--:__/\n"
         << "|/  /      |\n"
         << "(  ' \\     |\n"
         << " \\    `.  /\n"
         << "  |      |\n"
         << "  |      |\n";
}

void printPaper() {
    cout << "    --.--.\n"
         << "   |  |  |\n"
         << ".\"\"|  |  |_\n"
         << "|  |  |  | `|\n"
         << "|  | _|  |  |\n"
         << "|  |` )  |  |\n"
         << "| /'  /     /\n"
         << "|/   /      |\n"
         << "(   ' \\     |\n"
         << "\\      `.  /\n"
         << " |        |\n"
         << " |        |\n";
}

void printScissors() {
    cout << ".\"\".   .\"\",\n"
         << "|  |  /  /\n"
         << "|  | /  /\n"
         << "|  |/ .--._\n"
         << "|    _|  | `|\n"
         << "|  /` )  |  |\n"
         << "| /  /'--:__/\n"
         << "|/  /      |\n"
         << "(  ' \\     |\n"
         << " \\    `.  /\n"
         << "  |      |\n"
         << "  |      |\n";
}

void exercise2() {
    srand(time(0));
    char player;
    char choices[3] = {'R', 'P', 'S'};

    cout << "\n--- Exercise 2: Rock-Paper-Scissors ---\n";
    cout << "Welcome to the Rock-Paper-Scissors game!\n";

    while (true) {
        cout << "Select your element:\n\tR/r - rock\n\tP/p - paper\n\tS/s - scissors\n> ";
        cin >> player;
        player = toupper(player);
        if (player == 'R' || player == 'P' || player == 'S') break;
        cout << "Invalid choice, try again.\n";
    }

    char computer = choices[rand() % 3];

    if (player == 'R') printRock();
    if (player == 'P') printPaper();
    if (player == 'S') printScissors();
    cout << "Player\n\n";

    if (computer == 'R') printRock();
    if (computer == 'P') printPaper();
    if (computer == 'S') printScissors();
    cout << "Computer\n\n";

    if (player == computer) {
        cout << "It's a tie!\n";
    } else if ((player == 'R' && computer == 'S') ||
               (player == 'P' && computer == 'R') ||
               (player == 'S' && computer == 'P')) {
        cout << "Player won! (" 
             << (player=='R'?"rock":player=='P'?"paper":"scissors")
             << " beats "
             << (computer=='R'?"rock":computer=='P'?"paper":"scissors")
             << ")\n";
    } else {
        cout << "Computer won! (" 
             << (computer=='R'?"rock":computer=='P'?"paper":"scissors")
             << " beats "
             << (player=='R'?"rock":player=='P'?"paper":"scissors")
             << ")\n";
    }
}

// ------------------- Exercise 3 -------------------
void exercise3() {
    int N;
    cout << "\n--- Exercise 3: Standard Deviation ---\n";
    cout << "How many numbers:\n> ";
    cin >> N;

    vector<double> numbers(N);
    cout << "Insert " << N << " numbers:\n";
    for (int i = 0; i < N; ++i) {
        cout << "> ";
        cin >> numbers[i];
    }

    double sum = 0;
    for (double x : numbers) sum += x;
    double mean = sum / N;

    double variance = 0;
    for (double x : numbers) variance += (x - mean) * (x - mean);
    variance /= N;

    double stdDev = sqrt(variance);
    cout << "\nTheir std is " << stdDev << endl;
}

// ------------------- Exercise 4 -------------------
void exercise4() {
    cout << "\n--- Exercise 4: Horizontal Bar Chart ---\n";
    vector<int> bars = {1, 3, 5, 0, 2};
    cout << "Bar chart:\n";
    for (int length : bars) {
        for (int i = 0; i < length; ++i) 
            cout << "=";
        cout << "\n";
    }
}

// ------------------- Main -------------------
int main() {
    exercise1();
    exercise2();
    exercise3();
    exercise4();
    return 0;
}
