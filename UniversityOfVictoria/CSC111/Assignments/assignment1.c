/*
     Purpose: Creates ACSII art for a Jet and a Hourglass. And outputs the surface area of a cylinder.
     @author: Zac Matthias

     Notes:

     \" Escape sequence for double quotes
     \\ Escape sequence for backslash

    
    Calculate the SA of a cylinder using formula:
    Surface Area = 2 * (Area of Top) + (Area of Walls)
    ↓
    Area of Top = π * (r^2)
    Area of Walls = Circumference * Height

*/

#include <stdio.h>
#define PI 3.14

// Function 1: Prints a jet
void print_jet() {
    /*
    Prints an ASCII jet figure.
    */
    printf("                          /\\\n");
    printf("                         |  |\n");
    printf("                         |  |\n");
    printf("                        .'  '.\n");
    printf("                        |    |\n");
    printf("                        |    |\n");
    printf("                        | /\\ |\n");
    printf("                      .' |__|'.\n");
    printf("                      |  |  |  |\n");
    printf("                     .'  |  |  '.\n");
    printf("                /\\   |   \\__/   |   /\\\n");
    printf("               |  |  |   |  |   |  |  |\n");
    printf("           /|  |  |,-\\   |  |   /-,|  |  |\\\n");
    printf("           ||  |,-'   |  |  |  |   '-,|  ||\n");
    printf("           ||-'       |  |  |  |       '-||\n");
    printf("|\\     _,-'           |  |  |  |           '-,_     /|\n");
    printf("||  ,-'   _           |  |  |  |               '-,  ||\n");
    printf("||-'    =(*)=         |  |  |  |                  '-||\n");
    printf("||                    |  \\  /  |                    ||\n");
    printf("|\\________....--------\\   ||   /--------....________/|\n");
    printf("                      /|  ||  |\\\n");
    printf("                     / |  ||  | \\\n");
    printf("                    /  |  \\/  |  \\\n");
    printf("                   /   |      |   \\\n");
    printf("                 //   .|      |.   \\\\\n");
    printf("               .' |_./ |      | \\._| '.\n");
    printf("              /     _.-|||  |||-._     \\\n");
    printf("              \\__.-'   \\||/\\||/   '-.__/ \n");

    // Untitled by Jasin 
    // https://www.asciiart.eu/vehicles/airplanes
}

// Function 2: Prints an hourglass
void print_hourglass() {
    /*
    Prints an ASCII hourglass figure.
    */  
    printf("-8-=-=-=-=-8-\n");
    printf(" | ,dOOOb, |\n");
    printf(" |d       b|\n");
    printf(" |\\::. .::/|           S A N D S\n");
    printf(" | \\:::::/ |\n");
    printf(" |  \\:::/  |\n");
    printf(" |   x:x   |              O F\n");
    printf(" |  / . \\  |\n");
    printf(" | /  .  \\ |\n");
    printf(" |/  .:.  \\|            T I M E\n");
    printf(" |Y.:::::.Y|\n");
    printf(" | \"YOOOY\" |\n");
    printf("-8-=-=-=-=-8-\n");

    // Sands of Time 
    // https://www.asciiart.eu/miscellaneous/hourglass
}

// Function to alternate figures and print spacer lines
void print_logo() {
    /*
    Alternates the ASCII art figures and adds a spacer line.
    */
    printf("/~~~~~~~~\\\n\n");
    print_hourglass();
    printf("/~~~~~~~~\\\n\n");
    print_jet();
    printf("/~~~~~~~~\\\n\n");
    print_hourglass();
    printf("/~~~~~~~~\\\n\n");
    print_jet();
    printf("/~~~~~~~~\\\n\n");
}

// Function to calculate and print the surface area of a cylinder
void print_surface_area() {
    int height = 6;
    int diameter = 5;
    double radius = diameter / 2.0;
    
    // Circumference of the cylinder
    double circumference = 2 * PI * radius;
    
    // Area of the top of the cylinder
    double area_top = PI * (radius * radius);
    
    // Area of the walls of the cylinder
    double area_walls = circumference * height;
    
    // Calculate the total surface area (top + bottom + walls)
    double total_surface_area = 2 * area_top + area_walls;
    
    // Print the result with 2 decimals (%.2f)
    printf("%.2f\n", total_surface_area);
}

// Calls the Alternating logo function and the surface area function.
int main(void) {
    print_logo();
    print_surface_area();
    
    return 0;
}
