#include <stdio.h>

void Divide() {
    int number = 111;
    float divisor = 1000.000000;
    float result = number/divisor;

    //Add print statements here
    printf("\t Exercise 1 \t  The value of n is 111 and the divisor is 1000.000000. The result is %.6f \n", result);
    printf("\t Exercise Bonus \t The value of n is \"%d\" and the divisor is %f. The result is \"%.6f\" \n",number, divisor, result);
}

void PhysicsHW() {
     // Make a funct. for the formula, so theres no need to rewrite it 24/7 
    // store values of a, t, and v, and computes the result d using the formula ||| (velocity * time) + ((acceleration * (time * time)) / 2);
     float Formula(float acceleration, float time, float velocity) {
       return (velocity * time) + ((acceleration * (time * time)) / 2);
    }
    
    void Question1() {
        float a = 2.000000; // Acceleration
        float t = 2.000000; // Time
        float v = 3.000000; // Velocity
        float d = Formula(a, t, v); // Distance (result)
        printf("\n\n\n\n");// Spacer
        printf("\t For when, a = %f, t = %f, v = %f. The distance (or d) will be\n", a, t, v);
        printf("\t %f result\n", d);
    }
    
    void Question2() {
        float a = 6.000000; // Acceleration
        float t = 10.000000; // Time
        float v = 17.000000; // Velocity
        float d = Formula(a, t, v); // Distance (result)
        printf("\n\n\n\n");// Spacer
        printf("\t For when, a = %f, t = %f, v = %f. The distance (or d) will be\n", a, t, v);
        printf("\t %f result\n", d);
    }

    void Question3() {
        float a = 0.500000; // Acceleration
        float t = 1.060000; // Time
        float v = 11.100000; // Velocity
        float d = Formula(a, t, v); // Distance (result)
        printf("\n\n\n\n");// Spacer
        printf("\t For when, a = %f, t = %f, v = %f. The distance (or d) will be\n", a, t, v);
        printf("\t %f result\n", d);
    }

    Question1();
    Question2();
    Question3();
    printf("\n\n\n\n");// Spacer
}

int main(void) {

   // Call divide function
    Divide();

    // Call the phys hw
    PhysicsHW();
    return 0;
}