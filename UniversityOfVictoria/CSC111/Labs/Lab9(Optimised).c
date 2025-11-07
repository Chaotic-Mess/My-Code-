#include <stdio.h>
#include <string.h>

#define MAX_WORD_LEN 30
#define MAX_WORD_ARRAY (MAX_WORD_LEN+1)

                    //// PART 1 - Exericse 1: Define your order struct here: 
// Date Struct
typedef struct {
    int year;
    int month;
    int day;
} Date;

// Order Struct
typedef struct {
    char customer_name[MAX_WORD_ARRAY];
    int order_number;
    Date delivery_date;
    double charge;
} Order;



void print_order(Order* ord);
Order create_order(char customer_name[], int order_number, int year, int month, int day, double charge);
void add_to_charge(Order* ord, double amount);
void update_date(Order* ord, int new_year, int new_month, int new_day);
void print_orders(Order store_orders[], int num_orders);
double get_total_charges(Order store_orders[], int num_orders);
void add_surcharge(Order store_orders[], int num_orders, double surcharge);
void get_highest_paying_customer(Order store_orders[], int num_orders, char name[]);
int get_earliest_order(Order store_orders[], int num_orders);

int main(void){
      // Orders
    Order my_orders[] = {
        create_order("Rebecca Raspberry", 111, 2019, 3, 30, 6.10),
        create_order("Fiona Framboise", 116, 2019, 3, 29, 17.0),
        create_order("Neal Naranja", 225, 2020, 1, 6, 18.7),
        create_order("Hannah Hindbaer", 120, 2019, 3, 29, 12.0)
    };

        // PART 1 Tests
    printf("\nPART 1 TESTS: create_order and print_order:\n");
    for (int i = 0; i < 4; ++i) { // Loop through orders array
        print_order(&my_orders[i]);
    }

        // PART 2 Tests
    printf("\nPart 2 - Exercise 1: Test add_to_charge:\n");
    add_to_charge(&my_orders[0], 5.60);
    print_order(&my_orders[0]);

    printf("\nPart 2 - Exercise 2: Test update_date:\n");
    update_date(&my_orders[1], 2011, 4, 19);
    print_order(&my_orders[1]);

        // PART 3 Tests
    printf("\nPart 3 - Exercise 1: Test print_orders:\n");
    print_orders(my_orders, 4);

    printf("\nPart 3 - Exercise 2: Test get_total_charges:\n");
    printf("\tTotal Charges: $%.2f\n", get_total_charges(my_orders, 4));

    printf("\nPart 3 - Exercise 2: Test add_surcharge:\n");
    add_surcharge(my_orders, 4, 5.60);
    print_orders(my_orders, 4);

        // PART 4 Tests
    printf("\nPart 4 - Exercise 1: Test get_highest_paying_customer:\n");
    char highest_paying_customer[MAX_WORD_ARRAY];
    get_highest_paying_customer(my_orders, 4, highest_paying_customer);
    printf("\t%s\n", highest_paying_customer);

    printf("\nPart 4 - Exercise 2: Test get_earliest_order:\n");
    int earliest_index = get_earliest_order(my_orders, 4);
    printf("\t%d\n", earliest_index);
    print_order(&my_orders[earliest_index]);


    // TODO 6:
    // What do you notice about the orders in my_orders
    //      compared to order1, order2, order3 and order4?
    // Did the calls to add_to_charge and update_date change the values in my_orders?  Why?

    /* TODO 6 ANSWER
        The values in my_orders changed because add_to_charge and update_date 
        modified the original data directly, affecting charges and dates in the array 
        because arrays in C do not create copies of their elements when passed to functions,
        the functions operate on the original data.
    */


    return 0;
}

/**
 * @brief Outputs the details of the order pointed to by o_ptr
 * @param o_ptr a pointer to an instance of an Order
 */
void print_order(Order* o_ptr) {
   printf("\t\tOrder #%d by %s on %d/%d/%d. Price: $%.2f\n", 
           o_ptr->order_number, o_ptr->customer_name, 
           o_ptr->delivery_date.year, o_ptr->delivery_date.month, 
           o_ptr->delivery_date.day,  o_ptr->charge);
}

/**
 * @brief Creates and returns an Order with the given intial values
 * @param customer_name the customer name for the order to create, given 
 *                      as a null terminated string with length MAX_WORD_LEN
 * @param order_number the integer order number for the order
 * @param year the integer year for the order
 * @param month the integer month for the order
 * @param day the integer day for the order 
 * @param charge the charge for the order in dollars
 * @return a new Order instance with fields initialized to the given values
 * @pre day, month, and year combine to create a valid day of the year
 */
Order create_order(char customer_name[], int order_number, int year, int month, int day, double charge) {
  Order new_order;
    strncpy(new_order.customer_name, customer_name, MAX_WORD_LEN);
    new_order.customer_name[MAX_WORD_LEN] = '\0'; // Ensure null-termination
    new_order.order_number = order_number;
    new_order.delivery_date.year = year;
    new_order.delivery_date.month = month;
    new_order.delivery_date.day = day;
    new_order.charge = charge;
    return new_order;
}

/**
 * @brief Adds the given amount to the charge field in the Order pointed to by o_ptr
 * @param o_ptr A pointer to an instance of an Order
 * @param amount the amount in dollars to add to the order's charge field
 */
void add_to_charge(Order* o_ptr, double amount) {
    o_ptr->charge += amount; // Adds the amount to the charge field
}

/**
 * @brief Updates the date in the Order pointed to by o_ptr with given values
 * @param o_ptr A pointer to an instance of an Order
 * @param new_year the new integer year for the delivery date
 * @param new_month the new integer month for the delivery date
 * @param new_day the new integer day for the delivery date
 * @pre new_day, new_month, and new_year combine to create a valid day of the year
 */
void update_date(Order* o_ptr, int new_year, int new_month, int new_day) {
    o_ptr->delivery_date.year = new_year;
    o_ptr->delivery_date.month = new_month;
    o_ptr->delivery_date.day = new_day;
}

/**
 * @brief Outputs num_orders Orders in orders_array
 * @param orders_array the array of Order instances
 * @param num_orders the integer number of Orders in store_orders
 * @note This function should call print_order
 */
void print_orders(Order orders_array[], int num_orders) {
    for (int i = 0; i < num_orders; ++i) {
        print_order(&orders_array[i]); // Call print_order for each Order (MEM ADRESSS!!!!!)
    }
}

/**
 * @brief Calculates and returns the total charge of all Orders in orders_array
 * @param store_orders the array of Order instances
 * @param num_orders the integer number of Orders in store_orders
 * @return the sum of all of the array's order charges as a double
 */
double get_total_charges(Order orders_array[], int num_orders) {
    double total = 0.0;
    for (int i = 0; i < num_orders; ++i) {
        total += orders_array[i].charge; // Accumulate the charges
    }
    return total;
}

/**
 * @brief Adds the given surcharge to charge to all order in the array
 * @param orders_array the array of Order instances
 * @param num_orders the integer number of Orders in store_orders
 * @param surcharge the amount to add to each charge, in dollars as a double
 * @note This function should call add_to_charge
 */
void add_surcharge(Order orders_array[], int num_orders, double surcharge) {
    for (int i = 0; i < num_orders; ++i) {
        add_to_charge(&orders_array[i], surcharge);
    }
}

/**
 * @brief Gets the name of the customer with the highest charge in the array of orders
 * @param orders_array the array of Order instances
 * @param num_orders the integer number of Orders in store_orders
 * @param name[] the destination char array to place customer_name
 * @pre num_orders > 0 (ie. there is at least one item in the array)
 * @note If more than one Order with the highest charge, of those Orders,
 *       choose the Order that comes first in the array.
 */
void get_highest_paying_customer(Order orders_array[], int num_orders, char name[]) {
    int highest_index = 0;
    for (int i = 1; i < num_orders; ++i) {
        if (orders_array[i].charge > orders_array[highest_index].charge) {
            highest_index = i; // Update if a higher charge
        }
    }
    strcpy(name, orders_array[highest_index].customer_name); // Copy the name to the destination array
}

/**
 * @brief Finds the index of the earliest order in the array
 * @param orders_array the array of Order instances
 * @param num_orders the integer number of Orders in store_orders
 * @return the integer position number of the earliest order
 * @pre num_orders > 0 (ie. there is at least one item in the array)
 */
int get_earliest_order(Order orders_array[], int num_orders) {
    int earliest_index = 0;
    for (int i = 1; i < num_orders; ++i) {
        if (orders_array[i].delivery_date.year < orders_array[earliest_index].delivery_date.year ||
            (orders_array[i].delivery_date.year == orders_array[earliest_index].delivery_date.year &&
             orders_array[i].delivery_date.month < orders_array[earliest_index].delivery_date.month) ||
            (orders_array[i].delivery_date.year == orders_array[earliest_index].delivery_date.year &&
             orders_array[i].delivery_date.month == orders_array[earliest_index].delivery_date.month &&
             orders_array[i].delivery_date.day < orders_array[earliest_index].delivery_date.day)) {
            earliest_index = i; // Update if  earlier date found
        }
    }
    return earliest_index;
}
