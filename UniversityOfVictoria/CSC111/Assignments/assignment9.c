/**
 * @brief Implements a linked list with operations such as adding elements, 
 *        filtering based on thresholds, summing squares, and checking factor relationships.
 *        Demonstrates efficient list manipulation for various computational tasks.
 * @author Zac Matthias
 */


#include <stdio.h>
#include <stdlib.h>

typedef struct IntNode {
    int value;
    struct IntNode* next; //Pointer to the next node in the list, or NULL if no further nodes exist.
} IntNode;

typedef struct {
    IntNode* front; //Pointer to the first node in the list, or NULL if the list is empty.
} IntList;

// Functions provided for you:
IntNode* create_node(int num, IntNode* next);
IntList create_list();
void insert_front(IntList* list_ptr, int num);
void insert_back(IntList* list_ptr, int num);
void print_list(IntList* list_ptr);
void free_list(IntList* list_ptr);
int is_factor(int n1, int n2);

// Function Prototypes
int sum_squares(IntList* list_ptr);
void add_to_all(IntList* list_ptr, int n);
int get_all_below(IntList* src_list, IntList* dest_list, int threshold);
int are_all_above(IntList* list_ptr, int threshold);
int does_contain_factors_of(IntList* list_ptr, int n);
int count_if_contains_factors_of(IntList* list1, IntList* list2);

int main(void) {
    IntList nums = create_list();
    insert_back(&nums, 2);
    insert_back(&nums, 4);
    insert_back(&nums, 3);
    printf("Original list: ");
    print_list(&nums);

    printf("Sum of squares: %d\n", sum_squares(&nums));

    add_to_all(&nums, 1);
    printf("After adding 1 to all elements: ");
    print_list(&nums);

    IntList below_threshold = create_list();
    int count = get_all_below(&nums, &below_threshold, 5);
    printf("Elements below threshold 5: ");
    print_list(&below_threshold);
    printf("Count: %d\n", count);

    printf("Are all elements above 3? %d\n", are_all_above(&nums, 3));

    printf("Does the list contain factors of 12? %d\n", does_contain_factors_of(&nums, 12));

    IntList list2 = create_list();
    insert_back(&list2, 12);
    insert_back(&list2, 5);
    insert_back(&list2, 4);
    printf("Second list: ");
    print_list(&list2);
    printf("Count of elements in second list that are factors of first list: %d\n", 
    count_if_contains_factors_of(&nums, &list2));

    free_list(&nums);
    free_list(&below_threshold);
    free_list(&list2);

    return 0;
}


/** 
 * @brief Dynamically allocates a new IntNode with the given values
 * @param num the integer value for the new node
 * @param next a pointer to the address of the node that follows this node in the list
 * @return the address of the newly created IntNode
 */
IntNode* create_node(int num, IntNode* next){
    IntNode* node = malloc(sizeof(IntNode));
    node->value = num;
    node->next = next;
    return node;
}

/**
 * @brief Creates and returns an instance of an IntList with its front set to NULL
 * @return the new IntList instance
 */
IntList create_list(){
    IntList list = {NULL};
    return list;
}

/**
 * @brief Creates a new IntNode and adds it to the front of the linked list
 * @param list_ptr a pointer to the list_ptr node in the list
 * @param num the integer value for the new node that is put into the list
 */
void insert_front(IntList* list_ptr, int num){
    list_ptr->front = create_node(num, list_ptr->front);
}

/**
 * @brief Creates a new IntNode and adds it to the back of the linked list
 * @param list_ptr a pointer to the list_ptr node in the list
 * @param num the integer value for the new node that is put into the list
 */
void insert_back(IntList* list_ptr, int num){
    if (list_ptr->front == NULL){
        list_ptr->front = create_node(num, NULL);
    } else {
        IntNode* cur = list_ptr->front;
        while (cur->next != NULL) {
            cur = cur->next;
        }
        cur->next = create_node(num, NULL);
    }
}

/**
 * @brief Prints out all values in the given linked list
 * @param list_ptr a pointer to the list_ptr node in the list
 */
void print_list(IntList* list_ptr) {
    IntNode* cur = list_ptr->front;
    
    printf("[");
    while (cur != NULL) {
        printf("%d", cur->value);
        if (cur->next != NULL) {
            printf(", ");
        }
        cur = cur->next;
    }
    printf("]\n");
}

/**
 * @brief Deallocates space for all nodes in a linked list
 * @param list_ptr a pointer to the list_ptr node in the list to deallocate
 */
void free_list(IntList* list_ptr) {
    IntNode* current_node = list_ptr->front;
    IntNode* tmp = current_node;

    while (current_node != NULL) {
        tmp = current_node;
        current_node = current_node->next;
        free(tmp);
    }
    list_ptr->front = NULL;
}

/**
 * @brief Determines whether n1 is a factor of n2
 * @param n1 the integer that may be a factor
 * @param n2 the integer n1 may be a factor of
 * @return the integer 1 if n1 is a factor of n2, or 0 if n1 is not a factor of n2
 * @pre n1 >= 0, n2 >= 0
 */
int is_factor(int n1, int n2) {
    return ((n1 == 0 && n2 == 0) || (n1 != 0 && n2 % n1 == 0));
}

/**
 * @brief Computes the sum of squares of all elements in a linked list.
 * @param list_ptr A pointer to the input linked list.
 * @return The sum of the squares of all integers in the list.
 */
int sum_squares(IntList* list_ptr) {
    IntNode* cur = list_ptr->front;
    int sum = 0;
    while (cur != NULL) {
        sum += cur->value * cur->value;
        cur = cur->next;
    }
    return sum;
}

/**
 * @brief Adds a given integer to every element in the linked list.
 * @param list_ptr A pointer to the linked list to update.
 * @param n The integer to add to each element.
 */
void add_to_all(IntList* list_ptr, int n) {
    IntNode* cur = list_ptr->front;
    while (cur != NULL) {
        cur->value += n;
        cur = cur->next;
    }
}

/**
 * @brief Copies all elements below a given threshold to a destination list.
 * @param src_list A pointer to the source linked list.
 * @param dest_list A pointer to the destination linked list.
 * @param threshold The threshold value.
 * @return The number of elements copied to the destination list.
 */
int get_all_below(IntList* src_list, IntList* dest_list, int threshold) {
    IntNode* cur = src_list->front;
    int count = 0;
    while (cur != NULL) {
        if (cur->value < threshold) {
            insert_back(dest_list, cur->value);
            count++;
        }
        cur = cur->next;
    }
    return count;
}

/**
 * @brief Checks if all elements in the list are above a given threshold.
 * @param list_ptr A pointer to the input linked list.
 * @param threshold The threshold value.
 * @return 1 if all elements are above the threshold, 0 otherwise.
 */
int are_all_above(IntList* list_ptr, int threshold) {
    IntNode* cur = list_ptr->front;
    while (cur != NULL) {
        if (cur->value <= threshold) {
            return 0;
        }
        cur = cur->next;
    }
    return 1;
}

/**
 * @brief Checks if the list contains any elements that are factors of a given number.
 * @param list_ptr A pointer to the input linked list.
 * @param n The number to check for factors.
 * @return 1 if any element in the list is a factor of n, 0 otherwise.
 */
int does_contain_factors_of(IntList* list_ptr, int n) {
    IntNode* cur = list_ptr->front;
    while (cur != NULL) {
        if (is_factor(n, cur->value)) {
            return 1;
        }
        cur = cur->next;
    }
    return 0;
}

/**
 * @brief Counts how many elements in the second list are factors of at least one element in the first list.
 * @param list1 A pointer to the first linked list.
 * @param list2 A pointer to the second linked list.
 * @return The count of elements in the second list that are factors of at least one element in the first list.
 */
int count_if_contains_factors_of(IntList* list1, IntList* list2) {
    IntNode* cur2 = list2->front;
    int count = 0;
    while (cur2 != NULL) {
        if (does_contain_factors_of(list1, cur2->value)) {
            count++;
        }
        cur2 = cur2->next;
    }
    return count;
}
