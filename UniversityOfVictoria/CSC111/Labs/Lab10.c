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

// Functions that you will implement
int get_min(IntList* list_ptr);
int all_odd(IntList* list_ptr);
int contains_odds(IntList* list_ptr);
void multiply_by(IntList* list_ptr, int multiplier);
void clamp(IntList* list_ptr, int min_value, int max_value);
void print_alternating(IntList* list_ptr);

int main(void) {
    int result;
    printf("\nPART 1 TESTS:\n");
    // PART 1: Memory Trace
    // TODO: Draw a memory diagram showing the state of the linked list
    //       after each function call. Show your TA the final state 
    //       of each list before the call to free().
    // You can check your answers by calling the provided print_list function.
    IntList list1 = create_list();
    insert_front(&list1, 5);
    insert_front(&list1, 6);
    insert_front(&list1, 7);
    insert_front(&list1, 8);
    // Show the TA the linked list (connected nodes) at this point.
    free_list(&list1);
    
    insert_back(&list1, 5);    
    insert_back(&list1, 6);
    insert_back(&list1, 7);
    insert_back(&list1, 8);
    // Show the TA the linked list (connected nodes) at this point.
    free_list(&list1);

    
    // Creating lists for Part 2, 3, and 4 tests:
    insert_back(&list1, 2);
    
    IntList list2 = create_list();
    insert_back(&list2, 9);
    insert_back(&list2, 1);
    insert_back(&list2, 3);
    insert_back(&list2, 5);

    IntList list3 = create_list();
    insert_back(&list3, 5);
    insert_back(&list3, 11);
    insert_back(&list3, 9);
    insert_back(&list3, 33);
    insert_back(&list3, 4);

    IntList empty_list = create_list();

    
    // Part 2 Tests:
    printf("\nPART 2 TESTS:\n");
    print_list(&empty_list);
    print_list(&list1);
    print_list(&list2);
    print_list(&list3);

    result = get_min(&list1);
    printf("min in list1: %d\n", result);
    result = get_min(&list2);
    printf("min in list2: %d\n", result);
    result = get_min(&list3);
    printf("min in list3: %d\n", result);

    result = all_odd(&list1);
    printf("are all_odd in list1: %d\n", result);
    result = all_odd(&list2);
    printf("are all_odd in list2: %d\n", result);
    result = all_odd(&list3);
    printf("are all_odd in list3: %d\n", result);
    result = all_odd(&empty_list);
    printf("are all_odd in empty_list: %d\n", result);

    result = contains_odds(&list1);
    printf("list1 contains odds: %d\n", result);
    result = contains_odds(&list2);
    printf("list2 contains odds: %d\n", result);
    result = contains_odds(&list3);
    printf("list3 contains odds: %d\n", result);
    result = contains_odds(&empty_list);
    printf("empty_list contains odds: %d\n", result);
        
    // Part 3 Tests:
    printf("\n\nPART 3 TESTS:\n");
    printf("list2 before multiply_by 3:\n");
    print_list(&list2);
    multiply_by(&list2, 3);
    printf("list2 after:\n");
    print_list(&list2);

    printf("\nlist3 before multiply_by 19:\n");
    print_list(&list3);
    multiply_by(&list3, 19);
    printf("list2 after:\n");
    print_list(&list3);

    printf("\nlist2 before clamp with 5 and 16:\n");
    print_list(&list2);
    clamp(&list2, 5, 16);
    printf("list2 after:\n");
    print_list(&list2);

    printf("\nlist3 before clamp with 100 and 250:\n");
    print_list(&list3);
    clamp(&list3, 100, 250);
    printf("list3 after:\n");
    print_list(&list3);

    // Part 4 Tests:
    printf("\n\nPART 4 TESTS:\n");
    printf("print alternating list1:\n");
    print_alternating(&list1);

    printf("\nprint alternating list2:\n");
    print_alternating(&list2);
    
    printf("\nprint alternating list3:\n");
    print_alternating(&list3);

    printf("\nprint alternating empty list:\n");
    print_alternating(&empty_list);
    
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
 * @param list_ptr a pointer to the linked list instance to insert into
 * @param num the integer value for the new node that is put into the list
 */
void insert_front(IntList* list_ptr, int num){
    list_ptr->front = create_node(num, list_ptr->front);
}

/**
 * @brief Creates a new IntNode and adds it to the back of the linked list
 * @param list_ptr a pointer to the linked list instance to insert into
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
 * @param list_ptr a pointer to the linked list instance to output
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
 * @param list_ptr a pointer to the linked list to deallocate
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
 * @brief Gets the minimum value found in the list
 * @param list_ptr a pointer to the linked list instance to examine
 * @return the minimum value
 * @pre the list contains at least one element (list_ptr != NULL)
 */
int get_min(IntList* list_ptr) {
    IntNode* cur = list_ptr->front;
    if (cur == NULL) {
        printf("Error: List is empty.\n");
        exit(EXIT_FAILURE);
    }
    int min = cur->value;
    while (cur != NULL) {
        if (cur->value < min) {
            min = cur->value;
        }
        cur = cur->next;
    }
    return min;
}

/**
 * @brief Determines if the list contains only odd values
 * @param list_ptr a pointer to the linked list instance to examine
 * @return the integer 1 if no evens are found, 0 otherwise
 */
int all_odd(IntList* list_ptr) {
    IntNode* cur = list_ptr->front;
    while (cur != NULL) {
        if (cur->value % 2 != 0) {
            return 1;
        }
        cur = cur->next;
    }
    return 0;    
}

/**
 * @brief Determines if the list contains any odd values
 * @param list_ptr a pointer to the linked list instance to examine
 * @return the integer 1 if an odd is found, 0 otherwise
 */
int contains_odds(IntList* list_ptr) {
    IntNode* cur = list_ptr->front;
    while (cur != NULL) {
        if (cur->value % 2 == 0) {
            return 0;
        }
        cur = cur->next;
    }
    return 1;
}

/**
 * @brief multiplies all list elements by the multiplier
 * @param list_ptr a pointer to the linked list instance to modify
 * @param multiplier the value to multiply all values by
 */
void multiply_by(IntList* list_ptr, int multiplier) {
    IntNode* cur = list_ptr->front;
    while (cur != NULL) {
        cur->value *= multiplier;
        cur = cur->next;
    } 
}

/**
 * @brief clamps the values of all elements in the list
 * @param list_ptr a pointer to the linked list instance to clamp
 * @param min_value ensures all values are this value or higher
 * @param max_value ensures all values are this value or lower
 */
void clamp(IntList* list_ptr, int min_value, int max_value) {
    IntNode* cur = list_ptr->front;
    while (cur != NULL) {
        if (cur->value < min_value) {
            cur->value = min_value;
        } else if (cur->value > max_value) {
            cur->value = max_value;
        }
        cur = cur->next;
    }   
}

/**
 * @brief Prints out every second value in the list beginning with the second element
 * @param list_ptr a pointer to the linked list instance to output
 */
void print_alternating(IntList* list_ptr) {
    IntNode* cur = list_ptr->front;
    int index = 0; // Start index at 0
    printf("[");
    while (cur != NULL) {
        if (index % 2 != 0) { // Check if the index is odd
            printf("%d", cur->value);
            if (cur->next != NULL && cur->next->next != NULL) {
                printf(", ");
            }
        }
        cur = cur->next;
        index++;
    }
    printf("]\n");
}
