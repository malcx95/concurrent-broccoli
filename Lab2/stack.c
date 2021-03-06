/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

stack_t* mini_stack_init() {
    stack_t* curr = NULL;
    for (size_t i = 0; i < MAX_PUSH_POP + 10; ++i) {
        stack_t* prev = curr;
        curr = malloc(sizeof(stack_t));
        curr->next = prev;
    }
    return curr;
}

void mini_stack_push(stack_t** head, stack_t* elem) {
    (*head)->next = elem;
    (*head) = elem;
}

stack_t* mini_stack_pop(stack_t** head) {
    stack_t* popped = *head;
    (*head) = (*head)->next;
    popped->entry = popped->next = NULL;
    popped->length = 0;
    return popped;
}

int
stack_check(stack_t* stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(stack != NULL);

    int len = stack->length;
    if (len == 0) {
        assert(stack->next == NULL);
        assert(stack->entry == NULL);
        return 1;
    }
    stack_t* next = stack;
    while (next != NULL) {
        len--;
        next = next->next;
    }

    assert(len == 0);

#endif
	// The stack is always fine
	return 1;
}

stack_t* stack_init() {
    stack_t* head = malloc(sizeof(stack_t));
    head->next = NULL;
    head->length = 0;
    head->entry = NULL;
    return head;
}

void stack_obliterate(stack_t* stack) {
    while(stack != NULL) {
        stack_t* next = stack->next;
#if NON_BLOCKING == 0
        // free(stack);
#endif
        stack = next;
    }
}

int stack_push(stack_t** head_ptr, void* elem SYNC_PARAM)
{
#if NON_BLOCKING == 0
    stack_t* head = *head_ptr;
    stack_t* new = malloc(sizeof(stack_t));
    
    pthread_mutex_lock(lock);
    // empty stack
    if (head->length == 0) {
        head->entry = elem;
        head->length = 1;
        stack_check(head);
        pthread_mutex_unlock(lock);
        return 0;
    }


    new->next = head->next;
    new->entry = head->entry;
    new->length = head->length;
    head->next = new;
    head->entry = elem;
    head->length += 1;

    stack_check(head);
    pthread_mutex_unlock(lock);

#elif NON_BLOCKING == 1
    stack_t* new = mini_stack_pop(&mini_stack);
    stack_t* old;
    do {
        old = *head_ptr;
        if(old->length != 0) {
            new->next = old;
            new->length = old->length + 1;
        }
        else {
            new->next = NULL;
            new->length = 1;
        }
        new->entry = elem;
    } while (cas((size_t*) head_ptr, (size_t) old, (size_t) new) != (size_t) old);

    stack_check(*head_ptr);
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses

  return 0;
}

void* stack_pop(stack_t** head_ptr SYNC_PARAM)
{
#if NON_BLOCKING == 0
    stack_t* head = *head_ptr;
    pthread_mutex_lock(lock);
    void* entry = head->entry;

    // only one element
    if (head->length == 1) {
        head->entry = NULL;
        head->length = 0;
        entry = NULL;
    } else if (head->length == 0) {
        entry = NULL;
    }
    else {
        stack_t* old_next = head->next;
        head->entry = head->next->entry;
        head->next = head->next->next;
        head->length -= 1;

        free(old_next);
    }
    
    stack_check(head);
    pthread_mutex_unlock(lock);
#elif NON_BLOCKING == 1
    // Implement a harware CAS-based stack
    if ((*head_ptr)->entry == NULL) {
        return NULL;
    }
    
    stack_t* new = mini_stack_pop(&mini_stack);
    void* entry;
    stack_t* old;
    stack_t* old_next;
    do {
        old = *head_ptr;
        old_next = old->next;
        entry = old->entry;

        if(old->next == NULL) {
            new->entry = NULL;
            new->length = 0;
        }
        else {
            new->next = old->next->next;
            new->entry = old->next->entry;
            new->length = old->length - 1;
        }
    } while(cas((size_t*) head_ptr, (size_t) old, (size_t) new) != (size_t) old);

    mini_stack_push(&mini_stack, old);
    mini_stack_push(&mini_stack, old_next);

    stack_check(*head_ptr);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

    return entry;
}


#if NON_BLOCKING == 1 || NON_BLOCKING == 2
void* aba_idiot_1(void* _arg) {
    idiot_data_t* arg = _arg;
    stack_t** head_ptr = arg->head_ptr;
    pthread_mutex_t* lock1 = arg->lock1;
    pthread_mutex_t* lock2 = arg->lock2;
    stack_t* mini_stack = mini_stack_init();

    stack_t* new = mini_stack_pop(&mini_stack);
    stack_t* old;
    printf("\nThread 1 waitig for thread 2\n");
    pthread_mutex_unlock(lock2);
    pthread_mutex_lock(lock1);


    do {
        old = *head_ptr;
        if(old->length != 0) {
            new->next = old;
            new->length = old->length + 1;
        }
        else {
            new->next = NULL;
            new->length = 1;
        }
        new->entry = (void*)8;
    } while (cas((size_t*) head_ptr, (size_t) old, (size_t) new) != (size_t) old);

    printf("Thread 1 is done\n");
    return NULL;
}

void* aba_idiot_2(void* _arg) {
    idiot_data_t* arg = _arg;
    stack_t* mini_stack = mini_stack_init();

    stack_t** head_ptr = arg->head_ptr;
    pthread_mutex_t* lock1 = arg->lock1;
    pthread_mutex_t* lock2 = arg->lock2;

    printf("T\nhread 2 set up, waiting for 1 to unlock\n");
    pthread_mutex_lock(lock2);
    printf("Thread 2 popping\n");
    stack_pop(head_ptr, mini_stack);
    printf("Thread 2 pushing\n");
    stack_push(head_ptr, (void*) 7, mini_stack);
    printf("Thread 2 done, unlocking lock 1\n");
    pthread_mutex_unlock(lock1);
    return NULL;
}


#endif

