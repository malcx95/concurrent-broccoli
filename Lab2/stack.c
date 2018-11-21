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
        free(stack);
        stack = next;
    }
}

int stack_push(stack_t* head, void* elem LOCK_PARAM)
{
#if NON_BLOCKING == 0
    stack_t* new = malloc(sizeof(stack_t));
    
    // empty stack
    if (head->entry == NULL) {
        pthread_mutex_lock(lock);
        head->entry = elem;
        pthread_mutex_unlock(lock);
        head->length = 1;
        stack_check(head);
        return 0;
    }

    pthread_mutex_lock(lock);

    new->next = head->next;
    new->entry = head->entry;
    new->length = head->length;
    head->next = new;
    head->entry = elem;
    head->length += 1;

    pthread_mutex_unlock(lock);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check(head);

  return 0;
}

void* stack_pop(stack_t* head LOCK_PARAM)
{
#if NON_BLOCKING == 0
    pthread_mutex_lock(lock);
    void* entry = head->entry;

    // only one element
    if (head->next == NULL) {
        head->entry = NULL;
        head->length = 0;
        return entry;
    } else if (head->entry == NULL) {
        return NULL;
    }

    stack_t* old_next = head->next;
    head->next = head->next->next;
    head->entry = head->next->entry;
    head->length -= 1;

    free(old_next);
    
    pthread_mutex_unlock(lock);
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif
    stack_check(head);

    return 0;
}

