/*
 * stack.h
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
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

#if NON_BLOCKING == 0
#define SYNC_PARAM , pthread_mutex_t* lock
#else
#define SYNC_PARAM , stack_t** mini_stack 
#endif

struct stack
{
    struct stack* next;
    void* entry;
    int length;
};

typedef struct stack stack_t;
stack_t* mini_stack_init();
void mini_stack_push(stack_t** head, stack_t* elem);
stack_t* mini_stack_pop(stack_t** head);


stack_t* stack_init();
void stack_obliterate(stack_t* stack);

int stack_push(stack_t** head_ptr, void* elem SYNC_PARAM);
void* stack_pop(stack_t** head_ptr SYNC_PARAM);

typedef struct {
    stack_t** head_ptr;
    pthread_mutex_t* lock1;
    pthread_mutex_t* lock2;
} idiot_data_t;

void aba_idiot_1(void* arg);
void aba_idiot_2(void* arg);

/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_t* stack);
#endif /* STACK_H */
