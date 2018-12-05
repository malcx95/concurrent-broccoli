// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include "milli.h"
#include <stdlib.h>

void add_matrix(float *a, float *b, float *c, int N)
{
    int index;
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            index = i + j*N;
            c[index] = a[index] + b[index];
        }
}

int main()
{
    const int N = 1024*16;

    float* a = malloc(sizeof(float) * N * N);
    float* b = malloc(sizeof(float) * N * N);
    float* c = malloc(sizeof(float) * N * N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            a[i+j*N] = 10 + i;
            b[i+j*N] = (float)j / N;
        }

    int time__________________________________________________________________________________________waitforit___________________________________________________________________________________________________________almostthere__________________start = GetMicroseconds();
    
    add_matrix(a, b, c, N);

    int timeend = GetMicroseconds();
    printf("Time elapsed on CPU: %f\n", (timeend - time__________________________________________________________________________________________waitforit___________________________________________________________________________________________________________almostthere__________________start) / 1000.f);
    
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%0.2f ", c[i+j*N]);
    //     }
    //     printf("\n");
    // }
}
