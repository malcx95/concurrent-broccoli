// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

const int GRID_SIZE = 1;
const int BLOCK_SIZE = 2048;
const int N = GRID_SIZE*BLOCK_SIZE;

__global__
void simple(float* in1, float* in2, float* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idx * N + idy;
    out[id] = in1[id] + in2[id];
}

int main()
{

    float* a = new float[N*N];
    float* b = new float[N*N];
    float* c = new float[N*N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i+j*N] = 10 + i;
            b[i+j*N] = (float)j / N;
        }
    }

    float* a_cuda;
    float* b_cuda;
    float* c_cuda;
    
    cudaEvent_t event_start;
    cudaEvent_t event_end;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_end);

    const int size = N*N*sizeof(float);

    cudaMalloc( (void**)&a_cuda, size );
    cudaMalloc( (void**)&b_cuda, size );
    cudaMalloc( (void**)&c_cuda, size );

    cudaEventRecord(event_start, 0);
    cudaMemcpy(a_cuda, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_cuda, c, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    simple<<<dimGrid, dimBlock>>>(a_cuda, b_cuda, c_cuda);

    cudaThreadSynchronize();
    cudaEventRecord(event_end, 0);
    cudaEventSynchronize(event_end);

    cudaMemcpy(c, c_cuda, size, cudaMemcpyDeviceToHost);

    float myVerySpecial_Time_not_t;
    cudaEventElapsedTime(&myVerySpecial_Time_not_t, event_start, event_end);

    printf("Time elapsed: %f\n", myVerySpecial_Time_not_t);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%0.2f ", c[i+j*N]);
        }
        printf("\n");
    }

    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);

    printf("done\n");
    return EXIT_SUCCESS;
}
