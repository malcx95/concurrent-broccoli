// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define MAX_KERNEL_SIZE_X 10
#define MAX_KERNEL_SIZE_Y 10
#define BLOCK_SIZE 16
#define SHARED_STUFF_SIZE_X (BLOCK_SIZE+MAX_KERNEL_SIZE_X * 2)
#define SHARED_STUFF_SIZE_Y (BLOCK_SIZE+MAX_KERNEL_SIZE_Y * 2)
#define SHARED_STUFF_SIZE (SHARED_STUFF_SIZE_X*SHARED_STUFF_SIZE_Y)

__device__
const unsigned int GAUSS[5] = {1, 4, 6, 4, 1};

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 
    __shared__ unsigned char shared_stuff[SHARED_STUFF_SIZE_X][SHARED_STUFF_SIZE_Y][3];

    // map from blockIdx to pixel position
    int patch_start_x = blockIdx.x * blockDim.x;
    int patch_start_y = blockIdx.y * blockDim.y;
    int xxx = patch_start_x + threadIdx.x;
    int yyy = patch_start_y + threadIdx.y;
    size_t offset = threadIdx.x + threadIdx.y*blockDim.x;
    for (size_t i = offset; i < SHARED_STUFF_SIZE; i+=blockDim.x*blockDim.y) {
        size_t r_x_index = patch_start_x + (i % SHARED_STUFF_SIZE_X);
        size_t r_y_index = patch_start_y + (i / SHARED_STUFF_SIZE_X);
        size_t r_index = (r_x_index + r_y_index * imagesizex) * 3;

        shared_stuff[i % SHARED_STUFF_SIZE_X][i / SHARED_STUFF_SIZE_X][0] = image[r_index];
        shared_stuff[i % SHARED_STUFF_SIZE_X][i / SHARED_STUFF_SIZE_X][1] = image[r_index+1];
        shared_stuff[i % SHARED_STUFF_SIZE_X][i / SHARED_STUFF_SIZE_X][2] = image[r_index+2];
    }

    int x = threadIdx.x + kernelsizex;
    int y = threadIdx.y + kernelsizey;


    int divby = 16;

    // If inside image 
    if (xxx < imagesizex && yyy < imagesizey) {
        // Filter kernel (simple box filter)
        unsigned int sumx = 0;
        unsigned int sumy = 0;
        unsigned int sumz = 0;

        //sumx = shared_stuff[x][y][0];
        //sumy = shared_stuff[x][y][1];
        //sumz = shared_stuff[x][y][2];

        for(int dy=-kernelsizey;dy<=kernelsizey;dy++) {
            for(int dx=-kernelsizex;dx<=kernelsizex;dx++) {
                // Use max and min to avoid branching!
                int xx = min(max(x+dx, 0), SHARED_STUFF_SIZE_X-1);
                int yy = min(max(y+dy, 0), SHARED_STUFF_SIZE_Y-1);

                int gauss_index = max(abs(dx), abs(dy));

                sumx += shared_stuff[xx][yy][0]*GAUSS[2 + gauss_index];
                sumy += shared_stuff[xx][yy][1]*GAUSS[2 + gauss_index];
                sumz += shared_stuff[xx][yy][2]*GAUSS[2 + gauss_index];
            }
        }
        out[(yyy*imagesizex+xxx)*3+0] = sumx/divby;
        out[(yyy*imagesizex+xxx)*3+1] = sumy/divby;
        out[(yyy*imagesizex+xxx)*3+2] = sumz/divby;
    }
}

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
    if (kernelsizex > MAX_KERNEL_SIZE_X || kernelsizey > MAX_KERNEL_SIZE_Y) { printf("Kernel size out of bounds!\n");
        return;
    }

    pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
    cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
    cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
    cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
    dim3 grid(imagesizex/BLOCK_SIZE,imagesizey/BLOCK_SIZE);

    int start = GetMicroseconds();
    filter<<<grid,dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(dev_input, dev_bitmap, imagesizex, imagesizey, 2, 0); // Awful load balance
    filter<<<grid,dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(dev_bitmap, dev_input, imagesizex, imagesizey, 0, 2); // Awful load balance
    cudaThreadSynchronize();
    int end = GetMicroseconds();

    printf("Time: %i us\n", end-start);

    // Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy( pixels, dev_input, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
    cudaFree( dev_bitmap );
    cudaFree( dev_input );
}

// Display images
void draw()
{
    // Dump the whole picture onto the screen.
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );

    // Not wide - probably square. Original left, result right.
    if (imagesizey >= imagesizex) {
        glRasterPos2f(-1, -1);
        glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
        glRasterPos2i(0, -1);
        glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
    }
    else { // Wide image! Original on top, result below.
        glRasterPos2f(-1, -1);
        glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
        glRasterPos2i(-1, 0);
        glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
    }
    glFlush();
}

// Main program, inits
int main( int argc, char** argv) {
#ifdef __APPLE__
    *(NULL) = 0xBAD;
#endif
    glutInit(&argc, argv);
    glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

    if (argc > 1) {
        image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
    }
    else {
        image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);
    }

    if (imagesizey >= imagesizex) {
        glutInitWindowSize( imagesizex*2, imagesizey );
    }
    else {
        glutInitWindowSize( imagesizex, imagesizey*2 );
    }
    glutCreateWindow("Lab 5");
    glutDisplayFunc(draw);

    ResetMilli();

    computeImages(5, 5);

    // You can save the result to a file like this:
    // writeppm("out.ppm", imagesizey, imagesizex, pixels);

    glutMainLoop();
    return 0;
}
