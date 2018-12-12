# 1
16 cores, ? SMs
# 2
It's identical here, but may not always be due to different implementations on CPU
and GPU.

# 3

```C
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
int id = idx * N + idy;
```

# 4

data = 0

# 5

Around 70

# 6

A large block size, like 1024.

# 7

Hard to tell, the fastest usually runs in ~2 ms but what configuration achieves that varies between runs

# 8

No discernable difference.

# 9

We replaced the for loop with threads on the GPU, by computing the position to draw and then
drawing that position. Also, we used cudaMemCpy to copy the result from the GPU to the host.

We also fixed the size of the window to make the computations simpler.

# 10

`16*16` threads per block and `DIM/16*DIM/16` blocks.

# 11

`__device__`

# 12

(With double)
The CPU implementation took about 1 second per frame, while the GPU version took only around 5
milliseconds, which is about 200 times faster.

# 13

## GPU

With float it took around 2 milliseconds, while it took around 5 milliseconds with double.

## CPU

It made basically no difference, it took around 1 second on both.

# 14

It doesn't seem to be an issue. Since we have one thread per pixel, it's fine if some threads
finish before others.


