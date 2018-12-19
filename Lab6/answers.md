# 1

clCreateBuffer
clSetKernelArg


# 2

clEnqueueNDRangeKernel

# 3

get_global_id

# 4

1.3 ms on the gpu compared to 0.007 ms on the cpu! MUCH SLOWER! SAD!


(such wow, much slow)

# 5

Around 2^20 elements (a million).

# 6

You could have multiple levels per kernel run to avoid overhead.

# 7

Two outputs, since it swaps two pairs of elements.

# 8

512 since the if-statement on line 66 switches to global work size there

# 9

Maybe use global id?

# 10

`~100` times faster for large datasets

The GPU is faster for 4096 elements but slower for 2048

A parallel CPU would probably be <number of cores> faster than the single threaded
CPU version for relatively large datasets
