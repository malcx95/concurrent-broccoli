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

