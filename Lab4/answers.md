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


