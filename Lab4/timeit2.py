import subprocess
import numpy as np
import matplotlib.pyplot as plt

skeleton = """
#define GRID_SIZE {}
#define BLOCK_SIZE {}
const int N = GRID_SIZE*BLOCK_SIZE;
"""

gpu_results = []
cpu_results = []

xrange = [2**i for i in range(1, 10)]
xrange_rev = reversed(xrange)

for block_size, grid_size in zip(xrange, xrange_rev):
    with open("config.h", "w") as f:
        f.write(skeleton.format(grid_size, block_size));
    gpu_results.append(float(subprocess.check_output(["make", "ring_gpun"]).decode("UTF-8")))

plt.plot(xrange, gpu_results)
plt.legend(["GPU"])
plt.show()

