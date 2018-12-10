import subprocess
import numpy
import matplotlib.pyplot as plt

skeleton = """
#define GRID_SIZE 1
#define BLOCK_SIZE {}
const int N = GRID_SIZE*BLOCK_SIZE;
"""

gpu_results = []
cpu_results = []

xrange = [2**i for i in range(1, 10)]

for res in xrange:
    with open("config.h", "w") as f:
        f.write(skeleton.format(res));
    gpu_results.append(float(subprocess.check_output(["make", "ring_gpun"]).decode("UTF-8")))
    cpu_results.append(float(subprocess.check_output(["make", "ring_cpun"]).decode("UTF-8")))

plt.plot(xrange, gpu_results, xrange, cpu_results)
plt.legend(["GPU", "CPU"])
plt.show()
# with open("aklagaren.csv", 'w') as f:
#     for gpu, cpu in results:
#         f.write(str(gpu) + ',' + str(cpu) + '\n')

