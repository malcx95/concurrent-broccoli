import matplotlib.pyplot as plt
import numpy as np

withlb = []
with open("shit.csv") as f:
    withlb = [float(l) for l in f.readlines()]

withoutlb = []
with open("shit2.csv") as f:
    withoutlb = [float(l) for l in f.readlines()]

plt.figure(1)
plt.plot(np.arange(1, 17), withlb, np.arange(1, 17), withoutlb)
plt.legend(["Time with load balancing", "Time witout load balancing"])

plt.show()

