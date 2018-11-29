import matplotlib.pyplot as plt
import numpy as np

shit_1_0 = []
with open("shit_1_0.csv") as f:
    shit_1_0 = [float(l) for l in f.readlines()]

shit_1_1 = []
with open("shit_1_1.csv") as f:
    shit_1_1 = [float(l) for l in f.readlines()]

shit_2_0 = []
with open("shit_2_0.csv") as f:
    shit_2_0 = [float(l) for l in f.readlines()]

shit_2_1 = []
with open("shit_2_1.csv") as f:
    shit_2_1 = [float(l) for l in f.readlines()]

plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, 17), shit_1_0)
plt.title("Pop with locks")
plt.subplot(2, 2, 2)
plt.plot(np.arange(1, 17), shit_1_1)
plt.title("Pop with CAS")

plt.subplot(2, 2, 3)
plt.plot(np.arange(1, 17), shit_2_0)
plt.title("Push with locks")
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, 17), shit_2_1)
plt.title("Push with CAS")

plt.show()

