#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load the history data
filename = glob.glob("*.npy")[0]  # Get the first .npy file in the current directory
history = np.load(filename)
print("Loaded history from:", filename)

x = history["x"]
print("x.shape:", x.shape)

f = history["f"]
f[np.logical_not(history["sim_ended"])] = np.inf  # Ignore unfinished sims
print("Minimum objective function value (f):", min(f))
print("Simulation with that value is at index:", np.argmin(f))

plt.figure(figsize=(14.5, 8))
n_rows = 1
n_cols = 1

plt.subplot(n_rows, n_cols, 1)
plt.semilogy(history["f"], '.-')
plt.xlabel("iteration")
plt.ylabel("objective function")

plt.tight_layout()
plt.show()
