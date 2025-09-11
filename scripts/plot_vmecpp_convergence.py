#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("force_residual_history.txt", skiprows=1)

max_residual = np.max(data[:, 1:], axis=1)

plt.figure(figsize=(10, 6))
plt.semilogy(max_residual)
plt.xlabel("Iteration")
plt.ylabel("Max Force Residual")
plt.grid()

plt.figtext(0.5, 0.995, os.getcwd(), ha="center", va="top", fontsize=7)
plt.figtext(0.5, 0.005, os.path.abspath(__file__), ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.show()