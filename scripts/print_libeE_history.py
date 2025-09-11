#!/usr/bin/env python

import glob
import numpy as np

# history = np.load("history.npy")
# history = np.load("output_history_length=20_evals=20_workers=4.npy")
#history = np.load("uniform_sampling_then_persistent_localopt_runs_history_length=1003_evals=1000_workers=4.npy")
# history = np.load("libE_history_at_abort_1000.npy")

filename = glob.glob("*.npy")[0]  # Get the first .npy file in the current directory
history = np.load(filename)

print([i for i in history.dtype.fields])  # (optional) to visualize our history array
print(history)
print("type(history):", type(history))
print("history.shape:", history.shape)
print("history.dtype:", history.dtype)

