#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Slowing-down time:
tau = 0.1

# Set the threshold
maxloss = 0.02

# Load the particle data
particle_data = pd.read_csv("particle_data.csv")
# particle_data = pd.read_csv("particle_data_with_maxloss.csv")
# particle_data = pd.read_csv("particle_data_without_maxloss.csv")

# Get the last_time column (time at which each particle is lost)
last_times = particle_data["last_time"].values

# Sort the loss times
sorted_times = np.sort(last_times)

# Calculate the cumulative fraction of lost particles
n_particles = len(sorted_times)
cumulative_particle_loss_fraction = np.arange(1, n_particles + 1) / n_particles

# Calculate the cumulative energy loss fraction
# Each particle lost at time t contributes energy loss of exp(-t/tau)
# normalized by n_particles
energy_loss_per_particle = np.exp(-sorted_times / tau)
cumulative_energy_loss_fraction = np.cumsum(energy_loss_per_particle) / n_particles

# Find the time at which the ENERGY loss fraction exceeds maxloss
# We need to find the first time where cumulative_energy_loss_fraction >= maxloss
idx = np.searchsorted(cumulative_energy_loss_fraction, maxloss, side='left')
if idx < n_particles:
    time_at_maxloss = sorted_times[idx]
    print(f"Energy loss fraction exceeds {maxloss} at time: {time_at_maxloss:.6e} s")
else:
    time_at_maxloss = None
    print(f"Energy loss fraction never exceeds {maxloss}")

# Create the plot
plt.figure(figsize=(10, 8))
plt.plot(sorted_times, cumulative_particle_loss_fraction, 'b-', linewidth=2, label='Particle loss fraction')
plt.plot(sorted_times, cumulative_energy_loss_fraction, 'r-', linewidth=2, label=f'Energy loss fraction (τ={tau})')

# Add horizontal line at maxloss
plt.axhline(y=maxloss, color='k', linestyle='--', linewidth=1.5, label=f'Loss threshold ({maxloss})')

# Add vertical line at time_at_maxloss
if time_at_maxloss is not None:
    plt.axvline(x=time_at_maxloss, color='g', linestyle='--', linewidth=1.5, 
                label=f'Time at threshold ({time_at_maxloss:.3e} s)')

# Set log scale for x-axis
plt.xscale('log')
plt.yscale('log')

# Labels and formatting
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Cumulative Loss Fraction', fontsize=12)
plt.title('Accumulated Particle and Energy Loss vs Time: ' + os.path.basename(os.getcwd()), fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=10)
# plt.ylim([0, 1.05])

plt.figtext(0.5, 0.005, os.path.abspath(__file__), ha="center", va="bottom", fontsize=6)
# Save and show the plot
plt.tight_layout()
# plt.savefig('accumulated_loss_plot.png', dpi=300)
# print("Plot saved as 'accumulated_loss_plot.png'")
plt.show()
