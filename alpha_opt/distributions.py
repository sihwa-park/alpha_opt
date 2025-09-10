import numpy as np
import matplotlib.pyplot as plt
import math
import time
from math import sqrt


from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )

# Compute the pdf of birth rate in s
def s_density(s):
	return ((1-s**5)**2)*((1-s)**(-2/3))*np.exp(-19.94*(12*(1-s))**(-1/3))

# Rejection sample s
def sample_s():
	bound = 3e-4
	x = np.random.uniform()
	y = bound * np.random.uniform()

	while s_density(x) < y:
		assert s_density(x) <= bound
		x = np.random.uniform()
		y = bound * np.random.uniform()
	return x

# Sample theta, zeta for a given s via rejection sampling
def sample_tz(s, J_max, field):
	J = rand_J = 0
	while rand_J  >= J:
		theta = np.random.uniform(low=0, high=2*math.pi, size=1)
		zeta = np.random.uniform(low=0, high=2*math.pi, size=1)
		rand_J = np.random.uniform(low=0, high=J_max, size=1)

		loc = np.array([s, theta[0], zeta[0]]).reshape(1,3)
		field.set_points(loc)

		G = field.G()
		iota = field.iota()
		I = field.I()
		modB = field.modB()
		J = (G + iota*I)/(modB**2)
		J = J[0][0]
		assert J <= J_max
	return theta[0], zeta[0]

# Sample s,t,z 
def sample_stz(field, J_max):
	s = sample_s()
	theta, zeta = sample_tz(s, J_max, field)
	return np.array([s, theta, zeta])

# Initialize vpar
vpar = np.sqrt(2*ENERGY/MASS)

# set seed for consistency
np.random.seed(8)

# trace particles
nparticles = 25000

stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
vpar_inits = vpar * np.random.uniform(low=-1, high=1, size=nparticles)

