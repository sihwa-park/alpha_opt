#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.io import netcdf_file
from matplotlib.colors import LightSource

myfigsize = (6, 4.0)

print("usage: vmecPlot3D <woutXXX.nc>")
wout_filename = "../../../evals/deep-violet-5_eval000289/wout_tmp.nc"
f = netcdf_file(wout_filename, "r", mmap=False)
ns = f.variables["ns"][()]
nfp = f.variables["nfp"][()]
xn = f.variables["xn"][()]
xm = f.variables["xm"][()]
xn_nyq = f.variables["xn_nyq"][()]
xm_nyq = f.variables["xm_nyq"][()]
rmnc = f.variables["rmnc"][()]
zmns = f.variables["zmns"][()]
bmnc = f.variables["bmnc"][()]
Rmajor = f.variables["Rmajor_p"][()]

f.close()
nmodes = len(xn)

########################################################
# Make 3D surface plot
########################################################

fig = plt.figure(figsize=myfigsize)

ntheta = 90
nphi = 500
theta1D = np.linspace(0, 2 * np.pi, num=ntheta)
phi1D = np.linspace(0, 2 * np.pi, num=nphi)
phi2D, theta2D = np.meshgrid(phi1D, theta1D)
iradius = ns - 1

# Vectorized calculation of R, Z, and B
angles = xm[:, None, None] * theta2D[None, :, :] - xn[:, None, None] * phi2D[None, :, :]
R = np.sum(rmnc[iradius, :, None, None] * np.cos(angles), axis=0)
Z = np.sum(zmns[iradius, :, None, None] * np.sin(angles), axis=0)

R /= Rmajor
Z /= Rmajor

angles_nyq = (
    xm_nyq[:, None, None] * theta2D[None, :, :]
    - xn_nyq[:, None, None] * phi2D[None, :, :]
)
B = np.sum(bmnc[iradius, :, None, None] * np.cos(angles_nyq), axis=0)

X = R * np.cos(phi2D)
Y = R * np.sin(phi2D)
# Rescale to lie in [0,1]:
B_rescaled = (B - B.min()) / (B.max() - B.min())

# Zoom in:
factor = 2
# fig.subplots_adjust(bottom=-factor+0.05,top=1+factor)
fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

ax = plt.axes(projection="3d")
ax.set_axis_off()  # Hide the axes

# Add lighting for a shiny appearance
# ls = LightSource(azdeg=315, altdeg=45)
ls = LightSource(azdeg=45, altdeg=60)
# Calculate illumination
# rgb = ls.shade(B_rescaled, cmap=cm.autumn, vert_exag=0.3, blend_mode="soft")
rgb = ls.shade(B_rescaled, cmap=cm.viridis, vert_exag=0.3, blend_mode="soft")

p = ax.plot_surface(
    X,
    Y,
    Z,
    facecolors=rgb,
    rstride=1,
    cstride=1,
    antialiased=True,
    shade=True,
    # alpha=0.5,
)

max_range = (
    np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
)
max_range *= 0.8

mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# plt.figtext(0.5,0.999,'Color = |B|',horizontalalignment='center',verticalalignment='top',fontsize=10)
plt.savefig("vmec_3d_surface.pdf")

# plt.show()