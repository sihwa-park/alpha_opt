#!/usr/bin/env python

import os
import numpy as np
import booz_xform as bx
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

wout_filename = "../../../evals/lively-sweep-3_eval000173/wout_tmp.nc"

# Run booz_xform
rho_values = [0.25, 0.5, 0.75, 1.0]
s_targets = [rho**2 for rho in rho_values]

b = bx.Booz_xform()
b.read_wout(wout_filename)

# Find half-grid surface indices closest to the target s = rho^2 values
ns_in = b.ns_in
s_half = np.array([(j + 0.5) / ns_in for j in range(ns_in)])
compute_surfs = [int(np.argmin(np.abs(s_half - s_t))) for s_t in s_targets]
print("compute_surfs:", compute_surfs)

b.mboz = 24
b.nboz = 24
b.compute_surfs = compute_surfs
b.run()

boozmn_filename = "boozmn_ns51.nc"
b.write_boozmn(boozmn_filename)

# print("Include any arguments to save a PDF")
# save_fig = (len(sys.argv) > 1)
save_fig = True

print('Reading file', boozmn_filename)
with netcdf_file(boozmn_filename, mmap=False) as f:
    nfp = int(f.variables['nfp_b'][()])
    ns_b = int(f.variables['ns_b'][()])
    jlist = f.variables['jlist'][()].copy()
    iota_b = f.variables['iota_b'][()].copy()
    xm = f.variables['ixm_b'][()].copy()
    xn = f.variables['ixn_b'][()].copy()
    bmnc = f.variables['bmnc_b'][()].copy()

print('bmnc.shape:', bmnc.shape)
print('jlist:', jlist)
if bmnc.shape[0] != 4 or len(jlist) != 4:
    raise ValueError(f"Expected exactly 4 saved Boozer surfaces, found {bmnc.shape[0]}")

# booz_xform writes jlist = compute_surfs + 2, with compute_surfs on the VMEC half grid.
saved_ss = (jlist - 1.5) / (ns_b - 1)
print('saved s values:', saved_ss)

fig = plt.figure(figsize=(4.7, 4.0))
nrows = 2
ncols = 2
ncontours = 20

ntheta = 70
nphi = 80
theta1d = np.linspace(0, 2 * np.pi, ntheta)
# phi1d = np.linspace(0, 2 * np.pi / nfp, nphi)
phi1d = np.linspace(0, 2 * np.pi / 2, nphi)
phi2d, theta2d = np.meshgrid(phi1d, theta1d)

axs = []

rounded_rhos = [0.25, 0.5, 0.75, 1]
for js in range(4):
    rounded_rho = rounded_rhos[js]
    desired_s = saved_ss[js]
    iota = iota_b[jlist[js] - 2]
    print('For saved surface', js, ' jlist=', jlist[js], ' s=', desired_s, ' iota=', iota)
    modB = np.zeros((ntheta, nphi))
    for jmn in range(len(xm)):
        modB += bmnc[js, jmn] * np.cos(xm[jmn] * theta2d - xn[jmn] * phi2d)

    """
    Bmax = np.max(modB)
    Bmin = np.min(modB)
    modB = (modB - Bmin) / (Bmax - Bmin)
    """
    #modB /= B00

    ax = plt.subplot(nrows, ncols, js + 1)
    contourplot = ax.contour(phi1d, theta1d, modB, ncontours, linewidths=1.0)
    phi_line = np.array([phi1d[0], phi1d[-1]])
    theta_line = iota * phi_line
    ax.plot(phi_line, theta_line, 'k-', linewidth=1.5)
    cbar = plt.colorbar(contourplot, ax=ax)
    lo = int(np.ceil(contourplot.levels.min()))
    hi = int(np.floor(contourplot.levels.max()))
    if lo <= hi:
        ticks = np.arange(lo, hi + 1, 1)
        if len(ticks) < 4:
            lo2 = int(np.ceil(2 * contourplot.levels.min()))
            hi2 = int(np.floor(2 * contourplot.levels.max()))
            if lo2 <= hi2:
                ticks = np.arange(lo2, hi2 + 1, 1) / 2
        cbar.set_ticks(ticks)
    axs.append(ax)
    ax.tick_params(direction='in', length=0)
    color = (0.9, 0.9, 0.9)
    plt.title(rf"|B| [T],  $\rho$={rounded_rho}", fontsize=10)
    # plt.text(0.025, 6.15, rf"|B| @ $\rho$={rounded_rho}", ha='left', va='top',
    #          fontsize=9,
    #          bbox=dict(boxstyle="round,pad=0.1",
    #                    ec=color,
    #                    fc=color,
    #                    ))
    plt.yticks([0, 2 * np.pi], ['0', r'$2\pi$'])
    plt.ylabel(r'$\theta_B$', labelpad=-12)
    # plt.xticks([0, np.pi / 2], ['0', r'$\pi/2$  '])
    plt.xticks([0, np.pi], ['0', r'$\pi$  '])
    plt.xlabel(r'$\varphi_B$', labelpad=-7)

#plt.figtext(0.5, 0.995, '|B| on flux surfaces of the quasi-helically symmetric field', ha='center', va='top', fontsize=11)
#plt.subplots_adjust(left=0.054, bottom=0.07, right=0.94, top=0.93, wspace=0.39, hspace=0.16)
plt.subplots_adjust(left=0.054, bottom=0.07, right=0.93, top=0.94, wspace=0.3, hspace=0.307)

if save_fig:
    output_filename = os.path.basename(boozmn_filename)[len("boozmn_"):-len(".nc")] + "_B_contours.pdf"
    print('Saving PDF to', output_filename)
    plt.savefig(output_filename)
# else:
# plt.show()


