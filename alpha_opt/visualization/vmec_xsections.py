#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf

myfigsize=(3,3)

print()
print("Include any additional arguments to save a PDF")


save_fig = True

filename = "../../../evals/deep-violet-5_eval000289/wout_tmp.nc"
f = netcdf.netcdf_file(filename, 'r', mmap=False)
phi = f.variables['phi'][()]
iotaf = f.variables['iotaf'][()]
presf = f.variables['presf'][()]
iotas = f.variables['iotas'][()]
pres = f.variables['pres'][()]
ns = f.variables['ns'][()]
nfp = f.variables['nfp'][()]
xn = f.variables['xn'][()]
xm = f.variables['xm'][()]
xn_nyq = f.variables['xn_nyq'][()]
xm_nyq = f.variables['xm_nyq'][()]
rmnc = f.variables['rmnc'][()]
zmns = f.variables['zmns'][()]
lmns = f.variables['lmns'][()]
bmnc = f.variables['bmnc'][()]
raxis_cc = f.variables['raxis_cc'][()]
zaxis_cs = f.variables['zaxis_cs'][()]
buco = f.variables['buco'][()]
bvco = f.variables['bvco'][()]
jcuru = f.variables['jcuru'][()]
jcurv = f.variables['jcurv'][()]
lasym = f.variables['lasym__logical__'][()]
if lasym==1:
    rmns = f.variables['rmns'][()]
    zmnc = f.variables['zmnc'][()]
    lmnc = f.variables['lmnc'][()]
    bmns = f.variables['bmns'][()]
    raxis_cs = f.variables['raxis_cs'][()]
    zaxis_cc = f.variables['zaxis_cc'][()]
else:
    rmns = 0*rmnc
    zmnc = 0*rmnc
    lmnc = 0*rmnc
    bmns = 0*bmnc
    raxis_cs = 0*raxis_cc
    zaxis_cc = 0*raxis_cc


print("nfp: ",nfp)
print("ns: ",ns)

mpol = f.variables['mpol'][()]
print("mpol: ",mpol)

ntor = f.variables['ntor'][()]
print("ntor: ",ntor)

Aminor_p = f.variables['Aminor_p'][()]
print("Aminor_p: ",Aminor_p)

Rmajor_p = f.variables['Rmajor_p'][()]
print("Rmajor_p: ",Rmajor_p)

data = f.variables['aspect'][()]
print("aspect:            ",data)

data = f.variables['betatotal'][()]
print("betatotal: ",data)

data = f.variables['betapol'][()]
print("betapol:   ",data)

data = f.variables['betator'][()]
print("betator:   ",data)

data = f.variables['betaxis'][()]
print("betaxis:   ",data)

ctor = f.variables['ctor'][()]
print("ctor:   ",ctor)

f.close()
nmodes = len(xn)

s = np.linspace(0,1,ns)
s_half = [(i-0.5)/(ns-1) for i in range(1,ns)]

phiedge = phi[-1]
phi_half = [(i-0.5)*phiedge/(ns-1) for i in range(1,ns)]

ntheta = 200
nphi = 4
theta = np.linspace(0,2*np.pi,num=ntheta)
phi = np.linspace(0,2*np.pi/nfp,num=nphi,endpoint=False)
iradius = ns-1
R = np.zeros((ntheta,nphi))
Z = np.zeros((ntheta,nphi))
for itheta in range(ntheta):
    for iphi in range(nphi):
        for imode in range(nmodes):
            angle = xm[imode]*theta[itheta] - xn[imode]*phi[iphi]
            R[itheta,iphi] = R[itheta,iphi] + rmnc[iradius,imode]*np.cos(angle) + rmns[iradius,imode]*np.sin(angle)
            Z[itheta,iphi] = Z[itheta,iphi] + zmns[iradius,imode]*np.sin(angle) + zmnc[iradius,imode]*np.cos(angle)

Raxis = np.zeros(nphi)
Zaxis = np.zeros(nphi)
for iphi in range(nphi):
    for n in range(ntor+1):
        angle = -n*nfp*phi[iphi]
        Raxis[iphi] += raxis_cc[n]*np.cos(angle) + raxis_cs[n]*np.sin(angle)
        Zaxis[iphi] += zaxis_cs[n]*np.sin(angle) + zaxis_cc[n]*np.cos(angle)

print('max R at phi=0:', R[0, 0])

fig = plt.figure(figsize=myfigsize)

plt.plot(R[:,0], Z[:,0], 'r',label=r'$\phi=0$')
plt.plot(R[:,1], Z[:,1], 'g',label='1/4 period')
plt.plot(R[:,2], Z[:,2], 'b',label='1/2 period')
plt.gca().set_aspect('equal',adjustable='box')
plt.legend(fontsize='x-small')
plt.xlabel('R [meters]')
plt.ylabel('Z [meters]')

##############################################################

plt.tight_layout()
if save_fig:
    print('Saving PDF')
    plt.savefig(f"vmec_xsections.pdf")
else:
    plt.show()
