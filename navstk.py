import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

data = np.loadtxt('psi.dat')
misc = np.loadtxt('misc.dat')

psi = data[:,0]
nu = data[:,1]
nv = data[:,2]
mag = data[:,3]
brzeg = data[:,4]

nx = int(misc[0])
ny = int(misc[1])
d = misc[2]

x = np.arange(0,nx+1,1)
y = np.arange(0,ny+1,1)
X, Y = np.meshgrid(x,y)
NU = nu.reshape((nx+1, ny+1)).T
NV = nv.reshape((nx+1, ny+1)).T
MAG = mag.reshape((nx+1, ny+1)).T

colors = ["#000000", "#000000"]
bbr_cmap = LinearSegmentedColormap.from_list('bbr', colors, N=2)


    # PSI
plt.figure(figsize=(12.8,3.2))
plt.contour(psi.reshape((nx+1,ny+1)).T, cmap=bbr_cmap, levels=20)
plt.title(rf'Stream function $\psi(i,j)$')
plt.xlim(0,nx+1)
plt.ylim(0,ny+1)
filename = os.path.join('images','stream.png')
plt.savefig(filename)
plt.close()

skip = nx//64


    # colored velocity field
plt.figure(figsize=(12.8,3.2))
plt.quiver(X[::skip, ::skip],Y[::skip, ::skip],NU[::skip, ::skip],NV[::skip, ::skip]
           , MAG[::skip, ::skip], cmap='jet'
           )
plt.xlim(0,nx+1)
plt.colorbar(orientation='horizontal')
plt.title(rf'Velocity field with corresponding magnitude $\vec{{v}}$')
plt.ylim(0,ny+1)
filename = os.path.join('images','vfieldc.png')
plt.savefig(filename)
plt.close()


    # black velocity field
plt.figure(figsize=(12.8,3.2))
plt.quiver(X[::skip, ::skip],Y[::skip, ::skip],NU[::skip, ::skip],NV[::skip, ::skip])
plt.xlim(0,nx+1)
plt.title(rf'Normalized velocity field $\vec{{v}}/|\vec{{v}}|$')
plt.ylim(0,ny+1)
filename = os.path.join('images','vfieldb.png')
plt.savefig(filename)
plt.close()


    # geometria
plt.figure(figsize=(12.8,3.2))
plt.imshow(brzeg.reshape((nx+1,ny+1)).T, origin='lower', cmap='cividis')
plt.xlim(0,nx+1)
plt.ylim(0,ny+1)
plt.title('System geometry')
filename = os.path.join('images','brzeg.png')
plt.savefig(filename)
plt.close()