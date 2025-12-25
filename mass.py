import numpy as np
import matplotlib.pyplot as plt
import os

data = np.loadtxt('mass.dat')
misc = np.loadtxt('misc.dat')

nx = int(misc[0])
ny = int(misc[1])
dt = misc[3]
D = misc[4]
d = misc[2]
s = 10*d

cku = np.unique(data[:,0]).astype(int)

print(f'Frames: {len(cku)}')

plt.figure(figsize=(12.8,3.2))
for ck in cku:
    mask = data[:,0]==ck
    t = ck*dt
    g=data[mask,1].reshape((nx+1,ny+1)).T
    plt.imshow(g,origin='lower',vmin=0.0,cmap='bone'
               #,vmax=1/(2*3.1416*s*s)
               )
    #plt.tight_layout()
    plt.title(rf'$t={t:.3f}s$')
    filename=os.path.join('frames',f'frame_{ck//cku[1]:05d}.png')
    plt.savefig(filename)
    plt.clf()
plt.close()