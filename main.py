# fourier transform 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size': 18})

# define domain
dx = 0.001
L = np.pi
x = L * np.arange(-1+dx, 1+dx, dx)
n = len(x)
nquart = int (np.floor(n/4))

# define hat function
f = np.zeros_like(x)
f[nquart:2*nquart] =(4/n)*np.arange(1, nquart+1)
f[2*nquart : 3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)
fig, ax = plt.subplots()
ax.plot(x,f,'-', color = 'k' , linewidth = 2)

# Compute Fourier Series

name = "Accent"
cmap = plt.get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle( color=colors)

A0 = np.sum(f * np.ones_like(x)) * dx
FFS = A0/2

A = np.zeros(100)
B = np.zeros(100)

for k in range (100):
    A[k] = np.sum(f * np.cos((k+1) * np.pi * x / L)) * dx
    B[k] = np.sum(f * np.sin((k+1) * np.pi * x / L)) * dx
    FFS = FFS + A[k] * np.cos((k+1) * np.pi * x / L) + B[k] * np.sin((k+1) * np.pi * x / L)
    ax.plot(x, FFS, '-')


# Plot ampltudes

FFS = ( A0/2 ) * np.ones_like(f)
kmax = 200
A = np.zeros(kmax)
B = np.zeros(kmax)
ERR = np.zeros(kmax)

A[0] = A0/2
ERR[0] = np.linalg.norm(f- FFS)/np.linalg.norm(f)

for k in range(1, kmax):
    A[k] = np.sum(f * np.cos(k * np.pi * x / L)) * dx
    B[k] = np.sum(f * np.sin(k * np.pi * x / L)) * dx
    FFS = FFS + A[k] * np.cos(k * np.pi * x / L) + B[k] * np.sin(k * np.pi * x / L)
    ERR[k] = np.linalg.norm(f- FFS)/np.linalg.norm(f)

threshold = np.median(ERR) * np.sqrt(kmax) * (4/np.sqrt(3))
r = np.max(np.where(ERR>threshold))

fig, axs = plt.subplots(2,1)

axs[0].semilogy(np.arange(kmax), A, color = 'k', linewidth = 1)
axs[0].semilogy(r, A[r], '.' , color = 'b', markersize = 8)
plt.sca(axs[0])
plt.title('Fourier Coeffecients', fontsize = '12')

axs[1].semilogy(np.arange(kmax), ERR, color = 'k', linewidth = 1)
axs[1].semilogy(r, ERR[r], '.' , color = 'b' , markersize = 8)
plt.sca(axs[1])
plt.title('Error', fontsize = '12')

plt.show()