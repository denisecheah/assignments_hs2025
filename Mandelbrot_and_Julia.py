# Assignment (Simulations in the Natural Sciences)

import matplotlib.image as img
from matplotlib import cm
import numpy as np
from numba import jit

resolution = 1000

x = np.linspace(-2.0, 1.0, resolution) # real axis
y = np.linspace(-1.5, 1.5, resolution) # imaginary axis

X, Y = np.meshgrid(x, y)
Z0 = X + 1j * Y
Z = np.copy(Z0)
rmax = 2.0
iterations = np.zeros(Z0.shape, dtype=np.float64)

maxiter = 200

# Loop style version of the Mandelbrot calculation, sped up with numba jit
@jit(nopython=True)
def Mandelbrot(Z, Z0, iterations, rmax, maxiter):
    nx, ny = Z.shape
    for i in range(maxiter):
        for ix in range(nx):
            for iy in range(ny):
                if abs(Z[ix, iy]) < rmax:
                    Z[ix, iy] = Z[ix, iy] ** 2 + Z0[ix, iy]
                    iterations[ix, iy] = i
    return iterations

Mandelbrot(Z, Z0, iterations, rmax, maxiter)
img.imsave('mandelbrot.png', iterations, cmap=cm.hot)

del x, y, X, Y, Z0, Z, rmax, iterations

# Loop style version of the Julia calculation, sped up with numba jit
@jit(nopython=True)
def Julia(Z, julia, c, rmax, maxiter):
    nx, ny = Z.shape
    for i in range(maxiter):
        for ix in range(nx):
            for iy in range(ny):
                if abs(Z[ix, iy]) < rmax:
                    Z[ix, iy] = Z[ix, iy] ** 2 + c
                    julia[ix, iy] = i
    return julia

constants = [
    -0.8 + 0.156j,
    0.285 + 0.01j,
    -0.4 + 0.6j,
    0.285 + 0.013j
] # constants to test

rmax = 2
maxiter = 200
x = np.linspace(-1.5, 1.5, resolution)
y = np.linspace(-1.5, 1.5, resolution)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

for i, c in enumerate(constants):
    julia = np.zeros_like(Z, dtype=np.float64)
    Julia(Z.copy(), julia, c, rmax, maxiter)
    img.imsave(f'julia_{i+1}.png', julia, cmap=cm.hot)

print("Mandelbrot and Julia sets saved")
