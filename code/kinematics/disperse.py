# This file is part of the Platypus project.
# Copyright 2016 David W. Hogg.

import numpy as np

CIRCULAR_VELOCITY = 220 * 1000. # m / s (Julianne Dalcanton)
KPC = 3.086e19 # m (Google)
GYR = 3.1556926e16 # s (Google)

def flat_rotation_curve(x):
    return -1. * CIRCULAR_VELOCITY * CIRCULAR_VELOCITY * x / np.dot(x, x)

def leapfrog_step(delta_t, x_zero, v_half, accelerationlaw):
    x_one = x_zero + v_half * delta_t
    v_one_and_a_half = v_half + accelerationlaw(x_one) * delta_t
    return x_one, v_one_and_a_half

if __name__ == "__main__":
    import pylab as plt
    x0 = np.array((8.0 * KPC, 0., 0.))
    v0 = np.array((0., CIRCULAR_VELOCITY, 0.))
    x1 = 1. * x0
    v1 = 1. * v0 + 1000. * np.random.normal(3) # add a 1 km/s dispersion
    v0 = 1. * v0 + 1000. * np.random.normal(3) # add a 1 km/s dispersion
    dv = np.sqrt(np.dot(v0 - v1, v0 - v1))
    nsteps = 10000
    x0s = np.zeros((nsteps, 3))
    v0s = np.zeros_like(x0s)
    x1s = np.zeros_like(x0s)
    v1s = np.zeros_like(x0s)
    dt = 0.0003 * GYR
    for n in range(nsteps):
        x0, v0 = leapfrog_step(dt, x0, v0, flat_rotation_curve)
        x1, v1 = leapfrog_step(dt, x1, v1, flat_rotation_curve)
        x0s[n], v0s[n] = x0, v0
        x1s[n], v1s[n] = x1, v1
    ts = dt * np.arange(nsteps)
    plt.clf()
    plt.plot(x0s[:,0] / KPC, x0s[:,1] / KPC, "k-", alpha=0.8)
    plt.plot(x1s[:,0] / KPC, x1s[:,1] / KPC, "k-", alpha=0.8)
    plt.axis("equal")
    plt.xlabel("X (kpc)")
    plt.ylabel("Y (kpc)")
    plt.savefig("foo.png")
    plt.clf()
    plt.plot((x0s[:,0] - x1s[:,0]) / KPC, (x0s[:,1] - x1s[:,1]) / KPC, "k-", alpha=0.8)
    plt.axis("equal")
    plt.xlabel("Delta-X (kpc)")
    plt.ylabel("Delta-Y (kpc)")
    plt.savefig("bar.png")
    plt.clf()
    distances = np.sqrt(np.sum((x0s - x1s) ** 2, axis=1))
    plt.plot(ts / GYR, distances / KPC, "k-", alpha=0.8)
    tps = np.array(plt.xlim())
    plt.plot(tps, dv * (tps * GYR) / KPC, "k-", alpha=0.25)
    plt.xlabel("time (Gyr)")
    plt.ylabel("separation (kpc)")
    plt.savefig("rix.png")
