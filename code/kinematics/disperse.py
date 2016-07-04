# This file is part of the Platypus project.
# Copyright 2016 David W. Hogg.

import numpy as np

CIRCULAR_VELOCITY = 220 * 1000. # m / s (Julianne Dalcanton)
PARSEC = 3.086e16 # m (Google)
GYR = 3.1556926e13 # s (Google)

def flat_rotation_curve(x):
    return -1. * CIRCULAR_VELOCITY * CIRCULAR_VELOCITY * x / np.dot(x, x)

def leapfrog_step(delta_t, x_zero, v_half, accelerationlaw):
    x_one = x_zero + v_half * delta_t
    v_one_and_a_half = v_half + accelerationlaw(x_one) * delta_t
    return x_one, v_one_and_a_half

if __name__ == "__main__":
    import pylab as plt
    x_00 = np.array((8.0 * PARSEC, 0., 0.))
    v_05 = np.array((0., CIRCULAR_VELOCITY, 0.))
    nsteps = 1000
    xs = np.zeros((nsteps, 3))
    vs = np.zeros_like(xs)
    dt = 0.01 * GYR
    for n in range(nsteps):
        x_00, v_05 = leapfrog_step(dt, x_00, v_05, flat_rotation_curve)
        xs[n], vs[n] = x_00, v_05
    plt.clf()
    plt.plot(xs[:,0], xs[:,1], "k-", alpha=0.8)
    plt.savefig("foo.png")
