import numpy as np 
import scipy.optimize as op
import scipy.stats as st
import matplotlib.pyplot as plt
import corner

def penalty(vec):
    return (np.dot(vec, vec) - 1.) ** 2

def scalars(vec, basis, data):
    return np.sum(data * np.dot(basis, vec)[None,:], axis=1)

def obj_kurtosis(vec, basis, data):
    this = vec / np.sqrt(np.dot(vec, vec)) # TOTALLY UNNECESSARY
    return st.kurtosis(scalars(this, basis, data)) + penalty(vec)

def obj_skew(vec, basis, data):
    this = vec / np.sqrt(np.dot(vec, vec)) # TOTALLY UNNECESSARY
    return st.skew(scalars(this, basis, data)) + penalty(vec)

def obj_variance(vec, basis, data):
    this = vec / np.sqrt(np.dot(vec, vec)) # MUST DO!
    return -1. * np.var(scalars(this, basis, data)) + penalty(vec)

def orthogonalize(basis):
    D, DD = basis.shape
    basis[0] /= np.sqrt(np.dot(basis[0], basis[0]))
    for i in range(1,D):
        for j in range(i):
            basis[i] -= np.dot(basis[i], basis[j]) * basis[j]
        basis[i] /= np.sqrt(np.dot(basis[i], basis[i]))

def main(data, objective, name):
    N, D = data.shape
    basis = np.eye(D)
    figure = corner.corner(data)
    figure.savefig("unrotated.png")
    for d in range(D-1):

        # initialize
        vec0 = np.zeros(D-d)
        vec0[0] = 1.
        subbasis = basis[d:].T

        # optimize
        def cb(vec): print(vec, np.dot(vec, vec))
        cb(vec0)
        vec1 = op.fmin_powell(objective, vec0, args=(subbasis, data), callback=cb) 
        cb(vec1)

        # plot
        plt.clf()
        plt.hist(scalars(vec1, subbasis, data), bins=1000)
        plt.xlabel("optimized projection")
        plt.savefig("{}_{:02d}.png".format(name, d))
    
        # orthogonalize
        print("before orthogonalization", objective(vec1, subbasis, data))
        basis[d] = np.dot(subbasis, vec1)
        print(basis[d])
        orthogonalize(basis)
        print(basis[d])
        print("after orthogonalization", objective(basis[d], np.eye(D), data))

    rotated_data = np.dot(data, basis.T)
    figure = corner.corner(rotated_data)
    figure.savefig("{}.png".format(name))

if __name__ == "__main__":
    
    filein2 = 'play_cnalmgnaosvmnni.txt' # ouch, ascii
    t,g,feh,alpha,c,n,o,na,mg,al,s,v,mn,ni = np.loadtxt(filein2, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack =1, dtype = float) 
    data = np.vstack((feh,c,n,o,na,mg,al,s,v,mn,ni) ).T
    data = data[feh > -1.5] # according to MKN

    # choose the objective
    for objective, name in [(obj_variance, "variance"),
                            (obj_skew, "skew"),
                            (obj_kurtosis, "kurtosis")]:
        main(data, objective, name)

