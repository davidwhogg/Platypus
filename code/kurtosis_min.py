import numpy as np 
import scipy.optimize as op
import scipy.stats as st
import matplotlib.pyplot as plt

def penalty(vec):
    return np.abs(np.dot(vec, vec) - 1.)

def scalars(vec, basis, data):
    return np.sum(data * np.dot(basis, vec)[None,:], axis=1)

def objective(vec, basis, data):
    this = vec / np.sqrt(np.dot(vec, vec)) # TOTALLY UNNECESSARY
    return st.kurtosis(scalars(this, basis, data)) + penalty(vec)

def orthogonalize(basis):
    basis[0] /= np.sqrt(np.dot(basis[0], basis[0]))
    for i in range(1,D):
        for j in range(i):
            basis[i] -= np.dot(basis[i], basis[j]) * basis[j]
        basis[i] /= np.sqrt(np.dot(basis[i], basis[i]))

if __name__ == "__main__":
    
    filein2 = 'play_cnalmgnaosvmnni.txt'
    t,g,feh,alpha,c,n,o,na,mg,al,s,v,mn,ni = np.loadtxt(filein2, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack =1, dtype = float) 
    data = np.vstack((feh,c,n,o,na,mg,al,s,v,mn,ni) ).T
    N, D = data.shape

    # start the loop
    basis = np.eye(D)
    for d in range(D):

        # initialize
        vec0 = np.zeros(D-d)
        vec0[0] = 1.
        thisbasis = basis[d:].T

        # optimize
        def cb(vec): print(vec, np.dot(vec, vec))
        cb(vec0)
        vec1 = op.fmin_powell(objective, vec0, args=(thisbasis, data), callback=cb) 
        cb(vec1)

        # plot
        plt.clf()
        plt.hist(scalars(vec1, basis, data), bins=1000)
        plt.xlabel("optimized projection")
        plt.savefig("kurtosis_{:02d}.png".format(d))
    
        # orthogonalize
        basis[0] = vec1
        orthogonalize(basis)
