#
import numpy as np 
import scipy.optimize as op
import scipy.stats as st
import matplotlib.pyplot as plt

def scalars(vec, data):
    return np.sum(data * vec[None,:], axis=1)

def objective(vec, data):
    return st.kurtosis(scalars(vec, data))

if __name__ == "__main__":
    
    filein2 = 'play_cnalmgnaosvmnni.txt'
    t,g,feh,alpha,c,n,o,na,mg,al,s,v,mn,ni = loadtxt(filein2, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack =1, dtype = float) 
    data = np.vstack((feh,c,n,o,na,mg,al,s,v,mn,ni) ).T
    N,D = data.shape

    # initialize
    np.random.seed(42)
    vec0 = np.random.normal(size=D)

    # optimize
    def cb(vec): print vec
    vec1 = op.fmin_powell(objective, vec0, args=(data, ), callback=cb) 
    print vec1

    # plot
    plt.clf()
    plt.hist(scalars(vec1, data), bins=1000)
    plt.xlabel("optimized projection")
    plt.savefig("kurtosis.png")
    
