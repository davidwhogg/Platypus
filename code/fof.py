"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

## bugs:
- does not create `dir`
- needs a concept of a metric in the space (or a rescaling of the data)
- does nothing
"""
import numpy as np 
import matplotlib.pyplot as plt
import corner
from scipy.spatial import KDTree
import scipy.sparse as sp
import pickle as cp

def pickle_to_file(fn, stuff):
    fd = open(fn, "wb")
    cp.dump(stuff, fd)
    print("writing", fn)
    fd.close()

if __name__ == "__main__":
    dir = "fof_figs"
    suffix = "png"
    
    print("reading data")
    filein2 = 'data/play_cnalmgnaosvmnni.txt' # ouch, ascii
    t,g,feh,alpha,c,n,o,na,mg,al,s,v,mn,ni = np.loadtxt(filein2, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack =1, dtype = float) 
    data = np.vstack((feh,c,n,o,na,mg,al,s,v,mn,ni) ).T
    data = data[feh > -1.5] # according to MKN, but I don't believe her
    N, D = data.shape
    print(data, data.shape)

    print("ensmallening data")
    data = data[np.random.uniform(size=N) < 0.1] # magic number
    N, D = data.shape
    print(data, data.shape)

    if False:
        print("plotting corner")
        figure = corner.corner(data)
        fn = "{}/data.{}".format(dir, suffix)
        figure.savefig(fn)
        print(fn)

    print("making KD tree")
    tree = KDTree(data)
    print(tree)

    print("making graph data")
    graph = tree.sparse_distance_matrix(tree, max_distance=0.1) # magic number
    print(graph.shape, len(graph))

    print("finding connected components")
    cc = sp.csgraph.connected_components(graph, directed=False, return_labels=True)
    print(cc, cc[1].shape)

    print("writing pickle")
    pfn = 'data/cc.pkl'
    pickle_to_file(pfn, (data, cc))
    print(pfn)

    K, ks = cc
    for k in range(K):
        I = (ks == k)
        if np.sum(I) > 16: # magic number
            kstr = "{:05d}".format(k)
            print("plotting corner", kstr, np.sum(I))
            figure = corner.corner(data)
            fn = "{}/group_{}.{}".format(dir, kstr, suffix)
            figure.savefig(fn)
            print(fn)
