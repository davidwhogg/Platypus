"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

## bugs:
- does not create `dir`
- does nothing
"""
import numpy as np 
import matplotlib.pyplot as plt
import corner
from scipy.spatial import KDTree

if __name__ == "__main__":
    
    filein2 = 'data/play_cnalmgnaosvmnni.txt' # ouch, ascii
    t,g,feh,alpha,c,n,o,na,mg,al,s,v,mn,ni = np.loadtxt(filein2, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack =1, dtype = float) 
    data = np.vstack((feh,c,n,o,na,mg,al,s,v,mn,ni) ).T
    data = data[feh > -1.5] # according to MKN
    dir = "fof_figs"
    suffix = "png"
    # figure = corner.corner(data)
    # figure.savefig("{}/data.{}".format(dir, suffix))

    tree = KDTree(data)
    print(tree.sparse_distance_matrix(tree, 0.01))
