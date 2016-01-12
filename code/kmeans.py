"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

## bugs:
- does not create `dir`
- needs a concept of a metric in the space (or a rescaling of the data)
- does nothing
"""
import numpy as np 
import sklearn.cluster as cl
import corner
import pickle as cp

def pickle_to_file(fn, stuff):
    fd = open(fn, "wb")
    cp.dump(stuff, fd)
    print("writing", fn)
    fd.close()

if __name__ == "__main__":
    dir = "kmeans_figs"
    suffix = "png"
    
    print("reading data")
    filein2 = 'data/play_cnalmgnaosvmnni.txt' # ouch, ascii
    t,g,feh,alpha,c,n,o,na,mg,al,s,v,mn,ni = np.loadtxt(filein2, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack =1, dtype = float) 
    data = np.vstack((feh,c,n,o,na,mg,al,s,v,mn,ni) ).T
    data = data[feh > -1.5] # according to MKN, but I don't believe her
    N, D = data.shape
    print(data, data.shape)

    if False:
        print("plotting corner")
        figure = corner.corner(data, color="0.5")
        fn = "{}/data.{}".format(dir, suffix)
        figure.savefig(fn)
        print(fn)

    print("running k-means")
    km = cl.KMeans()
    labels = km.fit_predict(data)
    centers = km.cluster_centers_.copy()
    print(labels)
    print(centers)

    if False:
        print("overplotting corner")
        corner.corner(centers, color="r", fig=figure)
        fn = "{}/centers.{}".format(dir, suffix)
        figure.savefig(fn)
        print(fn)
