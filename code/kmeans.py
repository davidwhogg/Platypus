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
    data = np.vstack(
             ( feh,         c,           n,           o,           na,
               mg,          al,          s,           v,           mn,          ni) ).T
    labels = ["Fe",        "C",         "N",         "O",         "Na",
              "Mg",        "Al",        "S",         "V",         "Mn",        "Ni"]
    ranges = [(-1.5, 0.6), (-0.5, 0.6), (-0.7, 0.7), (-0.2, 0.5), (-2.0, 1.0),
              (-0.2, 0.6), (-1.7, 0.8), (-0.2, 0.7), (-2.0, 0.9), (-2.0, 1.0), (-2.0, 0.7)]
    data = data[feh > -1.5] # according to MKN, but I don't believe her
    N, D = data.shape
    print(data.shape)

    for K in 2 ** np.arange(3, 10):
        print("running k-means at", K)
        km = cl.KMeans(n_clusters=K, random_state=42, n_init=32)
        clusters = km.fit_predict(data)
        centers = km.cluster_centers_.copy()
        print(centers.shape)

        print("writing pickle")
        pfn = "kmeans_{:04d}.pkl".format(K)
        pickle_to_file(pfn, (data, clusters, centers))
        print(pfn)

if False:
    for k in range(-1,K):
        print("plotting corner", k)
        # corner.corner() order matters, apparently
        if k < 0:
            subdata = data
            fn = "{}/data.{}".format(dir, suffix)
        else:
            subdata = data[clusters == k]
            fn = "{}/cluster_{:04d}.{}".format(dir, k, suffix)
        print(subdata.shape)
        N, D = subdata.shape
        if N > D:
            figure = corner.corner(subdata, range=ranges, labels=labels, color="k",
                                   bins=128, plot_datapoints=True, plot_density=True, plot_contours=True)
            figure.savefig(fn)
            print(fn)
