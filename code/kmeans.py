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
    ranges = [(-1.5, 0.5), (-0.6, 0.6), (-0.6, 0.6), (-0.3, 0.5), (-1.5, 1.0),
              (-0.2, 0.6), (-1.5, 0.7), (-0.2, 0.6), (-1.5, 0.8), (-1.5, 1.0), (-1.5, 0.6)]
    data = data[feh > -1.5] # according to MKN, but I don't believe her
    N, D = data.shape
    print(data.shape)

    print("running k-means")
    km = cl.KMeans(n_clusters=16, random_state=42, n_init=1)
    clusters = km.fit_predict(data)
    centers = km.cluster_centers_.copy()
    print(labels)
    print(centers)
    print(centers.shape)

    if True:
        print("plotting corner")
        # corner.corner() order matters, apparently
        figure = corner.corner(data, range=ranges, labels=labels, color="b",
                               bins=128, plot_datapoints=True, plot_density=True, plot_contours=True)
        corner.corner(centers, range=ranges, color="r", fig=figure,
                      plot_datapoints=True, plot_density=False, plot_contours=False)
        fn = "{}/centers.{}".format(dir, suffix)
        figure.savefig(fn)
        print(fn)
