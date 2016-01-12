"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

## bugs:
- needs a concept of a metric in the space (or a rescaling of the data)
"""
import os
import pickle as cp
import numpy as np 
import sklearn.cluster as cl
from astropy.io import fits
from corner import corner

def pickle_to_file(fn, stuff):
    fd = open(fn, "wb")
    cp.dump(stuff, fd)
    print("writing", fn)
    fd.close()

def read_pickle_file(fn):
    fd = open(fn, "rb")
    stuff = cp.load(fd)
    fd.close()
    return stuff

if __name__ == "__main__":
    dir = "./kmeans_figs"
    if not os.path.exists(dir):
        os.mkdir(dir)
    suffix = "png"
    data = None
    metadata = None
    
    for K in 2 ** np.arange(3, 10):
        pfn = "data/kmeans_{:04d}.pkl".format(K)
        try:
            print("attempting to read pickle", pfn)
            data, metadata, clusters, centers = read_pickle_file(pfn)
            K, D = centers.shape
            N, DD = data.shape
            assert D == DD
            NN = len(clusters)
            assert N == NN
            assert np.max(clusters) + 1 == K
            print(N, K, D)

        except:
            print("failed to read pickle", pfn)

            if (data is None) or (metadata is None):
                print("reading data")
                dfn = "data/results-unregularized-matched.fits.gz"
                hdulist = fits.open(dfn)
                hdu = hdulist[1]
                cols = hdu.columns
                table = hdu.data
                okay = ((table.field("TEFF_ASPCAP") > 3500.) *
                        (table.field("TEFF_ASPCAP") < 5500.) *
                        (table.field("LOGG_ASPCAP") > 0.) *
                        (table.field("LOGG_ASPCAP") < 3.5))
                table = table[okay]
                metadata_labels = ["RA", "DEC", "TEFF_ASPCAP", "LOGG_ASPCAP"]
                metadata = np.vstack((table.field(label) for label in metadata_labels)).T
                okay = np.all(np.isfinite(metadata), axis=1)
                table = table[okay]
                metadata = metadata[okay]
                data_labels = ["AL_H", "CA_H", "C_H", "FE_H", "K_H", "MG_H", "MN_H", "NA_H",
                               "NI_H", "N_H", "O_H", "SI_H", "S_H", "TI_H", "V_H"]
                data = np.vstack((table.field(label) for label in data_labels)).T
                okay = np.all(np.isfinite(data), axis=1)
                table = table[okay]
                metadata = metadata[okay]
                data = data[okay]
                print(dfn, metadata.shape, data.shape)

                print("plotting metadata")
                cfn = "{}/metadata.{}".format(dir, suffix)
                figure = corner(metadata, labels=metadata_labels, bins=128)
                figure.savefig(cfn)
                print(cfn)

                print("plotting data")
                cfn = "{}/data.{}".format(dir, suffix)
                figure = corner(data, labels=data_labels, bins=128)
                figure.savefig(cfn)
                print(cfn)

                assert False

            if False:
                filein2 = 'data/play_cnalmgnaosvmnni.txt' # ouch, ascii
                t,g,feh,alpha,c,n,o,na,mg,al,s,v,mn,ni = \
                    np.loadtxt(filein2, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack=1, dtype=float) 
                data = np.vstack(
                         ( feh,         c,           n,           o,           na,
                           mg,          al,          s,           v,           mn,          ni) ).T
                labels = ["Fe",        "C",         "N",         "O",         "Na",
                          "Mg",        "Al",        "S",         "V",         "Mn",        "Ni"]
                ranges = [(-1.5, 0.6), (-0.5, 0.6), (-0.7, 0.7), (-0.2, 0.5), (-2.0, 1.0),
                          (-0.2, 0.6), (-1.7, 0.8), (-0.2, 0.7), (-2.0, 0.9), (-2.0, 1.0), (-2.0, 0.7)]
                metadata = {"labels": labels, "ranges": ranges}
                data = data[feh > -1.5] # according to MKN, but I don't believe her
                print(data.shape)

            print("running k-means at", K)
            km = cl.KMeans(n_clusters=K, random_state=42, n_init=32)
            clusters = km.fit_predict(data)
            centers = km.cluster_centers_.copy()
            print(centers.shape)

            print("writing pickle")
            pickle_to_file(pfn, (data, metadata, clusters, centers))
            print(pfn)
        
        print("analyzing clusters")
        N, D = data.shape
        sizes = np.zeros(K).astype(int)
        logdets = np.zeros(K)
        for k in range(K):
            I = (clusters == k)
            sizes[k] = np.sum(I)
            subdata = data[I]
            if sizes[k] > (D + 1):
                variance = np.sum(subdata[:,:,None] * subdata[:,None,:], axis=0)
                s, logdets[k] = np.linalg.slogdet(variance)
                assert s > 0
            else:
                logdets[k] = -np.Inf
        densities = sizes * np.exp(-0.5 * logdets)
        print(K, "size range:", np.min(sizes), np.median(sizes), np.max(sizes))
        print(K, "logdet range:", np.min(logdets), np.median(logdets), np.max(logdets))
        print(K, "density range", np.min(densities), np.median(densities), np.max(densities))

if False:
    for k in range(-1,K):
        print("plotting cluster", k)
        if k < 0:
            subdata = data
            fn = "{}/data.{}".format(dir, suffix)
        else:
            subdata = data[clusters == k]
            fn = "{}/cluster_{:04d}.{}".format(dir, k, suffix)
        print(subdata.shape)
        N, D = subdata.shape
        if N > D:
            figure = corner(subdata, range=ranges, labels=labels, color="k",
                            bins=128, plot_datapoints=True, plot_density=True, plot_contours=True)
            figure.savefig(fn)
            print(fn)
