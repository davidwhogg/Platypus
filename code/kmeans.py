"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

## bugs:
- need to set ranges on corner
- needs better labels on corner
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

    scale_dict = {"FE_H": 0.0191707168068,
                  "AL_H": 0.0549037045265, # all from AC
                  "CA_H": 0.0426365845422,
                  "C_H": 0.0405909985963,
                  "K_H": 0.0680897262727,
                  "MG_H": 0.0324021951804,
                  "MN_H": 0.0410348634747,
                  "NA_H": 0.111044350016,
                  "NI_H": 0.03438986215,
                  "N_H": 0.0440559383568,
                  "O_H": 0.037015877736,
                  "SI_H": 0.0407894206516,
                  "S_H": 0.0543424861906,
                  "TI_H": 0.0718311106542,
                  "V_H": 0.146438163035, }

    range_dict = {"FE_H": (-1.9, 0.6),
                  "AL_H - FE_H": (-0.6, 0.7),
                  "CA_H - FE_H": (-0.4, 0.4),
                  "C_H - FE_H": (-0.6, 0.7),
                  "K_H - FE_H": (-0.9, 0.5),
                  "MG_H - FE_H": (-0.25, 0.5),
                  "MN_H - FE_H": (-0.4, 0.5),
                  "NA_H - FE_H": (-1.4, 1.2),
                  "NI_H - FE_H": (-0.4, 0.3),
                  "N_H - FE_H": (-0.6, 0.9),
                  "O_H - FE_H": (-0.25, 0.5),
                  "SI_H - FE_H": (-0.25, 0.6),
                  "S_H - FE_H": (-0.4, 0.9),
                  "TI_H - FE_H": (-0.6, 0.5),
                  "V_H - FE_H": (-0.9, 0.8),
                  "RA": (0., 360.),
                  "DEC": (-35., 90.),
                  "TEFF_ASPCAP": (3500., 5500.),
                  "LOGG_ASPCAP": (0., 3.9), }

    for K in 2 ** np.arange(3, 10):
        pfn = "data/kmeans_{:04d}.pkl".format(K)
        try:
            print("attempting to read pickle", pfn)
            data, data_labels, metadata, metadata_labels, clusters, centers = read_pickle_file(pfn)
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
                        (table.field("LOGG_ASPCAP") < 3.9)) # MKN recommendation
                table = table[okay]
                metadata_labels = ["RA", "DEC", "TEFF_ASPCAP", "LOGG_ASPCAP"]
                metadata = np.vstack((table.field(label) for label in metadata_labels)).T
                okay = np.all(np.isfinite(metadata), axis=1)
                table = table[okay]
                metadata = metadata[okay]
                data_labels = ["FE_H",
                               "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
                               "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"]
                data = np.vstack((table.field(label) for label in data_labels)).T
                okay = np.all(np.isfinite(data), axis=1)
                table = table[okay]
                metadata = metadata[okay]
                data = data[okay]
                N, D = data.shape
                print(dfn, metadata.shape, data.shape)

            # note the HORRIBLE metric (non-affine) hack here
            print("running k-means at", K)
            km = cl.KMeans(n_clusters=K, random_state=42, n_init=32)
            scales = np.array([scale_dict[label] for label in data_labels])
            clusters = km.fit_predict(data / scales[None, :]) # work with scaled data!
            centers = km.cluster_centers_.copy()
            print(centers.shape)

            print("writing pickle")
            pickle_to_file(pfn, (data, data_labels, metadata, metadata_labels, clusters, centers))
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


    # make plotting data
    plotdata = data.copy()
    plotdata_labels = data_labels.copy()
    for d in range(1, D):
        plotdata[:,d] = data[:,d] - data[:,0]
        plotdata_labels[d] = data_labels[d] + " - " + data_labels[0]
    plotdata_ranges = [range_dict[label] for label in plotdata_labels]
    metadata_ranges = [range_dict[label] for label in metadata_labels]

    # plot clusters from the last K run
    plotcount = 0
    for k in (np.argsort(densities))[::-1]:
        print("considering cluster", k)
        subdata = plotdata[clusters == k]
        submetadata = metadata[clusters == k]
        print(subdata.shape)
        N, D = subdata.shape
        if N > D:
            clustername = "cluster_{:04d}_{:04d}".format(K, k)
            cfn = "{}/{}_data.{}".format(dir, clustername, suffix)
            figure = corner(subdata, labels=plotdata_labels, range=plotdata_ranges, bins=128)
            figure.savefig(cfn)
            print(cfn)
            cfn = "{}/{}_metadata.{}".format(dir, clustername, suffix)
            figure = corner(submetadata, labels=metadata_labels, range=metadata_ranges, bins=128)
            figure.savefig(cfn)
            print(cfn)
            plotcount += 1
            if plotcount == 16:
                break

    # plot all metadata
    print("plotting all metadata")
    cfn = "{}/metadata.{}".format(dir, suffix)
    figure = corner(metadata, labels=metadata_labels, range=metadata_ranges, bins=128)
    figure.savefig(cfn)
    print(cfn)

    # plot all data
    print("plotting all data")
    cfn = "{}/data.{}".format(dir, suffix)
    figure = corner(plotdata, labels=plotdata_labels, range=plotdata_ranges, bins=128)
    figure.savefig(cfn)
    print(cfn)
