"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

## bugs:
- needs better labels on corner
"""
import os
import pickle as cp
import numpy as np 
import sklearn.cluster as cl
from astropy.io import fits
import pylab as plt

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

def get_data(fn, labels):
    print("get_data(): reading table")
    hdulist = fits.open(fn)
    hdu = hdulist[1]
    cols = hdu.columns
    table = hdu.data
    mask = ((table.field("TEFF_ASPCAP") > 3500.) *
            (table.field("TEFF_ASPCAP") < 5500.) *
            (table.field("LOGG_ASPCAP") > 0.) *
            (table.field("LOGG_ASPCAP") < 3.9)) # MKN recommendation
    metadata_labels = ["RA", "DEC", "TEFF_ASPCAP", "LOGG_ASPCAP"]
    metadata = np.vstack((table.field(label) for label in metadata_labels)).T
    mask *= np.all(np.isfinite(metadata), axis=1)
    data = np.vstack((table.field(label) for label in labels)).T
    mask *= np.all(np.isfinite(data), axis=1)
    data = data[mask]
    print("get_data()", dfn, data.shape)
    return data, mask

def get_metadata(fn, labels, mask):
    print("get_metadata(): reading table")
    hdulist = fits.open(fn)
    hdu = hdulist[1]
    cols = hdu.columns
    table = hdu.data
    metadata = np.vstack((table.field(label) for label in labels)).T
    metadata = metadata[mask]
    print("get_metadata()", dfn, metadata.shape)
    return metadata

def plot_one_cluster(data, labels, mask, name, dir):
    print("plot_one_cluster(): plotting", name)
    suffix = "png"
    for ly, lx, plotname in [
        ("Na_H - FE_H", "O_H - FE_H", "NaO"),
        ("C_H - FE_H", "N_H - FE_H", "CN"),
        ("AL_H - FE_H", "MG_H - FE_H", "AlMg"),
        ("S_H - FE_H", "AL_H - FE_H", "SAl"),
        ("DEC", "RA", "sky"),
        ("DEC", "VHELIO_AVG", "decv"),
        ("LOGG_ASPCAP", "TEFF_ASPCAP", "HR"),
        ]:
        y = np.where(np.array(labels) == ly)[0][0]
        x = np.where(np.array(labels) == lx)[0][0]
        plt.figure(figsize=(4,4))
        plt.subplots_adjust(left=0.2, right=0.96, bottom=0.2, top=0.96)
        plt.clf()
        kwargs = {"marker": ".", "ls": "none"}
        if mask is None:
            plt.plot(data[:,x], data[:,y], c="k", ms=1.0, alpha=0.20, **kwargs)
        else:        
            plt.plot(data[:,x], data[:,y], c="0.75", ms=1.0, alpha=0.20, **kwargs)
            plt.plot(data[mask,x], data[mask,y], c="k", ms=5.0, alpha=0.5, **kwargs)
        plt.ylim(range_dict[ly])
        plt.xlim(range_dict[lx])
        plt.ylabel(label_dict[ly])
        plt.xlabel(label_dict[lx])
        [l.set_rotation(45) for l in plt.gca().get_xticklabels()]
        [l.set_rotation(45) for l in plt.gca().get_yticklabels()]
        fn = "{}/{}_{}.{}".format(dir, name, plotname, suffix)
        plt.savefig(fn)
        print("plot_one_cluster(): plotting", fn)

if __name__ == "__main__":
    dfn = "./data/results-unregularized-matched.fits.gz"
    dir = "./kmeans_figs"
    if not os.path.exists(dir):
        os.mkdir(dir)
    suffix = "png"
    data = None

    scale_dict = {"FE_H": 0.0191707168068, # all from AC
                  "AL_H": 0.0549037045265,
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
                  "RA": (360., 0.),
                  "DEC": (-35., 90.),
                  "VHELIO_AVG": (-275., 275.),
                  "TEFF_ASPCAP": (5500., 3500.),
                  "LOGG_ASPCAP": (3.9, 0.0), }

    label_dict = {"FE_H": "[Fe/H] (dex)",
                  "AL_H - FE_H": "[Al/Fe] (dex)",
                  "CA_H - FE_H": "[Ca/Fe] (dex)",
                  "C_H - FE_H":  "[C/Fe] (dex)",
                  "K_H - FE_H":  "[K/Fe] (dex)",
                  "MG_H - FE_H": "[Mg/Fe] (dex)",
                  "MN_H - FE_H": "[Mn/Fe] (dex)",
                  "NA_H - FE_H": "[Na/Fe] (dex)",
                  "NI_H - FE_H": "[Ni/Fe] (dex)",
                  "N_H - FE_H":  "[N/Fe] (dex)",
                  "O_H - FE_H":  "[O/Fe] (dex)",
                  "SI_H - FE_H": "[Si/Fe] (dex)",
                  "S_H - FE_H":  "[S/Fe] (dex)",
                  "TI_H - FE_H": "[Ti/Fe] (dex)",
                  "V_H - FE_H":  "[V/Fe] (dex)",
                  "RA":          "RA (J2000 deg)",
                  "DEC":         "Dec (J2000 deg)",
                  "VHELIO_AVG":  "heliocentric RV (km/s)",
                  "TEFF_ASPCAP": "ASPCAP Teff (K)",
                  "LOGG_ASPCAP": "ASPCAP log g (dex)", }

    data_labels = ["FE_H",
                   "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
                   "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"]

    for K in 2 ** np.arange(3, 11):
        pfn = "data/kmeans_{:04d}.pkl".format(K)
        try:
            print("attempting to read pickle", pfn)
            data, data_labels, mask, clusters, centers = read_pickle_file(pfn)
            print(data.shape, len(mask), np.sum(mask), len(clusters), centers.shape)
            K, D = centers.shape
            N, DD = data.shape
            assert D == DD
            assert N == np.sum(mask)
            NN = len(clusters)
            assert N == NN
            assert np.max(clusters) + 1 == K
            print(N, K, D)

        except:
            print("failed to read pickle", pfn)
            data, mask = get_data(dfn, data_labels)

            # note the HORRIBLE metric (non-affine) hack here
            print("running k-means at", K)
            km = cl.KMeans(n_clusters=K, random_state=42, n_init=32)
            scales = np.array([scale_dict[label] for label in data_labels])
            clusters = km.fit_predict(data / scales[None, :]) # work with scaled data!
            centers = km.cluster_centers_.copy()
            print(centers.shape)

            print("writing pickle")
            pickle_to_file(pfn, (data, data_labels, mask, clusters, centers))
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
    metadata_labels = ["RA", "DEC", "VHELIO_AVG", "TEFF_ASPCAP", "LOGG_ASPCAP"]
    metadata = get_metadata(dfn, metadata_labels, mask)
    plotdata = np.hstack((plotdata, metadata))
    plotdata_labels = plotdata_labels + metadata_labels

    # plot clusters from the last K run
    plotcount = 0
    for k in (np.argsort(densities))[::-1]:
        clustername = "cluster_{:04d}_{:04d}".format(K, k)
        plot_one_cluster(plotdata, plotdata_labels, (clusters==k), clustername, dir)
        plotcount += 1
        if plotcount == 16:
            break
