"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).
"""
import os
import pickle as cp
import numpy as np 
import sklearn.cluster as cl
from astropy.io import fits
import pylab as plt

Xsun = -8.0 # kpc MAGIC

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

def get_data(fn):
    data_labels = ["FE_H",
                   "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
                   "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"]
    metadata_labels = np.array(["APOGEE_ID", "RA", "DEC", "GLON", "GLAT", "VHELIO_AVG", "DIST_Padova",
                                "TEFF_ASPCAP", "LOGG_ASPCAP"])

    # read FITS file
    print("get_data(): reading table from", fn)
    hdulist = fits.open(fn)
    hdu = hdulist[1]
    cols = hdu.columns
    table = hdu.data
    mask = ((table.field("TEFF_ASPCAP") > 3500.) *
            (table.field("TEFF_ASPCAP") < 5500.) *
            (table.field("LOGG_ASPCAP") > 0.) *
            (table.field("LOGG_ASPCAP") < 3.9) *
            (table.field("DIST_Padova") > 0.))
    data = np.vstack((table.field(label) for label in data_labels)).T
    metadata = np.vstack((table.field(label) for label in metadata_labels)).T
    mask *= np.all(np.isfinite(data), axis=1)
    data = data[mask]
    metadata = metadata[mask]

    # make distances and Galactic coordinates
    distances = (metadata[:, metadata_labels == "DIST_Padova"]).flatten().astype(float)
    ls = (np.pi / 180.) * (metadata[:, metadata_labels == "GLON"]).flatten().astype(float)
    bs = (np.pi / 180.) * (metadata[:, metadata_labels == "GLAT"]).flatten().astype(float)
    print(np.cos(1.2))
    GXs = distances * np.cos(ls) * np.cos(bs) + Xsun
    GYs = distances * np.sin(ls) * np.cos(bs)
    GZs = distances * np.sin(bs)
    metadata = np.vstack((metadata.T, GXs, GYs, GZs)).T # T craziness
    metadata_labels = np.append(metadata_labels, ["GX", "GY", "GZ"])

    # done!
    print("get_data()", fn, data.shape, metadata.shape)
    return data, data_labels, metadata, metadata_labels

def stats_in_slices(data, xs):
    N = len(xs)
    NN, D = data.shape
    assert NN == N
    ranks = np.zeros(N)
    ranks[np.argsort(xs)] = np.arange(N)
    ranklims = np.arange(0.5, N, 512)
    M = len(ranklims)
    xmedians = np.zeros(M-2)
    medians = np.zeros((M-2, D))
    sigmas = np.zeros((M-2, D))
    rmses = np.zeros((M-2, D))
    for i in range(M-2):
        rmin, rmax = ranklims[i], ranklims[i+2]
        mask = (ranks > rmin) * (ranks < rmax)
        xmedians[i] = np.median(xs[mask])
        medians[i] = np.median(data[mask], axis=0)
        for d in range(D):
            diffs = data[mask, d] - medians[i, d]
            mad = np.median(np.abs(diffs))
            dmask = (np.abs(diffs) < (5. * mad))
            rmses[i, d] = np.sqrt(np.mean(diffs[dmask] ** 2))
            print(i, d, np.sum(mask), np.sum(dmask))
        sigmas[i] = rmses[i] / np.sqrt(np.sum(mask))
    return xmedians, medians, sigmas, rmses

def hogg_savefig(fn):
    print("hogg_savefig():", fn)
    return plt.savefig(fn)

if __name__ == "__main__":
    print("Hello World!")
    dir = "./gradients_figs"
    if not os.path.exists(dir):
        os.mkdir(dir)

    # read data
    dfn = "./data/cannon-distances.fits"
    pfn = "./data/gradients.pkl"
    try:
        print("attempting to read pickle", pfn)
        data, data_labels, metadata, metadata_labels = read_pickle_file(pfn)
        print(pfn, data.shape, metadata.shape)
    except:
        print("failed to read pickle", pfn)
        data, data_labels, metadata, metadata_labels = get_data(dfn)
        pickle_to_file(pfn, (data, data_labels, metadata, metadata_labels))
        print(pfn, data.shape, metadata.shape)

    # check and adjust data
    N, D = data.shape
    assert len(data_labels) == D
    NN, DD = metadata.shape
    assert N == NN
    assert len(metadata_labels) == DD
    plotdata = data.copy()
    plotdata_labels = data_labels.copy()
    for d in np.arange(1,D):
        plotdata[:,d] = data[:,d] - data[:,0]
        plotdata_labels[d] = data_labels[d] + " - " + data_labels[0]
    print(plotdata_labels)
    plotmetadata = metadata.copy()

    # compute shit
    Rs = (np.sqrt(plotmetadata[:, metadata_labels == "GX"].astype(float) ** 2 +
                  plotmetadata[:, metadata_labels == "GY"].astype(float) ** 2 +
                  plotmetadata[:, metadata_labels == "GZ"].astype(float) ** 2)).flatten()
    Zs = plotmetadata[:, metadata_labels == "GZ"].astype(float).flatten()

    # cut in vertical direction
    zcut = np.abs(metadata[:, metadata_labels == "GZ"].astype(float).flatten()) < 0.3 # kpc

    # cut in cylinder (around Sun)
    cylcut = (np.sqrt((plotmetadata[:, metadata_labels == "GX"].astype(float) - Xsun) ** 2 +
                      plotmetadata[:, metadata_labels == "GY"].astype(float) ** 2)).flatten() < 3.0 # kpc


    # compute shit
    Rmedians, Rdatamedians, Rsigmas, Rrmses = stats_in_slices(plotdata[zcut], Rs[zcut])
    Zmedians, Zdatamedians, Zsigmas, Zrmses = stats_in_slices(plotdata[cylcut], Zs[cylcut])

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
                  "V_H - FE_H":  "[V/Fe] (dex)",}

    # plot slices
    for medians, datamedians, sigmas, rmses, infix in [(Rmedians, Rdatamedians, Rsigmas, Rrmses, "R"),
                                                       (Zmedians, Zdatamedians, Zsigmas, Zrmses, "Z"),
                                                       ]:
        for d in range(len(plotdata_labels)):
            plotfn = dir + "/G" + infix + "_" + plotdata_labels[d].replace(" ", "") + ".png"
            plt.clf()
            plt.plot(medians, datamedians[:,d], "k-", alpha=0.5, lw=2)
            for j in range(len(medians)):
                plt.plot([medians[j], medians[j]],
                         [datamedians[j,d] - rmses[j,d], datamedians[j,d] + rmses[j,d]],
                         "k-", alpha=0.25)
                plt.plot([medians[j], medians[j]],
                         [datamedians[j,d] - sigmas[j,d], datamedians[j,d] + sigmas[j,d]],
                         "k-", lw=3)
            plt.xlabel("Galactic " + infix + " (kpc)")
            plt.ylabel(label_dict[plotdata_labels[d]])
            hogg_savefig(plotfn)
