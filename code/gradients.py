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
    data_labels = np.array(["FE_H",
                            "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
                            "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"])
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
    GXs = distances * np.cos(ls) * np.cos(bs) + 8.0 # kpc MAGIC
    GYs = distances * np.sin(ls) * np.cos(bs)
    GZs = distances * np.sin(bs)
    metadata = np.vstack((metadata.T, GXs, GYs, GZs)).T # T craziness
    metadata_labels = np.append(metadata_labels, ["GX", "GY", "GZ"])

    # done!
    print("get_data()", fn, data.shape, metadata.shape)
    return data, data_labels, metadata, metadata_labels

def stats_in_slices(data, xs):
    M = 32
    N = len(xs)
    assert len(data) == N
    ranks = np.zeros(N)
    ranks[np.argsort(xs)] = np.arange(N)
    ranklims = np.arange(N / M, N, N / M)
    xmedians = []
    medians = []
    rmses = []
    for rmin, rmax in zip(ranklims[:-2], ranklims[2:]):
        mask = (ranks > rmin) * (ranks < rmax)
        xmedians.append(np.median(xs[mask]))
        medians.append(np.median(data[mask], axis=0))
    return np.array(xmedians), np.array(medians)

def hogg_savefig(fn):
    print("hogg_savefig():", fn)
    return plt.savefig(fn)

if __name__ == "__main__":
    print("Hello World!")

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

    # compute shit
    Rs = (np.sqrt(metadata[:, metadata_labels == "GX"].astype(float) ** 2 +
                  metadata[:, metadata_labels == "GY"].astype(float) ** 2 +
                  metadata[:, metadata_labels == "GZ"].astype(float) ** 2)).flatten()
    Rmedians, datamedians = stats_in_slices(data, Rs)

    # plot whole sample
    plt.clf()
    plt.plot(metadata[:, metadata_labels == "GX"],
             metadata[:, metadata_labels == "GY"], "k.", alpha=0.25)
    plt.xlabel("Galactic X (kpc)")
    plt.ylabel("Galactic Y (kpc)")
    hogg_savefig("GX_GY.png")

    # plot slices
    for i in range(len(data_labels)):
        pfn = data_labels[i] + "_GR.png"
        plt.clf()
        plt.plot(Rmedians, datamedians[:,i], "ko")
        plt.xlabel("Galactic R (kpc)")
        plt.ylabel(data_labels[i] + " (dex)")
        hogg_savefig(pfn)
