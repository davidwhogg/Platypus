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

def get_data(fn, labels, metadata_labels):
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
    metadata = np.vstack((table.field(label) for label in metadata_labels)).T
    data = np.vstack((table.field(label) for label in labels)).T
    mask *= np.all(np.isfinite(data), axis=1)
    data = data[mask]
    metadata = metadata[mask]
    print("get_data()", fn, data.shape, metadata.shape, mask.shape)
    return data, metadata, mask

if __name__ == "__main__":
    print("Hello World!")

    data_labels = np.array(["FE_H",
                            "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
                            "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"])
    metadata_labels = np.array(["APOGEE_ID", "RA", "DEC", "GLON", "GLAT", "VHELIO_AVG", "DIST_Padova",
                                "TEFF_ASPCAP", "LOGG_ASPCAP"])
    dfn = "./data/cannon-distances.fits"
    data, metadata, mask = get_data(dfn, data_labels, metadata_labels)
    print(data.shape, metadata.shape, mask.shape)

    # TO-DO: MOVE THIS LOGIC INTO get_data
    distances = (metadata[:, metadata_labels == "DIST_Padova"]).flatten().astype(float)
    ls = (np.pi / 180.) * (metadata[:, metadata_labels == "GLON"]).flatten().astype(float)
    bs = (np.pi / 180.) * (metadata[:, metadata_labels == "GLAT"]).flatten().astype(float)
    print(np.cos(1.2))
    GXs = distances * np.cos(ls) * np.cos(bs) + 8.0 # kpc MAGIC
    GYs = distances * np.sin(ls) * np.cos(bs)
    GZs = distances * np.sin(bs)
    GRs = np.sqrt(GXs ** 2 + GYs ** 2 + GZs ** 2)

    # TO-DO: MAKE PRETTY PLOTS of slices
    plt.clf()
    plt.plot(GXs, GYs, "k.", alpha=0.25)
    plt.savefig("foo.png")

    # TO-DO: COMPUTE MEDIANS AND ROOT MEDIAN SQUARES OF ABUNDANCES in slices
