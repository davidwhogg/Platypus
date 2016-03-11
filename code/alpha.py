"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

# alpha.py

## purpose
- Show that there is more than one kind of alpha element!

## bugs / notes
- Need to write derivatives for the model posterior.
- Ought to make it so we can optimize amplitudes and vectors separately.
- Plots are just TERRIBLE.
- Paranoid temperature cuts!
- Uses Jason Sanders's distances and Andy Casey's element abundances.
  Both of these are out-of-date and should be updated regularly.

"""
import os
import pickle as cp
import numpy as np 
from astropy.io import fits

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
    data_labels = ["FE_H", "O_H", "MG_H", "SI_H", "S_H", "CA_H"]
    metadata_labels = np.array(["APOGEE_ID", "RA", "DEC", "GLON", "GLAT", "VHELIO_AVG", "DIST_Padova",
                                "TEFF_ASPCAP", "LOGG_ASPCAP"])

    # read FITS file
    print("get_data(): reading table from", fn)
    hdulist = fits.open(fn)
    hdu = hdulist[1]
    cols = hdu.columns
    table = hdu.data
    mask = ((table.field("TEFF_ASPCAP") > 4500.) * # paranoid about cool stars!
            (table.field("TEFF_ASPCAP") < 5000.) * # paranoid about hot stars!
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
        sigmas[i] = rmses[i] / np.sqrt(np.sum(mask))
    return xmedians, medians, sigmas, rmses

def hogg_savefig(fn):
    print("hogg_savefig():", fn)
    return plt.savefig(fn)

class abundance_model:
    """
    Assumes that the `self.data` are [X/Y] `log_10` element ratios.
    """

    def __init__(self, data, ivars, K):
        N, D = data.shape
        self.N = N
        self.D = D
        self.data = data
        assert data.shape == ivars.shape
        self.ivars = ivars
        self.K = K
        print("initialized with ", N, D, K)

    def set_pars_from_vector(self, parsvec):
        self.amplitudes = (parsvec[0:self.N*self.K]).reshape((self.N,self.K))
        self.vectors = (parsvec[self.N*self.K:self.N*self.K+self.K*self.D]).reshape((self.K,self.D))

    def ln_prior(self):
        if np.any(self.amplitudes < 0.):
            return -np.Inf
        if np.any(self.vectors < 0.):
            return -np.Inf
        return -10000. * np.sum((self.vectors[:,0] - 1.) ** 2) # MAGIC: force [Fe/H] components near 1

    def predicted_data(self):
        return np.log10(np.dot(self.amplitudes, self.vectors))

    def ln_like(self):
        resids = self.data - self.predicted_data()
        return -0.5 * np.sum(resids * self.ivars * resids)

    def ln_post(self):
        lnp = self.ln_prior()
        if not np.isfinite(lnp):
            return -np.Inf
        return lnp + self.ln_like()

    def __call__(self, parsvec):
        self.set_pars_from_vector(parsvec)
        return -2. * self.ln_post()

if __name__ == "__main__":
    import pylab as plt
    import scipy.optimize as op

    print("Hello World!")
    dir = "./alpha_figs"
    if not os.path.exists(dir):
        os.mkdir(dir)

    # read data
    dfn = "./data/cannon-distances.fits"
    pfn = "./data/alpha.pkl"
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
    plotdata_labels = np.array(plotdata_labels)
    plotmetadata = metadata.copy()

    # make horrible fake uncertainties!
    ivars = np.zeros_like(plotdata)
    ivars[:,0] = 1. / 0.02 ** 2
    ivars[:,1:] = 1. / 0.05 ** 2

    # initialize model
    K = 2
    fitsubsample = np.random.randint(N, size=512)
    model = abundance_model(plotdata[fitsubsample], ivars[fitsubsample], K)
    amplitudes = 10. ** plotdata[fitsubsample,:K]
    vectors = np.ones((K, D))
    parvec0 = np.append(amplitudes.flatten(), vectors.flatten())
    result = op.minimize(model, parvec0, method="Powell")
    parvec1 = result["x"]
    print(model(parvec1)) # required to set parameters
    predicteddata = model.predicted_data()

    label_dict = {"FE_H": "[Fe/H] (dex)",
                  "alpha_FE": "[alpha/Fe] (dex)",
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

    # make scatterplots
    alpha_indexes = range(1,6)
    for yy in alpha_indexes:
        plotfn = dir + "/a" + plotdata_labels[yy].replace(" ", "") + ".png"
        plt.clf()
        plt.plot(plotdata[fitsubsample, 0], plotdata[fitsubsample, yy], "k.", ms=0.5, alpha=0.5)
        plt.plot(predicteddata[:, 0], predicteddata[:, yy], "r.", ms=1.0)
        plt.xlabel(label_dict[plotdata_labels[0]])
        plt.ylabel(label_dict[plotdata_labels[yy]])
        plt.xlim(-0.9, 0.5)
        y0 = np.median(plotdata[:, yy])
        plt.ylim(y0 - 0.3, y0 + 0.4)
        hogg_savefig(plotfn)
        for xx in alpha_indexes:
            if xx == yy:
                continue
            plotfn = dir + "/b" + plotdata_labels[yy].replace(" ", "") + "vs" + plotdata_labels[xx].replace(" ", "") + ".png"
            plt.clf()
            plt.plot(plotdata[:, xx], plotdata[:, yy], "k.", ms=0.5, alpha=0.50)
            plt.xlabel(label_dict[plotdata_labels[xx]])
            plt.ylabel(label_dict[plotdata_labels[yy]])
            x0 = np.median(plotdata[:, xx])
            plt.xlim(x0 - 0.6, x0 + 0.6)
            y0 = np.median(plotdata[:, yy])
            plt.ylim(y0 - 0.6, y0 + 0.6)
            hogg_savefig(plotfn)

    # compute shit for slicing
    Rs = (np.sqrt(plotmetadata[:, metadata_labels == "GX"].astype(float) ** 2 +
                  plotmetadata[:, metadata_labels == "GY"].astype(float) ** 2 +
                  plotmetadata[:, metadata_labels == "GZ"].astype(float) ** 2)).flatten()
    Zs = plotmetadata[:, metadata_labels == "GZ"].astype(float).flatten()
    Cs = plotdata[:, plotdata_labels == "CA_H - FE_H"].astype(float).flatten()
    Fs = plotdata[:, plotdata_labels == "FE_H"].astype(float).flatten()
    Ts = plotmetadata[:, metadata_labels == "TEFF_ASPCAP"].astype(float).flatten()

    # cut in vertical direction
    zcut = np.abs(metadata[:, metadata_labels == "GZ"].astype(float).flatten()) < 0.3 # kpc

    # cut in cylinder (around Sun)
    cylcut = (np.sqrt((plotmetadata[:, metadata_labels == "GX"].astype(float) - Xsun) ** 2 +
                      plotmetadata[:, metadata_labels == "GY"].astype(float) ** 2)).flatten() < 3.0 # kpc

    # compute shit
    Rmedians, Rdatamedians, Rsigmas, Rrmses = stats_in_slices(plotdata[zcut], Rs[zcut])
    Zmedians, Zdatamedians, Zsigmas, Zrmses = stats_in_slices(plotdata[cylcut], Zs[cylcut])
    Cmedians, Cdatamedians, Csigmas, Crmses = stats_in_slices(plotdata, Cs)
    Fmedians, Fdatamedians, Fsigmas, Frmses = stats_in_slices(plotdata, Fs)
    Tmedians, Tdatamedians, Tsigmas, Trmses = stats_in_slices(plotdata, Ts)

    # plot slices
    for medians, datamedians, sigmas, rmses, infix, xlabel in \
            [(Rmedians, Rdatamedians, Rsigmas, Rrmses, "R", "Galactic radius R (kpc)"),
             (Zmedians, Zdatamedians, Zsigmas, Zrmses, "Z", "Galactic height Z (kpc)"),
             (Cmedians, Cdatamedians, Csigmas, Crmses, "C", "[Ca/Fe] (dex)"),
             (Fmedians, Fdatamedians, Fsigmas, Frmses, "F", "[Fe/H] (dex)"),
             (Tmedians, Tdatamedians, Tsigmas, Trmses, "T", "ASPCAP Teff (K)"),
             ]:
        for prefix,ds in [("FE_H", [0,]),
                          ("alpha_FE", [1, 2, 3, 4, 5])]:
            plotfn = dir + "/G" + infix + "_" + prefix.replace(" ", "") + ".png"
            plt.clf()
            for d in ds:
                plt.plot(medians, datamedians[:,d], "k-", alpha=0.5, lw=2)
                plt.plot(medians, datamedians[:,d], "k.", lw=3)
                thislabel = (label_dict[plotdata_labels[d]].split(" "))[0]
                plt.text(medians[0], datamedians[0,d], thislabel, ha="right", color="r", alpha=0.5)
                plt.text(medians[-1], datamedians[-1,d], thislabel, ha="left", color="r", alpha=0.5)
                for j in range(len(medians)):
                    plt.plot([medians[j], medians[j]],
                             [datamedians[j,d] - sigmas[j,d], datamedians[j,d] + sigmas[j,d]],
                             "k-", lw=3)
            plt.xlabel(xlabel)
            plt.ylabel(label_dict[prefix])
            hogg_savefig(plotfn)
