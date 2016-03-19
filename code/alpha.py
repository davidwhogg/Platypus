"""
This project is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

# alpha.py

## purpose
- Show that there is more than one kind of alpha element!

## bugs / notes
- All math very unstable to underflow; fix this; see HACK.
- Need to regularize the yield vectors to be near Solar.
- Need to regularize the amplitudes to be sparse?
- Plots are just TERRIBLE.
- Paranoid temperature cuts!
- Uses Jason Sanders's distances and Andy Casey's element abundances.
  Both of these are out-of-date and should be updated regularly.

"""
import os
import pickle as cp
import numpy as np 
import scipy.optimize as op
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

def tento(x):
    result = 10. ** x
    return result + 1.e-14 * np.max(result) # HACK HACK

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
        self.log10amps = np.zeros((self.N, self.K))
        self.log10vecs = np.zeros((self.K, self.D))
        print("initialized with ", N, D, K)

    def set_amps(self):
        """
        Needs obj() to return gradient too!
        """
        thislog10amps = np.zeros(self.K)
        offset2 = np.max(self.log10vecs)
        vecs = tento(self.log10vecs - offset2)
        for n in range(self.N):
            thislog10amps[:] = self.log10amps[n]
            data = self.data[n]
            ivars = self.ivars[n]
            def obj(log10amps):
                offset1 = np.max(log10amps)
                foo = tento(log10amps - offset1)
                denominator = np.dot(foo, vecs)
                derivs = (foo[:,None] * vecs).T / denominator[:,None]
                resids = data - (np.log10(denominator) + offset1 + offset2)
                return np.sum(resids * ivars * resids), -2. * np.sum(resids[:,None] * ivars[:,None] * derivs, axis=0)
            """ # testing
            obj1, deriv = obj(thislog10amps)
            tiny = 1.e-5
            thislog10amps2 = thislog10amps.copy()
            thislog10amps2[0] += tiny
            obj2, foo = obj(thislog10amps2)
            print(obj1, deriv, (obj2 - obj1) / tiny)
            assert False
            """
            thisresult = op.minimize(obj, thislog10amps, method="BFGS", jac=True)
            self.log10amps[n] = thisresult["x"]

    def set_vecs(self):
        """
        - Bugs in obj() gradient, I think.
        - divide in obj() unstable.
        """
        thislog10vecs = np.zeros(self.K)
        offset1 = np.max(self.log10amps)
        amps = tento(self.log10amps - offset1)
        for d in range(self.D):
            thislog10vecs[:] = self.log10vecs[:,d]
            data = self.data[:,d]
            ivars = self.ivars[:,d]
            def obj(log10vecs):
                offset2 = np.max(log10vecs)
                foo = tento(log10vecs - offset2)
                denominator = np.dot(amps, foo)
                derivs = (amps * foo[None,:]) / denominator[:,None]
                resids = data - (np.log10(denominator) + offset1 + offset2)
                return np.sum(resids * ivars * resids), -2. * np.sum(resids[:,None] * ivars[:,None] * derivs, axis=0)
            """ # testing
            obj1, deriv = obj(thislog10vecs)
            tiny = 1.e-5
            thislog10vecs2 = thislog10vecs.copy()
            thislog10vecs2[0] += tiny
            obj2, foo = obj(thislog10vecs2)
            print(obj1, deriv, (obj2 - obj1) / tiny)
            assert False
            """
            thisresult = op.minimize(obj, thislog10vecs, method="BFGS", jac="True")
            self.log10vecs[:,d] = thisresult["x"]

    def optimize(self, tol=0.1):
        prevchisq = np.Inf
        for t in range(512):
            print(t)
            self.set_amps()
            print(t, self.chisq())
            self.set_vecs()
            chisq = self.chisq()
            print(t, chisq)
            if chisq > prevchisq:
                print(t, "optimize(): The sky is falling", prevchisq, chisq)
                assert False
            if chisq > (prevchisq - tol):
                print(t, "optimize(): okay, good enough")
                break
            prevchisq = chisq
        print(self.log10vecs)

    def predicted_data(self):
        offset1 = np.max(self.log10amps)
        offset2 = np.max(self.log10vecs)
        return np.log10(np.dot(tento(self.log10amps - offset1),
                               tento(self.log10vecs - offset2))) \
                               + offset1 + offset2

    def chisq(self):
        resids = self.data - self.predicted_data()
        return np.sum(resids * self.ivars * resids)

if __name__ == "__main__":
    import pylab as plt
    np.random.seed(42)

    print("Hello World!")
    dir = "./alpha_figs"
    if not os.path.exists(dir):
        os.mkdir(dir)

    # read data
    dfn = "./data/cannon-distances.fits"
    pfn = "./data/alpha.pkl"
    yfn = "./data/alpha_model.pkl"
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
    ivars[:,1:] = 1. / 0.1 ** 2

    # build and optimize model
    try:
        print("attempting to read pickle", yfn)
        model = read_pickle_file(yfn)
        print(model)
        K = model.K
    except:
        K = 2
        bestlog10vecs = None
        for Nfit in 2. ** np.arange(8, 13):
            fitsubsample = np.random.randint(N, size=Nfit)
            model = abundance_model(plotdata[fitsubsample, 1:], ivars[fitsubsample, 1:], K) # drop [Fe/H] from fitting.
            if bestlog10vecs is None:
                model.log10vecs = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],
                                            [0.0, 0.3, 0.0, 0.0, 0.0]])
            else:
                model.log10vecs = bestlog10vecs
            model.optimize()
            bestlog10vecs = model.log10vecs.copy()
        model = abundance_model(plotdata[:, 1:], ivars[:, 1:], K) # drop [Fe/H] from fitting.
        model.log10vecs = bestlog10vecs # initialize
        model.optimize()
        pickle_to_file(yfn, model)

    # make predicted data
    predicteddata = plotdata.copy()
    predicteddata[:, 1:] = model.predicted_data()

    # get ready to plot
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
        plotfn = dir + "/a" + plotdata_labels[yy].replace(" ", "") + "_{:1d}.png".format(K)
        plt.figure(figsize=(6, 12))
        plt.clf()
        plt.subplot(311)
        plt.plot(plotdata[:, 0], plotdata[:, yy], "k.", ms=1.0)
        plt.xlim(-0.9, 0.5)
        xlim = plt.xlim()
        y0 = np.median(plotdata[:, yy])
        plt.ylim(y0 - 0.3, y0 + 0.4)
        ylim = plt.ylim()
        plt.ylabel(label_dict[plotdata_labels[yy]])
        plt.title("K = {:1d}".format(model.K))
        plt.text(0.02, 0.01, "data", transform=plt.gca().transAxes)
        plt.subplot(312)
        plt.plot(predicteddata[:, 0], predicteddata[:, yy], "k.", ms=1.0)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.ylabel(label_dict[plotdata_labels[yy]])
        plt.text(0.02, 0.01, "model", transform=plt.gca().transAxes)
        plt.subplot(313)
        plt.plot(plotdata[:, 0], plotdata[:, yy] - predicteddata[:, yy], "k.", ms=1.0)
        plt.xlim(xlim)
        plt.ylim(ylim - np.mean(ylim))
        plt.xlabel(label_dict[plotdata_labels[0]])
        plt.ylabel("delta " + label_dict[plotdata_labels[yy]])
        plt.text(0.02, 0.01, "residuals", transform=plt.gca().transAxes)
        hogg_savefig(plotfn)

        if False:
        # for xx in alpha_indexes:
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

    assert False

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
