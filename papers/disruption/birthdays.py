"""
This file is part of the Platypus project.
Copyright 2016 David W. Hogg (NYU).

# Purpose
- Develop the analog of the birthday paradox for constraining the
  formation of the Milky Way halo.
- Rule out stupid models of the Milky Way halo.

## Issues
- Not yet written
"""

import numpy as np
import pylab as plt

def _make_one_tag(fsmooth, Neff):
    if np.random.uniform() < fsmooth:
        return 0
    return np.random.randint(1, Neff + 1)


def make_tags(N, fsmooth, Neff, verbose=False):
    """
    Tag stars with tags 0...`Neff`.

    # inputs:
    - `N`: number of stars to tag.
    - `fsmooth`: fraction of stars to tag with a zero.
    - `Neff`: number of non-zero tags.

    # outputs:
    - `np.ndarray` of tags
    """
    assert fsmooth >= 0.
    assert fsmooth <= 1.
    if verbose:
        print("Making N={} tags for fsmooth={}, Neff={}".format(N, fsmooth, Neff))
    return np.array([_make_one_tag(fsmooth, Neff) for n in range(N)])

def coincidence(tags, verbose=False):
    nztags = np.sort(tags[tags > 0])
    bools = nztags[1:] == nztags[:-1]
    if verbose:
        if np.any(bools):
            print("YES: shared non-zero tags: ", (nztags[1:])[bools])
        else:
            print("NO")
    return np.sum(bools)

def plot_one_case(Nstr, fsmoothstr, Neffstr):
    fsmooth = float(fsmoothstr)
    N = int(Nstr)
    Neff = int(Neffstr)
    ns = np.zeros(10000).astype("int")
    for i in range(len(ns)):
        tags = make_tags(N, fsmooth, Neff)
        ns[i] = coincidence(tags)
    print(ns)
    plt.clf()
    plt.hist(ns, bins=np.arange(np.max(ns))-0.5, facecolor="none")
    plt.title("N = {} / fsmooth = {} / Neff = {}".format(Nstr, fsmoothstr, Neffstr))
    foo, xt = plt.xlim()
    foo, yt = plt.ylim()
    plt.text(xt-0.1, yt-0.1, "zero fraction: {:.4f}".format(np.sum(ns == 0) / len(ns)), ha="right", va="top")
    prefix = "histogram_{}_{}_{}".format(Nstr, fsmoothstr, Neffstr)
    plt.savefig(prefix + ".png")

if __name__ == "__main__":
    np.random.seed(42)
    fsmoothstr = "0.7"
    Neffstr = "10000"
    for Nstr in ["500", "1000", "1500"]:
        plot_one_case(Nstr, fsmoothstr, Neffstr)
