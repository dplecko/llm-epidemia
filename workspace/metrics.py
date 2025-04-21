import numpy as np
import pandas as pd

def ks_w(x, y, w_x=None, w_y=None):
    x = np.asarray(x)
    y = np.asarray(y)
    w_x = np.ones_like(x) if w_x is None else np.asarray(w_x)
    w_y = np.ones_like(y) if w_y is None else np.asarray(w_y)

    df = pd.concat([
        pd.DataFrame({'val': x, 'w': w_x, 'grp': 'x'}),
        pd.DataFrame({'val': y, 'w': w_y, 'grp': 'y'})
    ])

    df.sort_values("val", inplace=True)
    df["wx_cum"] = (df["w"] * (df["grp"] == "x")).cumsum()
    df["wy_cum"] = (df["w"] * (df["grp"] == "y")).cumsum()

    total_wx = w_x.sum()
    total_wy = w_y.sum()

    df["cdf_x"] = df["wx_cum"] / total_wx
    df["cdf_y"] = df["wy_cum"] / total_wy
    stat = (df["cdf_x"] - df["cdf_y"]).abs().max()

    return stat

def tabulate_w(x, w=None, nbins=None):
    x = np.asarray(x, dtype=int)
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w)

    if nbins is None:
        nbins = x.max()

    out = np.zeros(nbins)
    for xi, wi in zip(x, w):
        if 1 <= xi <= nbins:
            out[xi - 1] += wi
    return out


def cat_to_distr(x, w, nbins):
    x = np.asarray(x)
    distr = tabulate_w(x + 1, w=w, nbins=nbins)
    return distr / distr.sum()
