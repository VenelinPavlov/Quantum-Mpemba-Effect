"""Paths, the measures used to compare predictions with exact dynamics, and the typicality formulas.

Every script imports this module first. It finds the folder that holds the mpemba package (the top of the
repository or of the shared folder) and makes the package importable.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent                          # the folder with the scripts
ROOT = next(p for p in HERE.parents if (p/'mpemba').is_dir())   # the folder that holds mpemba/
sys.path.insert(0, str(ROOT))
RUNS = ROOT/'chain_runs'                                        # the chain runs (inputs)
if not RUNS.is_dir():                                           # their location in the full repository
    RUNS = ROOT/'results_chaotic_wavefunction/article_programme'
CACHE = ROOT/'cache'                                            # eigensystems and rebuilt Lambda (safe to delete)
DATA = HERE/'data'                                              # the data behind the figures
FIGURES = HERE.parent/'figures'                                 # the figure PDFs


# ---- measures

def relaxation_time(t, d):
    """The relaxation time t_i: the first time d(t) falls to d(0)/e, linearly interpolated (nan if never)."""
    k = np.flatnonzero(d <= d[0]/np.e)
    if not len(k):
        return np.nan
    k = k[0]
    return t[k-1]+(d[0]/np.e-d[k-1])*(t[k]-t[k-1])/(d[k]-d[k-1])


def first_crossing(t, a, b):
    """The first time curve a falls below curve b, linearly interpolated (nan if never)."""
    k = np.flatnonzero((a[:-1]-b[:-1] > 0) & (a[1:]-b[1:] <= 0))
    if not len(k):
        return np.nan
    k = k[0]
    da, db = a[k]-b[k], a[k+1]-b[k+1]
    return t[k]+da*(t[k+1]-t[k])/(da-db)


def relative_error(prediction, exact):
    """The relative error ||D_hat - D|| / ||D|| over all preparations and times."""
    return np.linalg.norm(prediction-exact)/np.linalg.norm(exact)


# ---- the typicality formulas, Eqs. (1) and (2)

def typicality(D, q, E, t):
    """Eq. (1): D_i(t) = D_i(0) |chi_i(t)|^2, with chi_i the characteristic function of the energy populations q_i."""
    return D[:, :1]*np.abs(q@np.exp(1j*np.outer(E, t)))**2


def windowed(D0, q, E, t, lo, hi):
    """Eq. (2): Eq. (1) for one state with its populations restricted to the window lo <= E < hi and renormalized."""
    m = (E >= lo) & (E < hi)
    return D0*np.abs((q[m]/q[m].sum())@np.exp(1j*np.outer(E[m], t)))**2


def best_window(E, q, D, t, step=0.25, min_weight=0.2):
    """The window [lo, hi) on a grid of the given step that makes Eq. (2) closest to the exact curve D(t) of one
    state, among windows that keep at least min_weight of its probability. This is a fit to the exact dynamics."""
    g = np.arange(np.floor(E.min()), np.ceil(E.max())+step, step)
    k = np.searchsorted(g, E, side='right')                          # bin k holds g[k-1] <= E < g[k]
    P = sp.csr_matrix((q, (k, np.arange(len(E)))), shape=(len(g)+1, len(E)))@np.exp(1j*np.outer(E, t))
    cum = np.cumsum(P, axis=0)
    best = (np.inf, None)
    for b in range(1, len(g)):
        S = cum[b]-cum[:b]                                          # every window [g[a], g[b]) with a < b
        w = S[:, 0].real                                            # its weight (t[0] = 0)
        ok = np.flatnonzero(w >= min_weight)
        if len(ok):
            err = np.linalg.norm(D[0]*np.abs(S[ok])**2/w[ok, None]**2-D, axis=1)
            j = int(np.argmin(err))
            if err[j] < best[0]:
                best = (err[j], (g[ok[j]], g[b]))
    return best[1]
