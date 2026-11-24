"""Pauli-convention impurity chain, cached eigensystems, and small file helpers.

N includes the impurity. The bath is open, with ZZ and optional exchange
Jxy*(XX+YY)/2, mixed fields, and a left-boundary field. The interaction is
g*((X_imp X_link + Y_imp Y_link)/2 + eta*Z_imp Z_link).
Eigenvector columns use the eigenbasis of the uncoupled Hamiltonian.
"""
from pathlib import Path
import csv
import gc
import hashlib
import json
import math
import numpy as np
import scipy.linalg as la


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def save_array(path, a):
    path = Path(path)
    tmp = path.with_suffix('.npy.tmp')
    with tmp.open('wb') as f:
        np.save(f, a, allow_pickle=False)
    tmp.replace(path)


def memory_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    raise RuntimeError('Cannot read available memory')


def estimated_peak_gib(n):
    # Dense real evr: input H and output U, bath-operator construction, plus
    # moment histograms, Python/BLAS overhead and a conservative margin.
    d = 2**n
    return 2.5 * d*d*8 / 2**30 + 0.85


def bath_dense(p):
    L = p['N'] - 1
    d = 2**L
    states = np.arange(d, dtype=np.int64)
    z = np.array([1. - 2.*((states >> (L-1-i)) & 1) for i in range(L)])
    diagonal = p['J'] * np.sum(z[:-1]*z[1:], axis=0)
    diagonal += p['hz']*z.sum(0) + p['hedge']*z[0]
    h = np.zeros((d, d), dtype=float, order='F')
    h[states, states] = diagonal
    for i in range(L):
        h[states ^ (1 << (L-1-i)), states] += p['hx']
    # Optional bath exchange: Jxy*(X_j X_{j+1}+Y_j Y_{j+1})/2.
    # Existing Ising model and cache keys are unchanged when Jxy is absent/zero.
    if p.get('Jxy', 0.) != 0.:
        for i in range(L-1):
            left, right = 1 << (L-1-i), 1 << (L-2-i)
            ids = states[((states & left) > 0) != ((states & right) > 0)]
            h[ids ^ left ^ right, ids] += p['Jxy']
    return h


def bath_eigensystem(p, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    pars = {k: p[k] for k in ('N', 'J', 'hx', 'hz', 'hedge')}
    if p.get('Jxy', 0.) != 0.:
        pars['Jxy'] = p['Jxy']
    key = 'bath_' + digest(pars)
    ep, up, mp = [cache/(key+s) for s in ('_E.npy', '_U.npy', '.json')]
    if mp.exists():
        if json.loads(mp.read_text()) != pars:
            raise ValueError('Bath cache metadata mismatch')
        return np.load(ep), np.load(up, mmap_mode='r')
    print('Diagonalizing bath, dimension', 2**(p['N']-1), flush=True)
    h = bath_dense(p)
    e, u = la.eigh(h, overwrite_a=True, check_finite=False, driver='evr')
    del h
    save_array(ep, e)
    save_array(up, u)
    atomic_json(mp, pars)
    del u
    gc.collect()
    return e, np.load(up, mmap_mode='r')


def bath_link_operators(ub, link, block):
    d = len(ub)
    L = int(round(math.log2(d)))
    states = np.arange(d)
    bit = 1 << (L-1-link)
    z = 1. - 2.*((states & bit) > 0)
    Ze = np.empty((d, d), order='F')
    Le = np.empty_like(Ze, order='F')
    # L=|down><up|, so H upper-right block is g L in the bath eigenbasis.
    down = states[(states & bit) > 0]
    up = down ^ bit
    for j in range(0, d, block):
        sl = slice(j, min(j+block, d))
        Ze[:, sl] = ub.T @ (z[:, None]*ub[:, sl])
        tmp = np.zeros((d, sl.stop-sl.start))
        tmp[down] = ub[up, sl]
        Le[:, sl] = ub.T @ tmp
    return Ze, Le


def full_eigensystem(p, cache, block):
    """Load or diagonalize the real Hamiltonian; return E, U, bath E, and H0 E."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    key = 'chain_' + digest(p)
    ep, up, mp = [cache/(key+s) for s in ('_E.npy', '_U.npy', '.json')]
    eb, ub = bath_eigensystem(p, cache)
    e0 = np.concatenate((eb+p['hs'], eb-p['hs']))
    if mp.exists():
        if json.loads(mp.read_text()) != p:
            raise ValueError('Chain cache metadata mismatch')
        del ub
        return np.load(ep), np.load(up, mmap_mode='r'), eb, e0
    print('Constructing impurity Hamiltonian in H0 basis', flush=True)
    Ze, Le = bath_link_operators(ub, p['link'], block)
    del ub
    b = len(eb)
    d = 2*b
    h = np.empty((d, d), dtype=float, order='F')
    h[:b, :b] = p['g']*p['eta']*Ze
    h[b:, b:] = -p['g']*p['eta']*Ze
    h[:b, b:] = p['g']*Le
    h[b:, :b] = p['g']*Le.T
    h[np.diag_indices(d)] += e0
    del Ze, Le
    gc.collect()
    print('Diagonalizing full Hamiltonian, dimension', d, flush=True)
    e, u = la.eigh(h, overwrite_a=True, check_finite=False, driver='evr')
    del h
    save_array(ep, e)
    save_array(up, u)
    atomic_json(mp, p)
    del u
    gc.collect()
    return e, np.load(up, mmap_mode='r'), eb, e0


def spin_columns(u, sl):
    """Real arrays x,Y,z such that the physical y operator is i*Y."""
    b = len(u)//2
    ab = u[:b].T @ u[b:, sl]
    ba = u[b:].T @ u[:b, sl]
    zz = 2*(u[:b].T @ u[:b, sl])
    zz[np.arange(sl.start, sl.stop), np.arange(sl.stop-sl.start)] -= 1.
    return np.array([ab+ba, ba-ab, zz])
