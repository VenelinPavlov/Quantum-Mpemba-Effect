"""Data for Fig. 5 (the long-range Ising chain) and Fig. S1(b), from scratch.

Builds H = sum_{i<j} X_i X_j / |i-j|^0.9 + Bz sum_i Z_i + Bx sum_{i=2}^{N-1} X_i + B1 X_1 + BN X_N, diagonalizes it,
and computes, for the coherent spin states
    |psi(phi)> = prod_j [cos(phi/2)|down>_j + sin(phi/2)|up>_j],   Z|down> = +|down>
(the convention of the typicality draft and its code: phi = 0 is the Z = +1 product state, QuTiP's basis(2,0)):
the exact distances, typicality, Eq. (1), Eq. (2) with best-fit windows, the two widths of Fig. 5(c), the averaged
propagator, Eq. (16), in the dressed basis, the LDOS widths and the level statistics.
    python ising_data.py            ->  data/fig_ising.npz
N = 14 needs about 5 GiB: 8 min the first time (the diagonalization, then cached in cache/ising), 3 min after.
"""
import argparse
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

from common import CACHE, DATA, best_window, relative_error, typicality, windowed
from mpemba.lorentzian import fit_histogram

BZ, BX, B1, BN, ALPHA = 0.2, 0.3, 0.1, 0.5, 0.9      # fields and interaction exponent
PHI = 2*np.pi*np.array([0.0, 0.035, 0.05])         # the three states of Fig. 5
SITE = 5                                            # observed spin, counting from 0 (the sixth spin)
T = np.linspace(0, 20, 401)
BIN = 0.1                                           # energy bins for the averaged envelopes


# ---- the model

def setup(N):
    """Basis indices, Z_i = +-1 of every basis state (bit 0 = Z = +1), x fields and couplings J_ij."""
    idx = np.arange(2**N)
    z = 1-2*((idx[:, None] >> (N-1-np.arange(N))) & 1)       # site 0 is the most significant bit (QuTiP order)
    bx = np.full(N, BX)
    bx[0], bx[-1] = B1, BN
    J = {(i, j): 1/(j-i)**ALPHA for i, j in combinations(range(N), 2)}
    return idx, z.astype(float), bx, J


def hamiltonian(N):
    idx, z, bx, J = setup(N)
    bit = lambda i: 1 << (N-1-i)
    rows, cols, vals = [idx], [idx], [BZ*z.sum(1)]
    for i in range(N):
        rows.append(idx); cols.append(idx ^ bit(i)); vals.append(np.full(len(idx), bx[i]))
    for (i, j), Jij in J.items():
        rows.append(idx); cols.append(idx ^ bit(i) ^ bit(j)); vals.append(np.full(len(idx), Jij))
    n = len(idx)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def coherent_state(N, phi):
    single = np.array([np.cos(phi/2), np.sin(phi/2)])      # (|down>, |up>) = (Z=+1, Z=-1)
    psi = np.array([1.])
    for _ in range(N):
        psi = np.kron(psi, single)
    return psi


def eigensystem(N, H):
    """Eigenvalues (ascending) and eigenvectors (columns) of H, cached in cache/ising."""
    path = CACHE/f'ising/ising_N{N}.npz'
    if path.exists():
        f = np.load(path)
        return f['E'], f['V']
    E, V = eigh(H.toarray(), overwrite_a=True, driver='evr', check_finite=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, E=E, V=V)
    return E, V


# ---- exact dynamics of the observed spin

def bloch(N, psi):
    """Bloch vector of spin SITE for states psi[:, k] (computational basis)."""
    b = 1 << (N-1-SITE)
    i0 = np.arange(2**N)[(np.arange(2**N) & b) == 0]
    ov = np.sum(psi[i0].conj()*psi[i0 ^ b], axis=0)
    return np.array([2*ov.real, 2*ov.imag, np.sum(np.abs(psi[i0])**2-np.abs(psi[i0 ^ b])**2, axis=0)])


def exact_bloch_vectors(N, E, V, C, t):
    """Bloch vectors r(t), shape (states, 3, times), for the states with eigenbasis amplitudes C (columns)."""
    n, k = len(E), C.shape[1]
    r = np.empty((k, 3, len(t)))
    for j in range(0, len(t), 16):
        ph = np.exp(-1j*np.outer(E, t[j:j+16]))
        A = (C[:, :, None]*ph[:, None, :]).reshape(n, -1)
        st = (V@np.ascontiguousarray(A.real)+1j*(V@np.ascontiguousarray(A.imag))).reshape(n, k, -1)   # contiguous: BLAS
        r[:, :, j:j+16] = np.moveaxis(bloch(N, st.reshape(n, -1)).reshape(3, k, -1), 0, 1)
    return r


def diagonal_ensemble(N, V, q):
    """Reduced diagonal ensemble r_inf = sum_mu q_mu r_mu of every state (rows of q), for a real H (y = 0)."""
    b = 1 << (N-1-SITE)
    i0 = np.arange(2**N)[(np.arange(2**N) & b) == 0]
    x_mu = 2*np.einsum('ij,ij->j', V[i0], V[i0 ^ b])
    z_mu = np.einsum('ij,ij->j', V[i0], V[i0])-np.einsum('ij,ij->j', V[i0 ^ b], V[i0 ^ b])
    return np.array([q@x_mu, np.zeros(len(q)), q@z_mu]).T


# ---- the averaged propagator in the dressed basis (SM Sec. S6)

def fwht(V, N):
    """Rotate rows of V to the X basis (Hadamard on every site), in place."""
    for c in range(0, V.shape[1], 1024):
        X = np.array(V[:, c:c+1024])
        for i in range(N):
            Y = X.reshape(2**i, 2, 2**(N-1-i), X.shape[1])
            a = Y[:, 0].copy()
            Y[:, 0] += Y[:, 1]
            Y[:, 1] = a-Y[:, 1]
        V[:, c:c+1024] = X*2**(-N/2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=14)
    N = ap.parse_args().N
    H = hamiltonian(N)
    psi = np.array([coherent_state(N, p) for p in PHI]).T
    E, V = eigensystem(N, H)
    n = len(E)

    # exact dynamics and the reduced diagonal ensemble
    C = V.T@psi                                         # amplitudes on the eigenstates
    q = (C**2).T                                        # energy populations
    r_inf = diagonal_ensemble(N, V, q)
    exact = .5*np.linalg.norm(exact_bloch_vectors(N, E, V, C, T)-r_inf[:, :, None], axis=1)

    # Eq. (1), and Eq. (2) with windows fitted on [0, 5]
    typ = typicality(exact, q, E, T)
    windows = np.array([windowed(exact[i, 0], q[i], E, T, *best_window(E, q[i], exact[i, :101], T[:101]))
                        for i in range(3)])

    # Fig. 5(c): the total energy spread, and the rate set by the spread of the local x field at SITE
    def energy_spread(phi):
        v = coherent_state(N, phi)
        h = H@v
        return np.sqrt(h@h-(v@h)**2)
    grid = np.linspace(0, 1, 721)
    S = sum(1/abs(SITE-j)**(2*ALPHA) for j in range(N) if j != SITE)

    # dressed unperturbed basis: Ising part (diagonal in the X basis) plus the observed spin's own Bz field,
    # which mixes the two X states of SITE in 2x2 blocks. Rows xp (x_SITE = +1) and xm (x_SITE = -1) of each block
    # become its upper and lower dressed state.
    _, xs, bx, J = setup(N)                             # x_i = +-1 in the X basis
    E0 = xs@bx+sum(Jij*xs[:, i]*xs[:, j] for (i, j), Jij in J.items())
    fwht(V, N)
    b = 1 << (N-1-SITE)
    xp = np.arange(n)[(np.arange(n) & b) == 0]
    xm = xp ^ b
    m, d = (E0[xp]+E0[xm])/2, (E0[xp]-E0[xm])/2
    th = np.arctan2(BZ, d)
    c, s = np.cos(th/2), np.sin(th/2)
    e = np.empty(n)                                     # dressed energies
    e[xp], e[xm] = m+np.sqrt(d*d+BZ*BZ), m-np.sqrt(d*d+BZ*BZ)
    for k in range(0, n, 2048):                         # rotate each 2x2 block, a column block at a time
        Vp, Vm = V[xp, k:k+2048], V[xm, k:k+2048]
        V[xp, k:k+2048], V[xm, k:k+2048] = c[:, None]*Vp+s[:, None]*Vm, -s[:, None]*Vp+c[:, None]*Vm
    xpsi = psi.copy()
    fwht(xpsi, N)
    dpsi = np.empty_like(xpsi)                          # the states in the dressed basis
    dpsi[xp], dpsi[xm] = c[:, None]*xpsi[xp]+s[:, None]*xpsi[xm], -s[:, None]*xpsi[xp]+c[:, None]*xpsi[xm]

    # survival amplitudes g_alpha(t) from the static wavefunctions, averaged over bins of dressed energy
    co, si = np.cos(np.outer(E, T)), np.sin(np.outer(E, T))
    g = np.empty((n, len(T)), complex)
    for k in range(0, n, 1024):
        W = V[k:k+1024]**2
        g[k:k+1024] = (W@co-1j*(W@si))*np.exp(1j*np.outer(e[k:k+1024], T))
    kb = np.unique(np.floor((e-e.min())/BIN).astype(int), return_inverse=True)[1]
    gbin = np.stack([np.bincount(kb, g[:, j].real)+1j*np.bincount(kb, g[:, j].imag) for j in range(len(T))], 1)
    g = (gbin/np.bincount(kb)[:, None])[kb]

    # Eq. (16): phi(t) = G(t) psi in the dressed basis; Bloch vector of SITE from the 2x2 blocks
    propagator = np.empty_like(exact)
    for i in range(3):
        phi = np.exp(-1j*np.outer(e, T))*g*dpsi[:, i, None]
        pa, pb = phi[xp], phi[xm]
        diff, cross = np.abs(pa)**2-np.abs(pb)**2, pa.conj()*pb
        rv = np.array([np.sum(np.cos(th)[:, None]*diff-2*np.sin(th)[:, None]*cross.real, 0),
                       np.sum(-2*cross.imag, 0),
                       np.sum(np.sin(th)[:, None]*diff+2*np.cos(th)[:, None]*cross.real, 0)])
        propagator[i] = .5*np.linalg.norm(rv-r_inf[i][:, None]*np.sum(np.abs(phi)**2, 0), axis=0)

    # Fig. 5(d): LDOS half-width of the dressed states, a Lorentzian fit per unit energy bin
    fit_edges = np.r_[-np.inf, np.linspace(-2, 2, 1001), np.inf]
    centers = np.arange(np.floor(e.min()), np.ceil(e.max())+.5, 1.0)
    kc = np.clip(np.round(e-centers[0]).astype(int), 0, len(centers)-1)
    nb = len(fit_edges)-1
    mass = np.zeros((len(centers), nb))
    for k in range(0, n, 512):
        di = np.clip(np.searchsorted(fit_edges, E[None, :]-e[k:k+512, None], side='right')-1, 0, nb-1)
        mass += np.bincount((kc[k:k+512, None]*nb+di).ravel(), (V[k:k+512]**2).ravel(),
                            minlength=len(centers)*nb).reshape(len(centers), nb)
    nk = np.bincount(kc, minlength=len(centers))
    gamma = np.full(len(centers), np.nan)
    for k in np.flatnonzero(nk >= 20):
        try:
            gamma[k] = fit_histogram(mass[k]/nk[k], fit_edges)[1]
        except RuntimeError:
            pass

    # Fig. S1(b): mean level-spacing ratio <r> per unit energy bin (bins with at least 50 levels)
    spacing = np.diff(E)
    ratio = np.minimum(spacing[:-1], spacing[1:])/np.maximum(spacing[:-1], spacing[1:])
    redges = np.arange(np.floor(E.min()), np.ceil(E.max())+1.0, 1.0)
    rk = np.digitize(E[1:-1], redges)-1
    cnt = np.bincount(rk, minlength=len(redges))
    r_by_bin = np.where(cnt >= 50, np.bincount(rk, ratio, minlength=len(redges))/np.maximum(cnt, 1), np.nan)

    pedges = np.arange(-10, 39.01, .5)
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA/'fig_ising.npz', t=T, phi_over_2pi=PHI/(2*np.pi), exact=exact, typicality=typ,
                        windows=windows, propagator=propagator, phi_grid=grid,
                        energy_spread_grid=np.array([energy_spread(2*np.pi*f) for f in grid]),
                        local_rate_grid=np.sqrt(2*S)*np.abs(np.cos(2*np.pi*grid)), population_edges=pedges,
                        population_hist=np.array([np.histogram(E, pedges, weights=q[i])[0] for i in range(3)]),
                        gamma_centers=centers, gamma=gamma, r_edges=redges, r_by_bin=r_by_bin)
    print(f'wrote {DATA/"fig_ising.npz"}; D(0) = {np.round(exact[:, 0], 3)}; relative errors over these three '
          f'states: typicality {100*relative_error(typ, exact):.0f}%, averaged propagator {100*relative_error(propagator, exact):.0f}%')


if __name__ == '__main__':
    main()
