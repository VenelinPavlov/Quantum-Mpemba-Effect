"""Compare the typicality pipeline of github.com/VenelinPavlov/Quantum-Mpemba-Effect (commit f469c3a) with the
paper's, on the N = 14 long-range Ising chain. Reproduces every number in CONSISTENCY_NOTE.pdf
and runs the code it suggests (item 5); items 6 and 7 are the notes on Fig. 5a and Fig. S1(b).

    python compare_pipelines.py [--their-spectrum path/to/H_spectrum_14.npz]

The two pipelines differ in two choices, both applied here to the same exact dynamics:
  reference state   theirs: rho_A averaged over t in [0, 5] (101 samples, "QME - numerical - using_data.py")
                    paper:  the reduced diagonal ensemble, r_inf = sum_mu (rho)_mu,mu r_mu (the t -> infinity average)
  energy window     theirs: E in [0, 11] ("QME_analytical_only_Ising.py")
                    paper (Fig. 5): for each state, the window that best fits the exact curve (common.best_window)
The eigensystem is the one built by ising_data.py (cached in cache/ising). With --their-spectrum, the script also
checks that their eigenvalues are the same (only the small eigvalsH member of their file is read).
6 min and 5 GiB the first time (the diagonalization), 30 s after.
"""
import argparse
from pathlib import Path

import numpy as np

from common import best_window, first_crossing
import ising_data as idt

N = 14
T5 = np.linspace(0, 5, 101)                     # their time grid


def reversal(t, far, near):
    """Times at which the farther state is nearer: first and last such sample, and the deepest reversal."""
    neg = np.flatnonzero(far-near < 0)
    return (t[neg[0]], t[neg[-1]], float((far-near).min())) if len(neg) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--their-spectrum', type=Path, default=None)
    a = ap.parse_args()
    E, V = idt.eigensystem(N, idt.hamiltonian(N))
    if a.their_spectrum:
        with np.load(a.their_spectrum) as f:
            E_their = np.sort(np.asarray(f['eigvalsH'], float))
        print(f'1. Same Hamiltonian: max |E_paper - E_theirs| = {np.abs(np.sort(E)-E_their).max():.1e}\n')
    phi = idt.PHI
    psi = np.array([idt.coherent_state(N, p) for p in phi]).T
    C = V.T@psi                                     # amplitudes on the eigenstates
    q = (C**2).T                                    # energy populations, their rho_mumu
    r_de = idt.diagonal_ensemble(N, V, q)           # the paper's reference
    r = idt.exact_bloch_vectors(N, E, V, C, T5)
    r_avg = r.mean(2)                               # their reference
    D2 = {'paper (diagonal ensemble)': (.5*np.linalg.norm(r-r_de[:, :, None], axis=1))**2,
          'theirs (average over [0,5])': (.5*np.linalg.norm(r-r_avg[:, :, None], axis=1))**2}
    labels = [f'{p/2/np.pi:g}' for p in phi]

    print('2. Reference Bloch vector (x, y, z) of the observed spin')
    for i in range(3):
        print(f'   phi/2pi = {labels[i]:5s}  diagonal ensemble {np.round(r_de[i], 3)}   average over [0,5] {np.round(r_avg[i], 3)}')
    print('\n3. Consequences for the exact D^2(t)')
    for name, d in D2.items():
        tc = [first_crossing(T5, d[0], d[j]) for j in (1, 2)]
        print(f'   {name:28s} D^2(0) = {np.round(d[:, 0], 3)}; crossing of 0 with 0.035 / 0.05 at t = {tc[0]:.2f} / {tc[1]:.2f}')
        for j in (1, 2):
            rv = reversal(T5, d[0], d[j])
            span = f'from t = {rv[0]:.2f} to {rv[1]:.2f}, depth {-rv[2]:.4f}' if rv else 'none'
            print(f'      phi = 0 nearer than {labels[j]}: {span}')
    print('\n4. Windows (Eq. 2), applied to all three states, against the exact curves with each reference')
    m = (E >= 0) & (E <= 11)
    d = D2['paper (diagonal ensemble)']
    best = []
    for i in range(3):
        lo, hi = best_window(E, q[i], np.sqrt(d[i]), T5)
        mb = (E >= lo) & (E < hi)
        best.append(mb)
        print(f'   phi/2pi = {labels[i]:5s}  weight in [0, 11]: {q[i, m].sum():.2f};  best-fit window [{lo:.2f}, {hi:.2f}) keeps {q[i, mb].sum():.2f}')
    for name, d in D2.items():
        win = np.array([d[i, 0]*np.abs((q[i, m]/q[i, m].sum())@np.exp(-1j*np.outer(E[m], T5)))**4 for i in range(3)])
        for j in (1, 2):
            rv = reversal(T5, win[0], win[j])
            depth = f'deepest reversal {-rv[2]:.1e}, from t = {rv[0]:.2f} to {rv[1]:.2f}' if rv else 'no reversal'
            print(f'   window [0, 11], {name}: 0 vs {labels[j]}: {depth}')

    print('\n5. The replacement for rhoA_DE suggested in the note, run with the variable names of their code')
    for i in range(3):
        rhoA_DE = diagonal_ensemble_rhoA(V, q[i], k=idt.SITE)
        x, z = r_de[i, 0], r_de[i, 2]
        target = .5*np.array([[1+z, x], [x, 1-z]])
        print(f'   phi/2pi = {labels[i]:5s}  max |rhoA_DE - (1 + r_inf.sigma)/2| = {np.abs(rhoA_DE-target).max():.1e}')

    print("\n6. Typicality in the draft's Fig. 5 and in the paper's Fig. 5a")
    d_avg, d_de = D2['theirs (average over [0,5])'], D2['paper (diagonal ensemble)']
    sel = T5 <= 1

    def draft_window(d, lo, widths):
        """Eq. (2): the typicality formula restricted to [lo, lo + width], with D^2(0) from the exact curve d."""
        out = []
        for i in range(3):
            m = (E >= lo) & (E <= lo+widths[i])
            out.append(d[i, 0]*np.abs((q[i, m]/q[i, m].sum())@np.exp(-1j*np.outer(E[m], T5)))**4)
        return np.array(out)

    sig = np.sqrt(q@E**2-(q@E)**2)
    full = np.array([d_avg[i, 0]*np.abs(q[i]@np.exp(-1j*np.outer(E, T5)))**4 for i in range(3)])
    half = lambda d: [round(float(T5[np.argmax(x < x[0]/2)]), 2) for x in d]
    print(f'   full spectrum (paper Eq. 1): sigma_E = {np.round(sig, 2)}; D^2 halves at t = {half(full)} (exact {half(d_avg)});'
          f' first crossing of 0 with 0.05 at t = {first_crossing(T5, full[0], full[2]):.2f}')
    dW = (12.5, 10.25, 9.3)                                      # the draft's widths (its Fig. 6)
    errs, cr = [], []
    for lo in np.arange(-4, 1.01, .5):                           # where the window sits in the bulk
        w = draft_window(d_avg, lo, dW)
        errs.append(max(np.linalg.norm(w[i, sel]-d_avg[i, sel])/np.linalg.norm(d_avg[i, sel]) for i in range(3)))
        cr.append(first_crossing(T5, w[0], w[2]))
    print(f'   draft widths dW = {dW}, lower edge from -4 to 1: largest relative error on [0, 1] against their exact curves'
          f' {min(errs):.3f}-{max(errs):.3f}; crossing of 0 with 0.05 at t = {min(cr):.2f}-{max(cr):.2f}'
          f' (exact {first_crossing(T5, d_avg[0], d_avg[2]):.2f})')
    for width in dW:
        w = draft_window(d_avg, -2.5, (width,)*3)
        print(f'   one width {width:5.2f} for all three states (lower edge -2.5): crossing of 0 with 0.035 / 0.05 at'
              f' t = {first_crossing(T5, w[0], w[1]):.2f} / {first_crossing(T5, w[0], w[2]):.2f}')
    fitted = np.array([d_de[i, 0]*np.abs((q[i, best[i]]/q[i, best[i]].sum())@np.exp(-1j*np.outer(E[best[i]], T5)))**4
                       for i in range(3)])
    for name, d in (('draft widths, lower edge -2.5', draft_window(d_de, -2.5, dW)), ('fitted windows of item 4', fitted),
                    ('exact', d_de)):
        diff = d[0]-d[2]
        print(f'   on the scale of Fig. 5a (diagonal ensemble), D^2(0) - D^2(0.05), {name}: changes sign at'
              f' t = {first_crossing(T5, d[0], d[2]):.2f}, deepest {diff.min():.4f} at t = {T5[diff.argmin()]:.2f}')

    print('\n7. Where the GOE region of Fig. S1(b) sits in the spectrum')
    Es = np.sort(E)
    _, _, bx, J = idt.setup(N)
    print(f'   spectrum from {Es[0]:.2f} to {Es[-1]:.2f} (top: all spins along x, sum J_ij + sum B^x_i ='
          f' {sum(J.values()):.2f} + {bx.sum():.2f}); median {np.median(Es):.2f};'
          f' middle half of the levels in [{np.quantile(Es, .25):.2f}, {np.quantile(Es, .75):.2f}]')
    h, ed = np.histogram(Es, bins=np.arange(np.floor(Es[0]), np.ceil(Es[-1])+1))
    mid = (Es[0]+Es[-1])/2
    print(f'   density of states peaks in [{ed[h.argmax()]:g}, {ed[h.argmax()+1]:g}); the midpoint {mid:.1f} of the range'
          f' lies above {np.mean(Es < mid):.1%} of the levels; {np.mean(Es < -8):.1%} of the levels lie below -8')
    s = np.diff(Es)
    rr = np.minimum(s[:-1], s[1:])/np.maximum(s[:-1], s[1:])
    Em = Es[1:-1]
    for lo, hi in ((-np.inf, -8), (-8, 1), (1, 5), (5, np.inf)):
        m = (Em >= lo) & (Em < hi)
        print(f'   E in [{lo:4g}, {hi:4g}): {m.mean():5.1%} of the levels, <r> = {rr[m].mean():.3f}'
              f'  (GOE 0.531, Poisson 0.386)')


def diagonal_ensemble_rhoA(eigvecsH, rho_mumu, k=5):
    """The snippet of CONSISTENCY_NOTE.pdf, Sec. 2, verbatim (N and k as in their code): the reduced diagonal
    ensemble of spin k, in the convention of their reduced_density_matrix (rho_A = M M^dagger, QuTiP site order)."""
    # axes of psi: spins before k, spin k, spins after k, eigenstate mu
    psi = eigvecsH.reshape(2**k, 2, 2**(N-k-1), -1)
    rhoA_DE = np.empty((2, 2), complex)
    for s in range(2):
        for u in range(2):
            rhoA_DE[s, u] = np.einsum('aem,aem->m', psi[:, s], psi[:, u].conj()) @ rho_mumu
    return rhoA_DE


if __name__ == '__main__':
    main()
