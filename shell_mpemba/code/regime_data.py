"""Data for Fig. 6, the two limits of the theory.

(a) Narrow preparations and typicality. The chain of Fig. 3 at N = 13 with bath filters of width s = 0.1, 0.25 and
    0.75 (runs selected_N13_width*; the same model, so the same general Lambda), and s = 0.02, for which the exact
    dynamics and all predictions are computed here from the cached eigensystem and that Lambda. At s = 0.02 the fully
    polarized states (mu = 1) are added. For each of the 68 preparations: the spread sigma_0 of the occupied
    unperturbed energies, the energy spread sigma_E, the LDOS width 2 Gamma at its mean energy, and the
    relaxation-time ratios of typicality, Eq. (1), detuned typicality, Eq. (10), and energy-shell relaxation, Eq. (12).
(b) A featureless bath with a spin-preserving share p of the coupling (SM, Sec. S5): the random-matrix pipeline of
    mpemba.reference, with in-sector coupling blocks carrying a share p of each row's variance at fixed total escape
    rate. For each p, the Z relaxation kernel of 24 test realizations against the pair-resolved kernel from a
    Lambda trained on 24 others (matrices of size 1024; cached in cache/reference_sweep).
    python regime_data.py           ->  data/fig_regimes.npz        (about 12 min, 5 GiB)
"""
import numpy as np
from scipy.linalg import eigh

from common import CACHE, DATA, RUNS, relative_error, relaxation_time, typicality
import chain_theory as ct
from mpemba import reference as ref
from mpemba.lorentzian import preparations
from mpemba.model import full_eigensystem
from mpemba.wavefunctions import fourier_moments, pair_histogram

WIDTHS = {0.1: 'selected_N13_width0.1', 0.25: 'selected_N13_width0.25', 0.75: 'selected_N13_width0.75'}
NARROW = 0.02                                        # computed directly, in narrow_direct
SHARES = [0., 0.1, 0.25, 0.5, 0.75, 0.9]
BASE = RUNS/WIDTHS[0.75]                             # the run whose Lambda, pair-weight bins and LDOS widths are used


def narrow_preparations():
    """Panel (a): one row per preparation."""
    cfg, numerics = ct.run_config(BASE)
    E, U, Eb, E0 = full_eigensystem(cfg['model'], CACHE/'spin_chain', numerics['block_columns'])
    L = ct.general_lambda(BASE, E, E0)               # the same model, so the same Lambda for all widths
    assert L.shape == (len(E), len(E))
    f = np.load(BASE/'exact/forecast.npz')            # LDOS half-width Gamma per energy bin
    centers, ok = (f['energy_edges'][:-1]+f['energy_edges'][1:])/2, f['gamma'] > 0
    out = {k: [] for k in ('width', 'polarization', 'sigma_0', 'sigma_E', 'two_gamma', 'r_typicality', 'r_detuned',
                           'r_shell', 'r_pair')}

    def add(width, states, v, q, t, D, curves):
        """Append one row per preparation; curves maps 'typicality', 'detuned', 'shell', 'pair' to D_hat(t)."""
        mean, spread = q@E, np.sqrt(q@E**2-(q@E)**2)
        t_exact = np.array([relaxation_time(t, d) for d in D])
        out['width'] += [width]*len(D)
        out['polarization'] += list(states[:, 1])
        out['sigma_0'] += list(np.sqrt(v@E0**2-(v@E0)**2))
        out['sigma_E'] += list(spread)
        out['two_gamma'] += list(2*np.interp(mean, centers[ok], f['gamma'][ok]))
        for key, Y in curves.items():
            out[f'r_{key}'] += list(np.array([relaxation_time(t, y) for y in Y])/t_exact)
        print(f's={width}: sigma_E {spread.min():.2f}-{spread.max():.2f}; relative errors '
              + ', '.join(f'{key} {100*relative_error(Y, D):.1f}%' for key, Y in curves.items())
              + '; median relaxation-time ratios typicality / detuned / energy-shell '
              + ' / '.join(f'{np.median(out[f"r_{key}"][-len(D):]):.2f}' for key in ('typicality', 'detuned', 'shell')),
              flush=True)

    for s, name in WIDTHS.items():                    # stored runs
        case = RUNS/name
        c, _ = ct.run_config(case)
        assert c['model'] == cfg['model'] and c['bath_width'] == s
        basis, mixing, states = preparations(Eb, c['bath_centers'], s, c['polarizations'])
        v = mixing@basis
        q = ct.energy_populations(v, U)
        t = np.load(case/'exact/forecast.npz')['times']
        D = ct.exact_distances(case)
        stored = np.load(case/'wavefunction/predictions.npz')
        assert np.allclose(stored['exact'], D)
        add(s, states, v, q, t, D, dict(typicality=typicality(D, q, E, t), detuned=ct.detuned_typicality(D, v, L, E, E0, t),
                                        shell=stored['general_factorized_raw'], pair=stored['general_full_raw']))

    d = narrow_direct(cfg, E, U, Eb, E0, L, NARROW, cfg['polarizations']+[1.0])     # s = 0.02, computed here
    add(NARROW, d['states'], d['v'], d['q'], d['t'], d['D'], {k: d[k] for k in ('typicality', 'detuned', 'shell', 'pair')})
    return {k: np.array(v) for k, v in out.items()}


def narrow_direct(cfg, E, U, Eb, E0, L, s, polarizations):
    """Exact dynamics and all predictions for bath filters of width s, from the eigensystem and Lambda."""
    n, half = len(E), len(E)//2
    basis, mixing, states = preparations(Eb, cfg['bath_centers'], s, polarizations)
    z = np.load(BASE/'wavefunction/contraction_general.npz')
    t, edges, omega, K = z['times'], z['edges'], z['omega'], z['K']
    dw = omega[1]-omega[0]
    wedge = np.r_[omega-dw/2, omega[-1]+dw/2]

    # exact dynamics: the amplitudes U diag(e^{-iEt}) U^T e_alpha of every populated unperturbed state alpha
    zmu = np.einsum('ij,ij->j', U[:half], U[:half])-np.einsum('ij,ij->j', U[half:], U[half:])
    xmu = 2*np.einsum('ij,ij->j', U[:half], U[half:])
    co, si = np.cos(np.outer(E, t)), np.sin(np.outer(E, t))
    bloch = np.zeros((len(basis), 3, len(t)))
    for b, w in enumerate(basis):
        A = np.flatnonzero(w > 1e-12*w.max())
        for a0 in range(0, len(A), 20):
            al = A[a0:a0+20]
            Ua = np.asarray(U[al])                                  # c_mu(alpha), rows alpha
            re = (U@np.ascontiguousarray((Ua[:, :, None]*co[None]).transpose(1, 0, 2).reshape(n, -1))).reshape(n, len(al), -1)
            im = -(U@np.ascontiguousarray((Ua[:, :, None]*si[None]).transpose(1, 0, 2).reshape(n, -1))).reshape(n, len(al), -1)
            up, dn = re[:half]+1j*im[:half], re[half:]+1j*im[half:]
            cross = np.sum(up.conj()*dn, 0)                        # (alpha, t)
            comp = np.array([2*cross.real, 2*cross.imag, np.sum(np.abs(up)**2-np.abs(dn)**2, 0)])
            bloch[b] += np.einsum('a,cat->ct', w[al], comp)
    del co, si
    v = mixing@basis
    q = ct.energy_populations(v, U)
    r = (mixing@bloch.reshape(len(basis), -1)).reshape(len(v), 3, len(t))
    r_inf = np.array([q@xmu, np.zeros(len(v)), q@zmu]).T                   # reduced diagonal ensemble
    D = .5*np.linalg.norm(r-r_inf[:, :, None], axis=1)

    # pair weights and their binned cosine sums, exactly as in mpemba.wavefunctions.contract
    Wu, Wd = np.asarray(L[:half]).T@np.asarray(L[:half]), np.asarray(L[half:]).T@np.asarray(L[half:])
    den = Wu+Wd
    f = np.divide(Wu, den, out=np.zeros_like(Wu), where=den > 0)
    del Wu, Wd, den
    full, Bb = np.zeros((len(basis), len(t))), np.zeros((len(basis), len(edges)-1))
    for b, w in enumerate(basis):
        A = np.flatnonzero(w > 1e-12*w.max())
        LA = np.asarray(L[A])
        P = LA.T@(w[A, None]*LA)
        C = 2*P*(1-f) if A[0] < half else -2*P*f
        ch = np.zeros((3, len(edges)-1, len(omega)))
        for j in range(0, n, 1024):
            cols = np.arange(j, min(n, j+1024))
            ch += pair_histogram(E, np.ascontiguousarray(C[:, cols]), cols, edges, wedge)[0]
        full[b], Bb[b] = fourier_moments(ch.sum(1), omega, t), ch[0].sum(1)      # summed over shells, as in contract()
    return dict(t=t, D=D, q=q, v=v, states=states, typicality=typicality(D, q, E, t),
                detuned=ct.detuned_typicality(D, v, L, E, E0, t),
                pair=.5*np.abs(mixing@full), shell=.5*np.abs(mixing@(Bb@K)))


def sample_with_share(n, kind, rng):
    """Random coupling with a share p of each row's variance inside its own spin sector (kind = 'p_<p>')."""
    p = float(kind.split('_')[1])
    h = n//2
    eb = np.linspace(-ref.CFG['span']/2, ref.CFG['span']/2, h)
    E0 = np.r_[eb+ref.CFG['hs'], eb-ref.CFG['hs']]
    var = 2*ref.CFG['g']**2/n                        # per-element variance of the reference flip-only coupling
    H = np.zeros((n, n))
    R = rng.normal(size=(h, h))*np.sqrt(var*(1-p))
    H[:h, h:], H[h:, :h] = R, R.T
    for sl in (slice(0, h), slice(h, n)):
        A = rng.normal(size=(h, h))
        H[sl, sl] = (A+A.T)/np.sqrt(2)*np.sqrt(var*p)
    H[np.diag_indices(n)] += E0
    return eigh(H, overwrite_a=True, check_finite=False, driver='evd')


def featureless_bath():
    """Panel (b): mpemba.reference with its coupling sampler replaced by sample_with_share."""
    ref.sample = sample_with_share
    ref.OUT = CACHE/'reference_sweep'
    ref.OUT.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ('share', 't_exact', 't_pair', 't_exact_sem', 'kernel_error')}
    curves = {}
    for p in SHARES:
        d = ref.run(1024, f'p_{p:g}')
        t = d['times']
        individual = [relaxation_time(t, k) for k in d['individual_test_K']]
        rows['share'].append(p)
        rows['t_exact'].append(relaxation_time(t, d['exact_K']))
        rows['t_pair'].append(relaxation_time(t, d['predicted_K']))
        rows['t_exact_sem'].append(np.nanstd(individual, ddof=1)/np.sqrt(len(individual)))
        rows['kernel_error'].append(float(d['relative_kernel_error']))
        curves[f'K_exact_{p:g}'], curves[f'K_pair_{p:g}'] = d['exact_K'], d['predicted_K']
        print(f'p={p}: relaxation time exact {rows["t_exact"][-1]:.2f}, pair-resolved {rows["t_pair"][-1]:.2f}, '
              f'ratio {rows["t_exact"][-1]/rows["t_pair"][-1]:.3f} (1/[2(1-p)] = {1/(2*(1-p)):.3f}), '
              f'kernel error {rows["kernel_error"][-1]:.3f}', flush=True)
    return {k: np.array(v) for k, v in rows.items()}, dict(t_reference=t, **curves)


if __name__ == '__main__':
    a = narrow_preparations()
    b, kernels = featureless_bath()
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA/'fig_regimes.npz', **a, **b, **kernels)
    print(f'wrote {DATA/"fig_regimes.npz"}')
