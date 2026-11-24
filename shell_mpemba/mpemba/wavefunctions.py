"""Direct general-envelope chaotic-wavefunction test for diagonal spin states.

Lambda[alpha,m] = E|U[alpha,m]|^2 is a coefficient variance, not an LDOS.
For m != n the leading orthogonality-corrected contraction is
    delta_ab w_a - w_a w_b / sum(w),  w_a=Lambda[a,m]Lambda[a,n].
For Z=+1/-1 and a preparation supported in sector s this reduces to
    O_mn^2 = 4 W_up W_down/(W_up+W_down),
    C_mn = 2 s P_s W_opposite/(W_up+W_down).
This tests that ansatz; resolving Lambda by spin does NOT establish that
orthogonality alone captures the extra correlations of pure-flip V.
No kinetic rate formula is substituted into these contractions.
"""
from pathlib import Path
import json
import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .lorentzian import fit_histogram, line_masses
from .model import atomic_json, write_csv


def progress(out, stage, **extra):
    d = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra)
    atomic_json(Path(out)/'progress.json', d)
    print(json.dumps(d), flush=True)


def normalized(a):
    return a/max(float(np.sum(a)), 1e-300)


def calibration_ids(E0, spin, center, width, min_states):
    """Fixed global folds in each spin sector; no row changes fold across bins."""
    half = len(E0)//2
    local = np.arange(half)
    allids = local+spin*half
    ids = allids[abs(E0[allids]-center) < width/2]
    if len(ids) < min_states:
        ids = allids[np.argsort(abs(E0[allids]-center), kind='stable')[:min_states]]
        ids = np.sort(ids)
    return [ids[((ids-spin*half) % 3) == fold] for fold in range(3)]


def calibrate(E, U, E0, cfg, out):
    """Three interleaved bare-state folds: train, select smoothing, final test.

    Profiles use the full available detuning range. Lorentz fits include both
    infinite tail bins; general profiles retain every finite transition.
    """
    out = Path(out)
    path = out/'envelopes.npz'
    if path.exists():
        return dict(np.load(path))
    de = cfg['profile_energy_width']
    centers = np.arange(np.floor(E0.min()), np.ceil(E0.max())+.1*de, de)
    extent = max(abs(E.max()-E0.min()), abs(E.min()-E0.max()))*1.000001
    edges = np.linspace(-extent, extent, cfg['detuning_bins']+1)
    dx = edges[1]-edges[0]
    fit_edges = np.r_[-np.inf, np.linspace(-2, 2, 1001), np.inf]
    models = []
    lor = []
    gauss = []
    masses = []
    widths = []
    shifts = []
    counts = []
    cv = []
    ids_meta = []
    for spin in (0, 1):
        for j, center in enumerate(centers):
            folds = calibration_ids(E0, spin, center, de, cfg['min_profile_states'])
            mm = []
            small = []
            for ids in folds:
                delta = E[None, :]-E0[ids, None]
                weights = np.asarray(U[ids, :])**2
                mm.append(np.histogram(delta, edges, weights=weights)[0]/len(ids))
                small.append(np.histogram(delta, fit_edges, weights=weights)[0]/len(ids))
            assert np.allclose(np.sum(mm, axis=1), 1, atol=1e-10)
            shift, gamma, _ = fit_histogram(small[0], fit_edges)
            gshift, sigma, _ = fit_histogram(small[0], fit_edges, True)
            candidates = [normalized(gaussian_filter1d(mm[0], x, mode='constant'))
                          if x else mm[0].copy() for x in cfg['smoothing_bins']]
            # Integrated squared error, fixed across candidates. Validation
            # decides smoothing; final test never selects a model parameter.
            losses = [float(np.sum((v-mm[1])**2)) for v in candidates]
            best = int(np.argmin(losses))
            general = candidates[best]
            lm = normalized(line_masses(edges, shift, gamma))
            gm = normalized(line_masses(edges, gshift, sigma, True))
            models.append(general)
            lor.append(lm)
            gauss.append(gm)
            masses.append(mm)
            widths.append(gamma)
            shifts.append(shift)
            counts.append([len(x) for x in folds])
            ids_meta.append((spin, j))
            cv.append(dict(spin=spin, energy=float(center), train_states=len(folds[0]), validation_states=len(folds[1]), test_states=len(folds[2]),
                           gamma=gamma, shift=shift, smoothing_bins=cfg['smoothing_bins'][best],
                           TV_lorentzian_test=float(.5*abs(lm-mm[2]).sum()), TV_general_test=float(.5*abs(general-mm[2]).sum()),
                           TV_gaussian_test=float(.5*abs(gm-mm[2]).sum()), general_fit_uses_test=False))
            if j % 8 == 0:
                progress(out, 'calibrating_envelopes', spin=spin, center=float(center))
    shape = (2, len(centers), len(edges)-1)
    data = dict(centers=centers, detuning_edges=edges, train=np.array(masses)[:, 0].reshape(shape),
                selection=np.array(masses)[:, 1].reshape(shape), test=np.array(masses)[:, 2].reshape(shape),
                general=np.array(models).reshape(shape), lorentzian=np.array(lor).reshape(shape), gaussian=np.array(gauss).reshape(shape),
                gamma=np.array(widths).reshape(2, -1), shift=np.array(shifts).reshape(2, -1), counts=np.array(counts).reshape(2, -1, 3))
    np.savez_compressed(path, **data)
    write_csv(out/'envelope_validation.csv', cv)
    return data


def balance_variance(A, tolerance=1e-6, max_iterations=6000, callback=None):
    """Positive row/column rescaling; records rather than hides its size.

    A is modified in place. No phase, observable or trajectory information is
    used. Both stochastic sums are required by a square eigenbasis.
    """
    n, m = A.shape
    if n != m:
        raise ValueError('Square full eigenbasis required')
    before_row = A.sum(1)
    before_col = A.sum(0)
    if min(before_row.min(), before_col.min()) <= 0:
        raise ValueError('Empty variance row/column')
    left = np.ones(n)
    right = np.ones(n)
    for iteration in range(max_iterations):
        left = 1/(A@right)
        right = 1/(A.T@left)
        if iteration % 20 == 0 or iteration == max_iterations-1:
            err = float(np.max(abs(left*(A@right)-1)))
            if callback and iteration % 200 == 0:
                callback(iteration, err)
            if err < tolerance:
                break
    if err >= tolerance:
        raise RuntimeError(f'Variance normalization did not converge: {err}')
    for j in range(0, n, 128):
        A[j:j+128] *= left[j:j+128, None]*right[None, :]
    if hasattr(A, 'flush'):
        A.flush()
    return dict(iterations=iteration+1, row_error=float(abs(A.sum(1)-1).max()), column_error=float(abs(A.sum(0)-1).max()),
                initial_row_min=float(before_row.min()), initial_row_max=float(before_row.max()),
                initial_column_min=float(before_col.min()), initial_column_max=float(before_col.max()),
                left_min=float(left.min()), left_max=float(left.max()), right_min=float(right.min()), right_max=float(right.max()))


def construct_variance(E, E0, profiles, kind, cfg, out):
    out = Path(out)
    path = out/f'lambda_{kind}.npy'
    meta = out/f'lambda_{kind}.json'
    if path.exists() and meta.exists():
        return np.load(path, mmap_mode='r')
    n = len(E)
    half = n//2
    # Smooth DOS on a scale larger than individual spacings, independently of
    # the preparation and the selected envelope fit.
    dedge = np.arange(np.floor(E.min())-1, np.ceil(E.max())+1.25, .25)
    dos = gaussian_filter1d(np.histogram(E, dedge)[0].astype(float), 2, mode='nearest')/.25
    rho = np.interp(E, (dedge[:-1]+dedge[1:])/2, np.maximum(dos, 1e-12))
    A = np.lib.format.open_memmap(path, mode='w+', dtype='float64', shape=(n, n))
    centers = profiles['centers']
    edges = profiles['detuning_edges']
    x = (edges[:-1]+edges[1:])/2
    dx = x[1]-x[0]
    for alpha in range(n):
        spin = int(alpha >= half)
        eps = E0[alpha]
        delta = E-eps
        if kind == 'lorentzian':
            gamma = np.interp(eps, centers, profiles['gamma'][spin])
            shift = np.interp(eps, centers, profiles['shift'][spin])
            density = gamma/(np.pi*((delta-shift)**2+gamma*gamma))
        elif kind == 'general':
            j = int(np.clip(np.searchsorted(centers, eps)-1, 0, len(centers)-2))
            frac = float(np.clip((eps-centers[j])/(centers[j+1]-centers[j]), 0, 1))
            mass = (1-frac)*profiles[kind][spin, j]+frac*profiles[kind][spin, j+1]
            density = np.interp(delta, x, mass/dx, left=0, right=0)
        else:
            raise ValueError(kind)
        A[alpha] = density/rho
    A.flush()
    norm = balance_variance(A, cfg['balance_tolerance'], cfg['balance_max_iterations'],
                            lambda it, err: progress(out, 'normalizing_variance', model=kind, iteration=it, error=err))
    atomic_json(meta, dict(kind=kind, shape=list(A.shape), normalization=norm,
                           interpretation='Spin-resolved strength profile divided by smooth exact DOS, then positive row/column balancing.',
                           contraction='Leading orthogonality-corrected covariance; no added spin-flip vertex correlations.'))
    del A
    return np.load(path, mmap_mode='r')


def pair_histogram(E, weights, columns, edges, wedge):
    ne = len(edges)-1
    nw = len(wedge)-1
    wc = (wedge[:-1]+wedge[1:])/2
    keep = np.arange(len(E))[:, None] < columns[None, :]
    gap = (E[columns][None, :]-E[:, None])[keep]
    mean = ((E[columns][None, :]+E[:, None])/2)[keep]
    ei = np.clip(np.searchsorted(edges, mean, side='right')-1, 0, ne-1)
    wi = np.clip(np.searchsorted(wedge, gap, side='right')-1, 0, nw-1)
    ix = ei*nw+wi
    dw = gap-wc[wi]
    weight = 2*weights[keep]
    moments = np.array([np.bincount(ix, weights=weight*dw**k,
                       minlength=ne*nw).reshape(ne, nw) for k in range(3)])
    remainder = np.bincount(ei, weights=abs(weight)*abs(dw)**3, minlength=ne)/6
    return moments, remainder


def fourier_moments(hist, omega, times):
    shape = hist.shape[1:-1]+(len(times),)
    out = np.zeros(shape)
    for j in range(0, len(times), 32):
        sl = slice(j, min(j+32, len(times)))
        t = times[sl]
        co = np.cos(omega[:, None]*t)
        si = np.sin(omega[:, None]*t)
        out[..., sl] = hist[0]@co-(hist[1]@si)*t-(hist[2]@co)*t*t/2
    return out


def contract(E, L, basis, edges, times, cfg, out, kind):
    """Column-block Gram contractions: no d*d observable/density is allocated."""
    out = Path(out)
    done = out/f'contraction_{kind}.npz'
    if done.exists():
        return dict(np.load(done))
    n = len(E)
    half = n//2
    ne = len(edges)-1
    nw = cfg['frequency_bins']
    ns = len(basis)
    wedge = np.linspace(0, np.nextafter(np.ptp(E), np.inf), nw+1)
    omega = (wedge[:-1]+wedge[1:])/2
    oh = np.zeros((3, ne, nw))
    ch = np.zeros((ns, 3, ne, nw))
    ob = np.zeros(ne)
    cb = np.zeros((ns, ne))
    for j in range(0, n, cfg['block_columns']):
        cols = np.arange(j, min(n, j+cfg['block_columns']))
        # Contiguous columns avoid accidental slow BLAS striding.
        up = np.ascontiguousarray(L[:half, cols])
        down = np.ascontiguousarray(L[half:, cols])
        Wu = L[:half].T@up
        Wd = L[half:].T@down
        den = Wu+Wd
        fraction = np.divide(Wu, den, out=np.zeros_like(Wu), where=den > 0)
        op = 4*Wd*fraction
        hh, bb = pair_histogram(E, op, cols, edges, wedge)
        oh += hh
        ob += bb
        for s, p in enumerate(basis):
            spin = int(np.sum(p[half:]) > 0)
            if spin == 0:
                if np.any(p[half:]):
                    raise ValueError('Basis preparation must have one spin sector')
                P = L[:half].T@(p[:half, None]*up)
                C = 2*P*(1-fraction)
            else:
                if np.any(p[:half]):
                    raise ValueError('Basis preparation must have one spin sector')
                P = L[half:].T@(p[half:, None]*down)
                C = -2*P*fraction
            hh, bb = pair_histogram(E, C, cols, edges, wedge)
            ch[s] += hh
            cb[s] += bb
        progress(out, 'contracting', model=kind, columns_complete=int(cols[-1]+1), dimension=n)
    power = oh[0].sum(1)
    kt = fourier_moments(oh, omega, times)
    K = np.divide(kt, power[:, None], out=np.zeros_like(kt), where=power[:, None] > 0)
    Bbase = ch[:, 0].sum(2)
    fullbase = np.array([fourier_moments(x.sum(1), omega, times) for x in ch])
    data = dict(K=K, power=power, Bbase=Bbase, fullbase=fullbase, operator_spectrum=oh[0], state_spectrum=ch[:, 0], omega=omega,
                operator_bound=np.divide(ob*times[-1]**3, power,
                                         out=np.zeros_like(ob), where=power > 0),
                full_basis_bound=cb.sum(1)*times[-1]**3, edges=edges, times=times)
    np.savez_compressed(done, **data)
    return data


def assess_variance(E, E0, L, profiles, cfg):
    """Final test discrepancy after row/column normalization (no refitting)."""
    rows = []
    edges = profiles['detuning_edges']
    for spin in (0, 1):
        for j, center in enumerate(profiles['centers']):
            ids = calibration_ids(
                E0, spin, center, cfg['profile_energy_width'], cfg['min_profile_states'])[2]
            delta = E[None, :]-E0[ids, None]
            mass = np.histogram(delta, edges, weights=np.asarray(L[ids]))[0]/len(ids)
            rows.append(dict(spin=spin, energy=float(center),
                        TV_test_after_balance=float(.5*abs(mass-profiles['test'][spin, j]).sum())))
    return rows
