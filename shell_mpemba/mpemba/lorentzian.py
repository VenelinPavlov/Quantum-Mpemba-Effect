"""Lorentzian calibration, static amplitudes, and exact spin dynamics.

Dense helpers support the pedagogical notebook (N<=12); run_streamed supports
N<=14 without retaining full operator or density matrices. The spin-channel
kinetic closure is explicitly separate from the direct wavefunction contraction.
Neither closure fits a dynamical rate to a relaxation trajectory.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from . import CACHE_DIR
from .model import full_eigensystem, spin_columns, atomic_json, write_csv
from .crossings import crossing, eligible_pairs


def line_masses(edges, shift, width, gaussian=False):
    """Integrated, normalized model including the two infinite tail bins."""
    x = (edges-shift)/width
    cdf = ndtr(x) if gaussian else .5+np.arctan(x)/np.pi
    return np.diff(cdf)


def fit_histogram(mass, edges, gaussian=False):
    """Same integrated-bin objective, analytic Jacobian, deterministic starts.

    Sparse edge spectra can trap the finite-difference optimizer. All starts
    and bounds are fixed in advance; no held-out data choose parameters.
    """
    def residual(par):
        return line_masses(edges, par[0], np.exp(par[1]), gaussian)-mass

    def jac(par):
        width = np.exp(par[1])
        z = (edges-par[0])/width
        finite = np.isfinite(z)
        a = np.zeros_like(z)
        b = np.zeros_like(z)
        if gaussian:
            phi = np.exp(-z[finite]**2/2)/np.sqrt(2*np.pi)
            a[finite] = -phi/width
            b[finite] = -z[finite]*phi
        else:
            f = 1/(np.pi*(1+z[finite]**2))
            a[finite] = -f/width
            b[finite] = -z[finite]*f
        return np.column_stack([np.diff(a), np.diff(b)])
    centers = (edges[1:-2]+edges[2:-1])/2
    mode = float(np.clip(centers[np.argmax(mass[1:-1])], -.49, .49))
    starts = [[0., np.log(.05)], [mode, np.log(.01)], [mode, np.log(.1)]]
    fits = [least_squares(residual, x, jac=jac, max_nfev=10000,
                          bounds=([-.5, np.log(.001)], [.5, np.log(1.)])) for x in starts]
    good = [f for f in fits if f.success]
    if not good:
        raise RuntimeError('All deterministic line-profile fits failed')
    fit = min(good, key=lambda x: x.cost)
    shift = float(fit.x[0])
    width = float(np.exp(fit.x[1]))
    return shift, width, line_masses(edges, shift, width, gaussian)


def fit_envelopes(E, U, E0, energy_edges, epsilon_edges, min_states=40):
    """Alternate eigenstates train/test; no operator or trajectory enters.
    Least squares of bin probabilities: unit normalization, two fit parameters,
    analytic bin integration. Test-half refits are sensitivity checks only.
    """
    rows, profiles = [], []
    for k, center in enumerate((energy_edges[:-1]+energy_edges[1:])/2):
        ids = np.flatnonzero((E >= energy_edges[k]) & (E < energy_edges[k+1]))
        if len(ids) < min_states:
            continue
        W = U[:, ids]**2
        detuning = E[ids][None, :]-E0[:, None]
        masses = [np.histogram(detuning[:, s::2], epsilon_edges, weights=W[:, s::2])[
            0]/len(ids[s::2]) for s in (0, 1)]
        assert all(abs(h.sum()-1) < 1e-10 for h in masses)
        shift, gamma, lor = fit_histogram(masses[0], epsilon_edges)
        _, sigma, gauss = fit_histogram(masses[0], epsilon_edges, True)
        _, gamma_test, _ = fit_histogram(masses[1], epsilon_edges)
        coarse_edges = np.r_[-np.inf, epsilon_edges[1:-1:2], np.inf]
        if coarse_edges[-2] != epsilon_edges[-2]:
            coarse_edges = np.insert(coarse_edges, -1, epsilon_edges[-2])
        coarse_mass = np.histogram(detuning[:, ::2], coarse_edges,
                                   weights=W[:, ::2])[0]/len(ids[::2])
        _, gamma_coarse, _ = fit_histogram(coarse_mass, coarse_edges)
        spacing = float(np.median(np.diff(E[ids])))
        rows.append(dict(bin=k, energy=float(center), eigenstates=len(ids), gamma=gamma, shift=shift,
                         gamma_test=gamma_test, gamma_coarse=gamma_coarse, sigma=sigma,
                         TV_lorentzian_test=float(.5*np.abs(masses[1]-lor).sum()),
                         TV_gaussian_test=float(.5*np.abs(masses[1]-gauss).sum()),
                         npc_median=float(np.median(1/(W**2).sum(0))), spacing=spacing,
                         gamma_over_spacing=gamma/spacing, gamma_over_edge=gamma/min(center-E0.min(), E0.max()-center)))
        profiles.append(
            dict(bin=k, training=masses[0], heldout=masses[1], lorentzian=lor, gaussian=gauss))
    if not rows:
        raise RuntimeError('No bin contains enough eigenstates to fit.')
    return rows, profiles


def fit_channel_envelopes(E, U, E0, energy_edges, epsilon_edges, min_states=20):
    """Fit spin-resolved bare-state strength functions from the SAME overlaps.

    Conditioning is on bare state alpha and its spin, rather than exact state m.
    fit_envelopes is reused after transposition; detuning sign is reversed back.
    No survival curve, operator spectrum, or state trajectory enters the fit.
    """
    b = len(E)//2
    rows = []
    profiles = []
    for spin, sl in [(0, slice(0, b)), (1, slice(b, 2*b))]:
        rr, pp = fit_envelopes(E0[sl], U[sl].T, E, energy_edges, epsilon_edges, min_states)
        for r in rr:
            r['spin'] = spin
            r['shift'] = -r['shift']
            k = r['bin']
            ids = (E >= energy_edges[k]) & (E < energy_edges[k+1])
            r['spacing'] = float(np.median(np.diff(E[ids])))
            r['gamma_over_spacing'] = r['gamma']/r['spacing']
        for profile in pp:
            profile['spin'] = spin
            for name in ['training', 'heldout', 'lorentzian', 'gaussian']:
                profile[name] = profile[name][::-1].copy()
        rows.extend(rr)
        profiles.extend(pp)
    return rows, profiles


def predict_channels(B, rows, energy_centers, times, fit_range):
    """Spin-flip population rate = 2*(Gamma_up + Gamma_down).

    Gamma_s is the strength-function HWHM. The probability escape rate from
    a bare state is 2*Gamma_s. Every resonant first-order V transition flips
    the impurity; the two-state population generator has the SUM of rates.
    Returns rate/2 in the legacy 'gamma' slot, not either physical linewidth.
    """
    widths = []
    outside = np.zeros(len(energy_centers), bool)
    for spin in [0, 1]:
        used = [r for r in rows if r['spin'] == spin and fit_range[0] <= r['energy'] <= fit_range[1]]
        ec = np.array([r['energy'] for r in used])
        widths.append(np.interp(energy_centers, ec, [r['gamma'] for r in used]))
        outside |= (energy_centers < ec.min()) | (energy_centers > ec.max())
    half_rate = np.sum(widths, axis=0)
    K = np.exp(-2*half_rate[:, None]*np.abs(times)[None, :])
    z = B@K
    weight = np.abs(B).sum(0)
    flat_half_rate = float(weight@half_rate/weight.sum())
    flat_z = B.sum(1)[:, None]*np.exp(-2*flat_half_rate*times)
    outside_weight = np.abs(B[:, outside]).sum(1)/np.abs(B).sum(1)
    return half_rate, K, z, flat_z, outside_weight


def preparations(Eb, bath_centers, bath_width, polarizations):
    basis, states, mixing = [], [], []
    for center in bath_centers:
        w = np.exp(-.5*((Eb-center)/bath_width)**2)
        w /= w.sum()
        basis.extend([np.r_[w, np.zeros_like(w)], np.r_[np.zeros_like(w), w]])
    for j, center in enumerate(bath_centers):
        for m in polarizations:
            weights = np.zeros(len(basis))
            weights[2*j:2*j+2] = [(1+m)/2, (1-m)/2]
            mixing.append(weights)
            states.append([center, m])
    return np.array(basis), np.array(mixing), np.array(states)


def spin_operators(U):
    """Physical operators (X, i*Yreal, Z), zero diagonal on return."""
    b = len(U)//2
    cross = U[:b].T@U[b:]
    X = cross+cross.T
    Yreal = cross.T-cross
    del cross
    Z = 2*(U[:b].T@U[:b])-np.eye(len(U))
    ops = [X, Yreal, Z]
    diag = np.array([o.diagonal().copy() for o in ops])
    for op in ops:
        np.fill_diagonal(op, 0.)
    return ops, diag


def static_amplitudes(E, U, basis, mixing, ops, diagonal, energy_edges):
    """Exact initial signed amplitudes: no Fourier transform/time samples."""
    nb = len(energy_edges)-1
    index = np.clip(np.searchsorted(
        energy_edges, (E[:, None]+E[None, :])/2, side='right')-1, 0, nb-1).astype(np.int16)
    B = np.zeros((len(basis), nb))
    stationary = np.zeros((len(basis), 3))
    for i, p in enumerate(basis):
        rho = U.T@(p[:, None]*U)
        stationary[i] = diagonal@rho.diagonal()
        B[i] = np.bincount(index.ravel(), weights=(rho*ops[2]).ravel(), minlength=nb)
        print(f'Static amplitudes {i+1}/{len(basis)}', flush=True)
    return mixing@B, mixing@stationary


def predict(B, rows, energy_centers, times, fit_range):
    """Training wavefunction widths set rates; explicitly clamp outside range."""
    used = [r for r in rows if fit_range[0] <= r['energy'] <= fit_range[1]]
    ec = np.array([r['energy'] for r in used])
    gamma = np.interp(energy_centers, ec, [r['gamma'] for r in used])
    K = np.exp(-2*gamma[:, None]*np.abs(times)[None, :])
    z = B@K
    weight = np.abs(B).sum(0)
    flat_gamma = float(weight@gamma/weight.sum())
    flat_z = B.sum(1)[:, None]*np.exp(-2*flat_gamma*times)
    outside = (energy_centers < ec.min()) | (energy_centers > ec.max())
    outside_weight = np.abs(B[:, outside]).sum(1)/np.abs(B).sum(1)
    return gamma, K, z, flat_z, outside_weight


def lock_forecast(out, config, states, times, B, stationary, rows, profiles, epsilon_edges, energy_edges):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ec = (energy_edges[:-1]+energy_edges[1:])/2
    predictor = predict_channels if config.get('kernel_model') == 'spin_flip' else predict
    gamma, K, z, flat_z, outside = predictor(B, rows, ec, times, config['fit_range'])
    initial_bloch = np.zeros_like(stationary)
    initial_bloch[:, 2] = states[:, 1]
    initial = .5*np.linalg.norm(initial_bloch-stationary, axis=1)
    records = []
    for far, near in eligible_pairs(initial, config['criteria']):
        for name, D in [('lorentzian', .5*abs(z)), ('flat', .5*abs(flat_z))]:
            records.append(dict(far=far, near=near, model=name, **
                           crossing(times, D[far], D[near], config['criteria'])))
    np.savez_compressed(out/'forecast.npz', states=states, times=times, B=B, stationary=stationary,
                        gamma=gamma, K=K, z=z, flat_z=flat_z, outside_fit_weight=outside, initial_distance=initial,
                        epsilon_edges=epsilon_edges, energy_edges=energy_edges, profile_bins=[
                            p['bin'] for p in profiles],
                        **{key: np.array([p[key] for p in profiles]) for key in ['training', 'heldout', 'lorentzian', 'gaussian']})
    write_csv(out/'envelope_fits.csv', rows)
    write_csv(out/'forecast_pairs.csv', records)
    lock = dict(config=config, created_unix=time.time(),
                forecast_sha256=hashlib.sha256((out/'forecast.npz').read_bytes()).hexdigest(),
                eigenstate_fits_only=True, trajectories_computed_before_lock=False,
                calibration='Exact B_z and preparation-specific diagonal ensembles; training eigenvector envelope fits.',
                interpretation='Population-dominated approximation; validation retains all spin components.',
                kernel_model=config.get('kernel_model', 'unstructured'),
                stored_gamma_meaning='Gamma_up + Gamma_down' if config.get(
                    'kernel_model') == 'spin_flip' else 'exact-state envelope HWHM',
                eligible_pairs=len(records)//2,
                predicted_crossings=sum(r['model'] == 'lorentzian' and r['observed'] for r in records))
    (out/'forecast_locked.json').write_text(json.dumps(lock, indent=2)+'\n')
    print('FORECAST LOCKED:', lock['eligible_pairs'], 'pairs,',
          lock['predicted_crossings'], 'predicted crossings', flush=True)
    return dict(gamma=gamma, K=K, z=z, flat_z=flat_z, outside=outside, initial=initial, records=records)


def exact_dynamics(E, U, basis, mixing, ops, times):
    """Unbinned ED sum: no fit, histogram, Markov approximation or time step."""
    phase = np.exp(-1j*E[:, None]*times)
    phase_real = np.ascontiguousarray(phase.real)
    phase_imag = np.ascontiguousarray(phase.imag)
    result = np.empty((len(basis), 3, len(times)))
    for i, p in enumerate(basis):
        rho = U.T@(p[:, None]*U)
        for a, op in enumerate(ops):
            A = rho*op
            val = np.sum(phase*(A@phase_real-1j*(A@phase_imag)), axis=0)
            result[i, a] = ((-1j if a == 1 else 1.)*val).real
        print(f'Exact trajectories {i+1}/{len(basis)}', flush=True)
    return np.einsum('is,sat->iat', mixing, result)


def operator_kernels(E, Z, energy_edges, times, n_omega=16385, block=128):
    """Independent operator validation with bounded second-order gap quadrature."""
    ne = len(energy_edges)-1
    wedges = np.linspace(0, np.nextafter(np.ptp(E), np.inf), n_omega+1)
    wc = (wedges[:-1]+wedges[1:])/2
    hist = np.zeros((3, ne, n_omega))
    bound = np.zeros(ne)
    for j in range(0, len(E), block):
        cols = np.arange(j, min(j+block, len(E)))
        keep = np.arange(len(E))[:, None] < cols[None, :]
        gap = (E[cols][None, :]-E[:, None])[keep]
        ebar = ((E[cols][None, :]+E[:, None])/2)[keep]
        weight = 2*Z[:, cols][keep]**2
        ei = np.clip(np.searchsorted(energy_edges, ebar, side='right')-1, 0, ne-1)
        wi = np.clip(np.searchsorted(wedges, gap, side='right')-1, 0, n_omega-1)
        dw = gap-wc[wi]
        ix = ei*n_omega+wi
        for k in range(3):
            hist[k] += np.bincount(ix, weights=weight*dw**k, minlength=ne *
                                   n_omega).reshape(ne, n_omega)
        bound += np.bincount(ei, weights=weight*abs(dw)**3, minlength=ne)*times[-1]**3/6
    power = hist[0].sum(1)
    K = np.zeros((ne, len(times)))
    for j in range(0, len(times), 32):
        sl = slice(j, j+32)
        t = times[sl]
        co = np.cos(wc[:, None]*t)
        si = np.sin(wc[:, None]*t)
        K[:, sl] = hist[0]@co-(hist[1]@si)*t-(hist[2]@co)*(t*t/2)
    np.divide(K, power[:, None], out=K, where=power[:, None] > 0)
    return K, wc, np.divide(hist[0], power[:, None], out=np.zeros_like(hist[0]), where=power[:, None] > 0), np.divide(bound, power, out=np.zeros_like(bound), where=power > 0)


def evaluate(out, states, times, exact, stationary, fc, criteria):
    out = Path(out)
    D = .5*np.linalg.norm(exact, axis=1)
    Dz = .5*abs(exact[:, 2])
    np.savez_compressed(out/'validation.npz', times=times, states=states,
                        bloch=exact, stationary=stationary)
    physical = exact+stationary[:, :, None]
    expected = np.zeros_like(stationary)
    expected[:, 2] = states[:, 1]
    assert np.allclose(physical[:, :, 0], expected, atol=1e-9)
    assert np.max(np.linalg.norm(physical, axis=1)) <= 1+1e-9
    rows = []
    summary = []
    pairs = eligible_pairs(fc['initial'], criteria)
    for target, ref in [('full_trace_distance', D), ('population_distance', Dz)]:
        for model, pred in [('lorentzian', .5*abs(fc['z'])), ('flat', .5*abs(fc['flat_z']))]:
            for far, near in pairs:
                ex = crossing(times, ref[far], ref[near], criteria)
                pr = crossing(times, pred[far], pred[near], criteria)
                rows.append(dict(target=target, model=model, far=far, near=near, exact=ex['observed'], predicted=pr['observed'],
                                 t_exact=ex['time'], t_predicted=pr['time'], exact_depth=ex['depth'], predicted_depth=pr['depth'],
                                 stationary_distance=float(.5*np.linalg.norm(stationary[far]-stationary[near]))))
            rr = [r for r in rows if r['target'] == target and r['model'] == model]
            ss = dict(target=target, model=model, pairs=len(rr), TP=sum(r['exact'] and r['predicted'] for r in rr),
                      FP=sum(not r['exact'] and r['predicted'] for r in rr), FN=sum(r['exact'] and not r['predicted'] for r in rr),
                      TN=sum(not r['exact'] and not r['predicted'] for r in rr))
            errors = [abs(r['t_exact']-r['t_predicted'])
                      for r in rr if r['exact'] and r['predicted']]
            ss['crossing_time_MAE'] = float(np.mean(errors)) if errors else float('nan')
            summary.append(ss)
    write_csv(out/'pairs.csv', rows)
    write_csv(out/'summary.csv', summary)
    return rows, summary


def density_columns(U, p, sl):
    """Exploit the up/down preparation basis without discarding small weights."""
    b = len(U)//2
    if not np.any(p[b:]):
        return U[:b].T@(p[:b, None]*U[:b, sl])
    if not np.any(p[:b]):
        return U[b:].T@(p[b:, None]*U[b:, sl])
    return U.T@(p[:, None]*U[:, sl])


def run_streamed(config, out, block=128, forecast_only=False, *, cache=CACHE_DIR):
    """N<=14 follow-up. No full spin operator or density matrices are retained.

    Writes the compact data products used by the notebooks. Intermediate progress is
    recorded in progress.json; forecast remains locked throughout validation.
    """
    assert 4 <= config['model']['N'] <= 14
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'progress.json').exists():
        previous = json.loads((out/'progress.json').read_text())
        if previous['stage'] == 'complete':
            lock = json.loads((out/'forecast_locked.json').read_text())
            if lock['config'] != config:
                raise ValueError('Completed run configuration differs')
            if hashlib.sha256((out/'forecast.npz').read_bytes()).hexdigest() != lock['forecast_sha256']:
                raise ValueError('Forecast hash mismatch')
            if not all((out/f).exists() for f in ['validation.npz', 'operator_validation.npz']):
                raise ValueError('Completed run lacks validation files')
            print('Reusing completed size check:', out, flush=True)
            return previous['summary']

    def progress(stage, **extra):
        record = dict(stage=stage, updated_unix=time.time(), **extra)
        record = json.loads(json.dumps(record), parse_constant=lambda value: None)
        atomic_json(out/'progress.json', record)
        print(record, flush=True)
    progress('diagonalization')
    E, U, Eb, E0 = full_eigensystem(config['model'], Path(cache), block)
    d = len(E)
    times = np.linspace(0, config['tmax'], config['nt'])
    edges = np.arange(np.floor(E.min()), np.ceil(E.max())+config['delta_E'], config['delta_E'])
    ec = (edges[:-1]+edges[1:])/2
    ne = len(ec)
    eps = np.r_[-np.inf, np.linspace(-2, 2, 1001), np.inf]
    basis, mixing, states = preparations(
        Eb, config['bath_centers'], config['bath_width'], config['polarizations'])
    ns = len(basis)
    if (out/'forecast_locked.json').exists():
        lock = json.loads((out/'forecast_locked.json').read_text())
        if lock['config'] != config:
            raise ValueError('Locked forecast configuration differs')
        if hashlib.sha256((out/'forecast.npz').read_bytes()).hexdigest() != lock['forecast_sha256']:
            raise ValueError('Forecast hash mismatch')
        f = np.load(out/'forecast.npz')
        B = f['B']
        stationary = f['stationary']
        fc = dict(z=f['z'], flat_z=f['flat_z'], initial=f['initial_distance'])
    else:
        if config.get('kernel_model') == 'spin_flip':
            rows, profiles = fit_channel_envelopes(
                E, U, E0, edges, eps, config['min_fit_states']//2)
        else:
            rows, profiles = fit_envelopes(E, U, E0, edges, eps, config['min_fit_states'])
        Bbase = np.zeros((ns, ne))
        stationary_base = np.zeros((ns, 3))
        for j in range(0, d, block):
            sl = slice(j, min(j+block, d))
            cols = np.arange(j, sl.stop)
            ops = spin_columns(U, sl)
            diagonal = ops[:, cols, np.arange(len(cols))].copy()
            ops[:, cols, np.arange(len(cols))] = 0
            ei = np.clip(np.searchsorted(
                edges, (E[:, None]+E[cols][None, :])/2, side='right')-1, 0, ne-1)
            for s, p in enumerate(basis):
                rho = density_columns(U, p, sl)
                stationary_base[s] += diagonal@rho[cols, np.arange(len(cols))]
                Bbase[s] += np.bincount(ei.ravel(), weights=(rho*ops[2]).ravel(), minlength=ne)
            progress('static_amplitudes', columns_complete=sl.stop, dimension=d)
        B = mixing@Bbase
        stationary = mixing@stationary_base
        assert np.allclose(B.sum(1), states[:, 1]-stationary[:, 2], atol=1e-9)
        fc = lock_forecast(out, config, states, times, B, stationary, rows, profiles, eps, edges)
    if forecast_only:
        progress('forecast_locked_only')
        return dict(forecast_locked=True)
    atomic_json(out/'validation_started.json', dict(started_unix=time.time()))
    phase = np.exp(-1j*E[:, None]*times)
    pr = np.ascontiguousarray(phase.real)
    pi = np.ascontiguousarray(phase.imag)
    exact_base = np.zeros((ns, 3, len(times)))
    nw = 16385
    wedges = np.linspace(0, np.nextafter(np.ptp(E), np.inf), nw+1)
    wc = (wedges[:-1]+wedges[1:])/2
    hist = np.zeros((3, ne, nw))
    bound = np.zeros(ne)
    for j in range(0, d, block):
        sl = slice(j, min(j+block, d))
        cols = np.arange(j, sl.stop)
        ops = spin_columns(U, sl)
        ops[:, cols, np.arange(len(cols))] = 0
        for s, p in enumerate(basis):
            rho = density_columns(U, p, sl)
            for a in range(3):
                A = rho*ops[a]
                val = np.sum(phase*(A@pr[sl]-1j*(A@pi[sl])), axis=0)
                exact_base[s, a] += ((-1j if a == 1 else 1)*val).real
        keep = np.arange(d)[:, None] < cols[None, :]
        gap = (E[cols][None, :]-E[:, None])[keep]
        ebar = ((E[cols][None, :]+E[:, None])/2)[keep]
        weight = 2*ops[2][keep]**2
        ei = np.clip(np.searchsorted(edges, ebar, side='right')-1, 0, ne-1)
        wi = np.clip(np.searchsorted(wedges, gap, side='right')-1, 0, nw-1)
        dw = gap-wc[wi]
        ix = ei*nw+wi
        for k in range(3):
            hist[k] += np.bincount(ix, weights=weight*dw**k, minlength=ne*nw).reshape(ne, nw)
        bound += np.bincount(ei, weights=weight*abs(dw)**3, minlength=ne)*times[-1]**3/6
        progress('exact_validation', columns_complete=sl.stop, dimension=d)
    exact = np.einsum('is,sat->iat', mixing, exact_base)
    _, summary = evaluate(out, states, times, exact, stationary, fc, config['criteria'])
    power = hist[0].sum(1)
    K = np.zeros((ne, len(times)))
    for j in range(0, len(times), 32):
        sl = slice(j, j+32)
        t = times[sl]
        co = np.cos(wc[:, None]*t)
        si = np.sin(wc[:, None]*t)
        K[:, sl] = hist[0]@co-(hist[1]@si)*t-(hist[2]@co)*(t*t/2)
    np.divide(K, power[:, None], out=K, where=power[:, None] > 0)
    Sh = np.divide(hist[0], power[:, None], out=np.zeros_like(hist[0]), where=power[:, None] > 0)
    bd = np.divide(bound, power, out=np.zeros_like(bound), where=power > 0)
    np.savez_compressed(out/'operator_validation.npz', K=K, omega=wc, spectrum=Sh, bound=bd)
    progress('complete', summary=summary)
    return summary


def calibration_sensitivity(config, out):
    """Repeat calibration and resolution checks; original forecast is unchanged."""
    import csv
    assert config['model']['N'] <= 12, 'Dense resolution check is capped at N=12.'
    out = Path(out)
    cfg = config
    lock = json.loads((out/'forecast_locked.json').read_text())
    if lock['config'] != cfg:
        raise ValueError('Sensitivity configuration mismatch')
    f = np.load(out/'forecast.npz')
    v = np.load(out/'validation.npz')
    t = f['times']
    initial = f['initial_distance']
    D = .5*np.linalg.norm(v['bloch'], axis=1)
    with (out/'envelope_fits.csv').open() as h:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(h)]
    B = f['B']
    ec = (f['energy_edges'][:-1]+f['energy_edges'][1:])/2
    results = []

    def score(label, pred, times=t, truth=D):
        pairs = eligible_pairs(initial, cfg['criteria'])
        tp = fp = fn = tn = 0
        for a, b in pairs:
            ex = crossing(times, truth[a], truth[b], cfg['criteria'])['observed']
            pr = crossing(times, pred[a], pred[b], cfg['criteria'])['observed']
            tp += ex and pr
            fp += not ex and pr
            fn += ex and not pr
            tn += not ex and not pr
        results.append(dict(check=label, TP=tp, FP=fp, FN=fn, TN=tn))
    score('Original', .5*abs(f['z']))
    for label, key in [('Other eigenstate half', 'gamma_test'), ('Double detuning-bin width', 'gamma_coarse')]:
        alt = [dict(r, gamma=r[key]) for r in rows]
        _, _, z, _, _ = predict(B, alt, ec, t, cfg['fit_range'])
        score(label, .5*abs(z))
    _, _, z, _, _ = predict(B, rows, ec, t, [-10, 7])
    score('Extend fitted energy range', .5*abs(z))
    E, U, Eb, E0 = full_eigensystem(cfg['model'], CACHE_DIR, 64)
    basis, mix, states = preparations(
        Eb, cfg['bath_centers'], cfg['bath_width'], cfg['polarizations'])
    ops, diag = spin_operators(U)
    fineedges = np.arange(f['energy_edges'][0], f['energy_edges'][-1]+.5, .5)
    Bfine, stationary = static_amplitudes(E, U, basis, mix, ops, diag, fineedges)
    fineec = (fineedges[:-1]+fineedges[1:])/2
    _, _, zfine, _, _ = predict(Bfine, rows, fineec, t, cfg['fit_range'])
    score('Half mean-energy-bin width', .5*abs(zfine))
    tmid = (t[:-1]+t[1:])/2
    exactmid = exact_dynamics(E, U, basis, mix, ops, tmid)
    fine_t = np.empty(2*len(t)-1)
    fine_t[::2] = t
    fine_t[1::2] = tmid
    exactfine = np.empty((len(states), 3, len(fine_t)))
    exactfine[:, :, ::2] = v['bloch']
    exactfine[:, :, 1::2] = exactmid
    _, _, ztime, _, _ = predict(B, rows, ec, fine_t, cfg['fit_range'])
    score('Half time-sampling interval', .5*abs(ztime),
          times=fine_t, truth=.5*np.linalg.norm(exactfine, axis=1))
    np.savez_compressed(out/'resolution_check.npz', times=fine_t, bloch=exactfine,
                        B_half_energy_bin=Bfine, z_half_energy_bin=zfine)
    write_csv(out/'calibration_sensitivity.csv', results)
    return results
