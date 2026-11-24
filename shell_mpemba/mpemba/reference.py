"""Controlled random-Hamiltonian tests of the four-wavefunction ansatz.

The original full-GOE versus spin-flip comparison diagnoses correlations.
The separate Mpemba example uses a nonuniform DOS and fixed preparations,
with independent variance training and dynamical validation.
"""
import json
import os
import hashlib
import time
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from . import RESULTS_DIR, PLOTS_DIR
from .wavefunctions import pair_histogram, fourier_moments
from .model import atomic_json, write_csv
from .crossings import crossing


def run_mpemba_demo(dimension=4096):
    """Controlled H0 + GOE example with an inhomogeneous density of states.

    This is a pedagogical random-Hamiltonian benchmark, not another local
    spin-chain validation. Train the variance on independent realizations;
    fix two physical preparations before training and keep all three tests.
    """
    from scipy.special import ndtri
    cfg = dict(dimension=dimension, train=12, tests=3, bath_sigma=1.5, hs=.35,
               g=.5, width=.4, states=[[0., .9], [2.5, .8]], seed=20261021,
               tmax=80., nt=401)
    out = RESULTS_DIR / "mpemba_reference" / f"dimension_{dimension}"
    out.mkdir(parents=True, exist_ok=True)
    cp = out / "config.json"
    if cp.exists() and json.loads(cp.read_text()) != cfg:
        raise ValueError("Existing benchmark configuration differs")
    if not cp.exists():
        atomic_json(cp, cfg)
    if all((out / f"validation_{k}.npz").exists() for k in range(cfg['tests'])):
        return plot_mpemba_demo(out)

    n, nb = dimension, dimension//2
    from .model import memory_available_gib
    required_gib = 1.5+20*8*n*n/2**30
    if memory_available_gib() < required_gib:
        raise MemoryError(f"Benchmark needs approximately {required_gib:.1f} GiB available")
    rng = np.random.default_rng(cfg['seed'])
    eb = cfg['bath_sigma']*ndtri((np.arange(nb)+.5)/nb)
    e0 = np.r_[eb+cfg['hs'], eb-cfg['hs']]
    o = np.r_[np.ones(nb), -np.ones(nb)]
    t = np.linspace(0, cfg['tmax'], cfg['nt'])
    probabilities = []
    for center, polarization in cfg['states']:
        w = np.exp(-.5*((eb-center)/cfg['width'])**2)
        w /= w.sum()
        probabilities.append(np.r_[.5*(1+polarization)*w, .5*(1-polarization)*w])
    probabilities = np.array(probabilities)

    def sample():
        R = rng.normal(size=(n, n))
        H = (R+R.T)*cfg['g']/np.sqrt(2*n)
        H[np.diag_indices(n)] += e0
        return eigh(H, check_finite=False, overwrite_a=True, driver='evd')

    L = np.zeros((n, n))
    for k in range(cfg['train']):
        E, U = sample()
        L += U*U
        print(f"Variance training {k+1}/{cfg['train']}", flush=True)
    L /= cfg['train']
    np.testing.assert_allclose(L.sum(0), 1., atol=1e-12)
    np.testing.assert_allclose(L.sum(1), 1., atol=1e-12)
    np.savez_compressed(out/'calibration.npz', Lambda=L, preparation=probabilities, E0=e0)

    W = L.T@L
    Wo = L.T@(o[:, None]*L)
    C = []
    for prob in probabilities:
        P = L.T@(prob[:, None]*L)
        PO = L.T@((prob*o)[:, None]*L)
        A = PO-P*Wo/W
        np.fill_diagonal(A, 0)
        C.append(A)
    operator = W-Wo*Wo/W
    np.fill_diagonal(operator, 0)

    for k in range(cfg['tests']):
        E, U = sample()
        phase = np.exp(-1j*E[:, None]*t)
        pred = np.array([np.sum(phase*(a@phase.conj()), axis=0).real for a in C])
        edges = np.arange(np.floor(E.min()), np.ceil(E.max())+.25, .25)
        wedge = np.linspace(0, np.nextafter(np.ptp(E), np.inf), 8194)
        omega = (wedge[:-1]+wedge[1:])/2
        h, _ = pair_histogram(E, operator, np.arange(n), edges, wedge)
        power = h[0].sum(1)
        K = np.divide(fourier_moments(h, omega, t), power[:, None],
                      out=np.zeros((len(power), len(t))), where=power[:, None] > 0)
        B = np.array([pair_histogram(E, a, np.arange(n), edges, wedge)[0][0].sum(1)
                      for a in C])
        shell_z = B@K
        np.testing.assert_allclose(shell_z[:, 0], pred[:, 0], atol=1e-12)
        common_z = B.sum(1)[:, None]*np.average(K, axis=0, weights=abs(B).sum(0))
        forecast = out/f'forecast_{k}.npz'
        data = dict(times=t, states=cfg['states'], predicted_z=pred, E=E, B=B,
                    K=K, energy_edges=edges, shell_z=shell_z, common_z=common_z)
        if forecast.exists():
            with np.load(forecast) as saved:
                for key, value in data.items():
                    np.testing.assert_allclose(saved[key], value, atol=1e-12)
        else:
            np.savez_compressed(forecast, **data)
            atomic_json(out/f'lock_{k}.json', dict(
                sha256=hashlib.sha256(forecast.read_bytes()).hexdigest(),
                created_unix=time.time(),
                selection='Two states fixed in config before training; no pair search.'))
        if (out/f'validation_{k}.npz').exists():
            continue

        ab = U[:nb].T@U[nb:]
        ops = [ab+ab.T, 1j*(ab.T-ab), 2*(U[:nb].T@U[:nb])-np.eye(n)]
        zdiag = np.diag(ops[2]).copy()
        stationary, bloch = [], []
        for prob in probabilities:
            rho = U.T@(prob[:, None]*U)
            stationary.append(rho.diagonal()@zdiag)
            components = []
            for op in ops:
                A = rho*op.T
                np.fill_diagonal(A, 0)
                z = np.sum(phase*(A@phase.conj()), axis=0)
                assert abs(z.imag).max() < 1e-10
                components.append(z.real)
            bloch.append(components)
        bloch = np.array(bloch)
        np.testing.assert_allclose(bloch[:, 2, 0],
                                   probabilities@o-np.array(stationary), atol=1e-12)
        np.savez_compressed(out/f'validation_{k}.npz', times=t, bloch=bloch,
                            exact=.5*np.linalg.norm(bloch, axis=1),
                            stationary_z=stationary, predicted=.5*abs(pred),
                            states=cfg['states'])
        print(f"Independent validation {k+1}/{cfg['tests']}", flush=True)
    return plot_mpemba_demo(out)


def plot_mpemba_demo(out):
    """Compare a fixed first realization with the existing notebook example."""
    cfg = json.loads((out/'config.json').read_text())
    criteria = dict(initial_gap=.01, post_gap=.005, persistence=1.,
                    decay_fraction=.95, distance_floor=.005)
    rows = []
    for k in range(cfg['tests']):
        fp = out/f'forecast_{k}.npz'
        lock = json.loads((out/f'lock_{k}.json').read_text())
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == lock['sha256']
        with np.load(fp) as f, np.load(out/f'validation_{k}.npz') as v:
            t, exact = f['times'], v['exact']
            for name, curves in [('exact', exact), ('full_wavefunction', .5*abs(f['predicted_z'])),
                                 ('shell', .5*abs(f['shell_z'])), ('common', .5*abs(f['common_z']))]:
                event = crossing(t, curves[0], curves[1], criteria)
                event = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                         for key, value in event.items()}
                rows.append(dict(realization=k, model=name, **event,
                                 initial_far=float(curves[0, 0]), initial_near=float(curves[1, 0]),
                                 late_far=float(curves[0, t >= 60].max()/curves[0, 0]),
                                 late_near=float(curves[1, t >= 60].max()/curves[1, 0]),
                                 relative_curve_error=float(np.linalg.norm(curves-exact)/np.linalg.norm(exact))))
    write_csv(out/'summary.csv', rows)
    target = PLOTS_DIR.parent/'regime_comparison'
    target.mkdir(parents=True, exist_ok=True)
    chain = RESULTS_DIR/'article_programme/selected_N14_width0.75/wavefunction'
    with np.load(chain/'forecast.npz') as f, np.load(chain/'predictions.npz') as v, \
            np.load(chain/'contraction_general.npz') as r:
        pair = json.loads((chain/'forecast_locked.json').read_text())['selected_pair']
        chain_data = dict(t=f['times'], exact=v['exact'][pair], full=f['general_full_raw'][pair],
                          shell=f['general_factorized_raw'][pair], B=f['general_B'][pair],
                          K=r['K'], energy=(f['energy_edges'][:-1]+f['energy_edges'][1:])/2)
    with np.load(out/'forecast_0.npz') as f, np.load(out/'validation_0.npz') as v:
        demo = dict(t=f['times'], exact=v['exact'], full=.5*abs(f['predicted_z']),
                    shell=.5*abs(f['shell_z']), B=f['B'], K=f['K'],
                    energy=(f['energy_edges'][:-1]+f['energy_edges'][1:])/2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), layout='constrained')
    for axs, data, title in zip(axes, [chain_data, demo],
                               ['Current N=14 spin chain', f'Controlled random Hamiltonian, d={cfg["dimension"]}']):
        t = data['t']
        for i, color, label in [(0, 'C0', 'Initially farther'), (1, 'C1', 'Initially nearer')]:
            b = np.argmax(abs(data['B'][i]))
            axs[0].plot(t, data['K'][b], color=color,
                        label=f'{label}: E ≈ {data["energy"][b]:.2g}')
            axs[1].plot(t, data['exact'][i], color=color, label=label)
            axs[1].plot(t, data['full'][i], '--', color=color)
            axs[1].plot(t, data['shell'][i], ':', color=color)
            axs[2].semilogy(t, data['exact'][i], color=color, label=label)
        event = crossing(t, *data['exact'], criteria)
        axs[1].axvline(event['time'], color='.5', lw=.8, label=f'Exact crossing: {event["time"]:.2f}')
        axs[1].plot([], [], '--', color='.3', label='Full wavefunction prediction')
        axs[1].plot([], [], ':', color='.3', label='Shell reduction')
        axs[0].set(xlim=(0, 20), ylim=(-.05, 1.05), xlabel='Time', ylabel='Predicted shell kernel', title=title)
        axs[1].set(xlim=(0, 20), ylim=(0, .5), xlabel='Time', ylabel='Distance to own diagonal ensemble', title='Raw distances: common vertical scale')
        axs[2].axvspan(60, 80, color='.8', alpha=.3)
        axs[2].set(xlim=(0, 80), ylim=(1e-5, .6), xlabel='Time', ylabel='Exact distance', title='Full window, including fluctuations')
        for ax in axs:
            ax.legend(fontsize=7)
    fig.savefig(target/'controlled_rmt_comparison.png', dpi=170)
    fig.savefig(target/'controlled_rmt_comparison.pdf')
    plt.close(fig)
    atomic_json(out/'completed.json', dict(config=cfg, summary=rows,
                interpretation='Exploratory controlled random-Hamiltonian demonstration; not a local-chain validation. First realization plotted without selection.'))
    return rows

OUT = RESULTS_DIR / "random_reference"
CFG = dict(dimensions=[512, 1024], train=24, test=24, span=8., hs=.35, g=.75,
           tmax=40., nt=321, mean_energy_window=[-1., 1.], seed=982134)


def sample(n, kind, rng):
    eb = np.linspace(-CFG['span']/2, CFG['span']/2, n//2)
    E0 = np.r_[eb+CFG['hs'], eb-CFG['hs']]
    if kind == 'full_GOE':
        R = rng.normal(size=(n, n))
        H = (R+R.T)*CFG['g']/np.sqrt(2*n)
    else:
        R = rng.normal(size=(n//2, n//2))*CFG['g']*np.sqrt(2/n)
        H = np.zeros((n, n))
        H[:n//2, n//2:] = R
        H[n//2:, :n//2] = R.T
    H[np.diag_indices(n)] += E0
    return eigh(H, overwrite_a=True, check_finite=False, driver='evd')


def histogram(E, A, wedge):
    # The middle bin is the predeclared central pair-energy window; retain
    # outside bins during histogramming, then select rather than clipping.
    h, b = pair_histogram(E, A, np.arange(len(E)), np.array([-np.inf, -1., 1., np.inf]), wedge)
    return h[:, 1], b[1]


def run(n, kind):
    out = OUT/f'{kind}_{n}.npz'
    if out.exists():
        return dict(np.load(out))
    rng = np.random.default_rng(CFG['seed']+n+(0 if kind == 'full_GOE' else 100000))
    L = np.zeros((n, n))
    first = np.zeros_like(L)
    for r in range(CFG['train']):
        E, U = sample(n, kind, rng)
        L += U**2
        if r < CFG['train']//2:
            first += U**2
    L /= CFG['train']
    first /= CFG['train']//2
    second = 2*L-first
    assert max(abs(L.sum(0)-1).max(), abs(L.sum(1)-1).max()) < 1e-11

    def contraction(A):
        u = A[:n//2].T@A[:n//2]
        d = A[n//2:].T@A[n//2:]
        return 4*u*d/(u+d)
    P = contraction(L)
    Pfirst = contraction(first)
    Psecond = contraction(second)
    times = np.linspace(0, CFG['tmax'], CFG['nt'])
    wedge = np.linspace(0, 20, 8194)
    omega = (wedge[:-1]+wedge[1:])/2
    H = []
    HP = []
    HS1 = []
    HS2 = []
    npc = []
    bound = []
    for r in range(CFG['test']):
        E, U = sample(n, kind, rng)
        assert np.ptp(E) < 20
        Z = 2*U[:n//2].T@U[:n//2]-np.eye(n)
        h, b = histogram(E, Z**2, wedge)
        hp, bp = histogram(E, P, wedge)
        h1, _ = histogram(E, Pfirst, wedge)
        h2, _ = histogram(E, Psecond, wedge)
        H.append(h)
        HP.append(hp)
        HS1.append(h1)
        HS2.append(h2)
        bound.append([b, bp])
        npc.append(np.median(1/(U[:, abs(E) < 1]**4).sum(0)))
        atomic_json(OUT/'progress.json', dict(stage='independent_validation', kind=kind,
                    dimension=n, completed=r+1, total=CFG['test'], pid=os.getpid()))
    H = np.array(H)
    HP = np.array(HP)
    power = H[:, 0].sum(1)
    ppower = HP[:, 0].sum(1)
    raw = np.array([fourier_moments(h, omega, times) for h in H])
    praw = np.array([fourier_moments(h, omega, times) for h in HP])
    K = raw.mean(0)/power.mean()
    KP = praw.mean(0)/ppower.mean()
    error = np.linalg.norm(KP-K)/np.linalg.norm(K)
    split = []
    for hs in [HS1, HS2]:
        hm = np.mean(hs, axis=0)
        split.append(fourier_moments(hm, omega, times)/hm[0].sum())
    # Ensemble mean and standard error of independently normalized test curves.
    curves = raw/power[:, None]
    sem = curves.std(0, ddof=1)/np.sqrt(len(curves))
    data = dict(times=times, exact_K=K, predicted_K=KP, test_standard_error=sem, individual_test_K=curves,
                split_K=split, relative_kernel_error=error, power_ratio=ppower.mean()/power.mean(), npc_median=np.median(npc),
                variance_row_error=abs(L.sum(1)-1).max(), variance_column_error=abs(L.sum(0)-1).max(),
                omega=omega, exact_spectrum=H[:, 0].mean(0)/power.mean(), predicted_spectrum=HP[:, 0].mean(0)/ppower.mean(),
                quadrature_bound=np.max(np.array(bound), axis=0)*times[-1]**3/np.array([power.min(), ppower.min()]))
    np.savez_compressed(out, **data)
    print(kind, n, 'error', error, 'power ratio',
          data['power_ratio'], 'npc', data['npc_median'], flush=True)
    return data


def run_reference():
    OUT.mkdir(parents=True, exist_ok=True)
    cp = OUT/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == CFG
    atomic_json(cp, CFG)
    results = {}
    rows = []
    for n in CFG['dimensions']:
        for kind in ['full_GOE', 'flip_only']:
            d = run(n, kind)
            results[(n, kind)] = d
            rows.append(dict(dimension=n, ensemble=kind, relative_kernel_error=float(d['relative_kernel_error']), power_ratio=float(d['power_ratio']), npc_median=float(
                d['npc_median']), split_sensitivity=float(np.linalg.norm(np.diff(d['split_K'], axis=0))/np.linalg.norm(d['predicted_K']))))
    write_csv(OUT/'summary.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for ax, kind in zip(axes, ['full_GOE', 'flip_only']):
        for n, c in zip(CFG['dimensions'], ['#2166ac', '#c44e52']):
            d = results[(n, kind)]
            t = d['times']
            ax.plot(t, d['exact_K'], color=c, label=f'd={n}, exact ensemble')
            ax.plot(t, d['predicted_K'], color=c, ls='--')
            ax.fill_between(t, d['exact_K']-2*d['test_standard_error'],
                            d['exact_K']+2*d['test_standard_error'], color=c, alpha=.15)
        ax.plot([], [], ls='--', color='.3', label='Four-point prediction')
        ax.set(xlim=(0, 15), xlabel='Time', ylabel='Normalized operator kernel',
               title=kind.replace('_', ' '))
        ax.legend(fontsize=8)
    fig.suptitle('Independent random-Hamiltonian reference; identical variance-to-correlation formula')
    target = PLOTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target/'S1_random_hamiltonian_reference.pdf', bbox_inches='tight')
    fig.savefig(target/'S1_random_hamiltonian_reference.png', dpi=180, bbox_inches='tight')
    atomic_json(OUT/'completed.json', dict(summary=rows,
                warning='Finite-participation ensemble diagnostic; not a physical-model validation.'))
    plt.close(fig)
