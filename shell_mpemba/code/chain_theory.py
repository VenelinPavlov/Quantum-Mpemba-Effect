"""Predictions for the impurity spin chain (Secs. III and IV A of the paper), from static inputs only.

Each chain run is a folder in chain_runs/ (common.RUNS; see CASES). The functions read
  wavefunction/forecast_locked.json   model, preparations and numerical settings ("forecasts_locked.json" in
                                      some runs; the name is historical: a stored prediction)
  wavefunction/envelopes.npz          LDOS profiles per spin sector and energy, the input of the general Lambda
  wavefunction/contraction_general.npz  the pair weights of Eq. (11), binned by mean pair energy and gap
  exact/validation.npz                exact Bloch vectors of the preparations (used only for comparison)
and the bath eigensystem, cached in cache/spin_chain. Nothing here is fitted to the exact dynamics.

Notation (paper / code):
  gamma_s, the sector widths of Eq. (14)   1/tau_gen: tau_gen is the mean residence time of the general Lambda
  kappa, the spin relaxation rate           b_up/tau_core_up + b_dn/tau_core_dn: Lorentzian core widths
                                            (tau_core = 1/(2 Gamma_core)) and detailed-balanced flip shares b_s
  k_sp/k_up, k_dn/k_up                      Golden-Rule rate ratios at each bath filter energy (Fig. 4)
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common import CACHE
from mpemba import vertex as vx                      # the channel correction (Sec. III F)
from mpemba.lorentzian import preparations
from mpemba.model import bath_eigensystem

CASES = {0.0: 'pure_flip_pilot_N14', 0.25: 'sweep_N14_eta0.25', 0.5: 'selected_N14_width0.75',
         0.75: 'sweep_N14_eta0.75', 1.0: 'sweep_N14_eta1'}          # coupling eta -> run folder
INDEPENDENT_TEST = 0.75        # exact dynamics computed only after the final definition of kappa was fixed


# ---- inputs of a run

def run_config(case):
    """Model, preparations and numerical settings of a run."""
    W = Path(case)/'wavefunction'
    lock = next(p for p in (W/'forecast_locked.json', W/'forecasts_locked.json') if p.exists())
    meta = json.loads(lock.read_text())
    return meta.get('config', meta), meta['numerics']


def general_lambda(case, E, E0):
    """The general Lambda(mu, alpha) of a run (rows alpha, columns mu): the stored lambda_general.npy if present,
    otherwise rebuilt from the run's envelopes.npz exactly as the run built it (mpemba.wavefunctions.construct_variance)
    and cached in cache/lambda/<run>. Rebuilding takes 5-10 min and 2 GiB of disk at N = 14."""
    stored = Path(case)/'wavefunction/lambda_general.npy'
    if stored.exists():
        return np.load(stored, mmap_mode='r')
    from mpemba.wavefunctions import construct_variance
    _, numerics = run_config(case)
    out = CACHE/'lambda'/Path(case).name
    out.mkdir(parents=True, exist_ok=True)
    return construct_variance(E, E0, dict(np.load(Path(case)/'wavefunction/envelopes.npz')), 'general', numerics, out)


def exact_distances(case):
    """D_i(t) = |r_i(t) - r_i^inf|/2 of every preparation (the stored Bloch vectors are measured from r_i^inf)."""
    return .5*np.linalg.norm(np.load(Path(case)/'exact/validation.npz')['bloch'], axis=1)


def energy_populations(v, U):
    """(rho_i)_mu,mu for preparations with unperturbed populations v (rows), from the eigenvectors U (columns)."""
    q = np.zeros((len(v), U.shape[1]))
    for c in range(0, U.shape[1], 1024):
        q[:, c:c+1024] = v@np.asarray(U[:, c:c+1024])**2
    return q


def load(case):
    """Everything the predictions below need for one run."""
    case = Path(case)
    W = case/'wavefunction'
    cfg, numerics = run_config(case)
    p = cfg['model']
    eb, ub = bath_eigensystem(p, CACHE/'spin_chain')
    basis, mixing, states = preparations(eb, cfg['bath_centers'], cfg['bath_width'], cfg['polarizations'])
    prof = dict(np.load(W/'envelopes.npz'))
    V, preps = vx.chain_inputs(p, eb, ub, basis)
    del ub
    e_up, e_dn = eb+p['hs'], eb-p['hs']                                          # unperturbed energies per sector
    rates = vx.channel_rates(e_up, e_dn, *V, prof['centers'], prof['gamma'])    # Golden Rule, per unperturbed state
    del V
    grid = vx.shell_grid(preps)                                                  # energy grid of the statics
    b_up = vx.to_grid(e_up, vx.flip_share(rates, 0), grid)                       # flip share of an up state
    b_dn = vx.to_grid(e_dn, vx.flip_share(rates, 1), grid)
    rho_u, rho_d = vx.density(e_up, grid), vx.density(e_dn, grid)
    z = np.load(W/'contraction_general.npz')
    t = z['times']
    _, tau_gen = vx.survival_kernels(prof, 'general', grid, t)                   # residence times, general Lambda
    _, tau_core = vx.survival_kernels(prof, 'lorentzian', grid, t)               # Lorentzian core: 1/(2 Gamma_core)
    return SimpleNamespace(case=case, cfg=cfg, numerics=numerics, model=p, t=t, basis=basis, mixing=mixing,
                           states=states, preps=preps, prof=prof, e_up=e_up, e_dn=e_dn, rates=rates, grid=grid,
                           b_up=b_up, b_dn=b_dn, pi_u=rho_u/(rho_u+rho_d), tau_gen=tau_gen, tau_core=tau_core,
                           omega=z['omega'], edges=z['edges'], H=z['state_spectrum'], A=z['operator_spectrum'],
                           Bbase=z['Bbase'], exact=exact_distances(case))


# ---- predictions

def detuned_typicality(D, v, L, E, E0, t):
    """Eq. (10): D_i(0) |sum_alpha (rho_i)_alpha,alpha g_alpha(t)|^2, with the detuned survival amplitudes
    g_alpha(t) = sum_mu Lambda(mu, alpha) e^{i (E_mu - E_alpha) t} of the general Lambda."""
    co, si = np.cos(np.outer(E, t)), np.sin(np.outer(E, t))
    amp = np.zeros((len(v), len(t)), complex)
    for a in range(0, len(E0), 1024):
        B = np.asarray(L[a:a+1024])
        amp += v[:, a:a+1024]@((B@co-1j*(B@si))*np.exp(1j*np.outer(E0[a:a+1024], t)))
    return D[:, :1]*np.abs(amp)**2


def factor(omega, gu, gd, kap, piu):
    """Phi_b(omega) of Eq. (14) for one shell: sector widths gu, gd, spin relaxation rate kap, up fraction piu."""
    w2 = omega**2
    return kap/(w2+kap**2)*(piu*(w2+gd**2)/gd+(1-piu)*(w2+gu**2)/gu)


def shell_rates(r, tau_w, tau_k, b_u, b_d):
    """Widths (from residence times tau_w), kappa (from tau_k and flip shares b_u, b_d) and the up fraction on the
    grid, then averaged over each pair shell."""
    f_u, f_d = vx.detailed_balance(b_u, b_d, tau_k, r.pi_u)                    # makes pi_u stationary
    gu, gd, kap = 1/tau_w[0], 1/tau_w[1], f_u/tau_k[0]+f_d/tau_k[1]
    nb = len(r.edges)-1
    sh = dict(gu=np.ones(nb), gd=np.ones(nb), kap=np.ones(nb), piu=np.full(nb, .5))   # empty shells: no correction
    for b in range(nb):
        m = (r.grid >= r.edges[b]) & (r.grid < r.edges[b+1])
        if m.any():
            sh['gu'][b], sh['gd'][b], sh['kap'][b], sh['piu'][b] = gu[m].mean(), gd[m].mean(), kap[m].mean(), r.pi_u[m].mean()
    return sh, kap


def channel_factor(r, rates=None):
    """Phi_b(omega) on every shell and gap bin, and kappa on the grid. The default rates are those of the paper
    (Sec. III F): widths from the general Lambda, kappa from the Lorentzian core widths."""
    sh, kap = shell_rates(r, *(rates or (r.tau_gen, r.tau_core, r.b_up, r.b_dn)))
    return factor(r.omega[None, :], sh['gu'][:, None], sh['gd'][:, None], sh['kap'][:, None], sh['piu'][:, None]), kap


def pair_resolved(r, F=None):
    """Eq. (11), or Eq. (15) when F = Phi_b is given: D_i(t) from the binned pair weights."""
    cos = np.cos(np.outer(r.omega, r.t))
    H = r.H if F is None else r.H*F[None]
    return .5*np.abs(r.mixing@np.einsum('sbw,wt->st', H, cos))


def energy_shell(r, F=None):
    """Eq. (12): the shell weights B_ib times the shell kernels K_b (with Phi_b if F is given)."""
    cos = np.cos(np.outer(r.omega, r.t))
    A = r.A if F is None else r.A*F
    K = (A@cos)/np.maximum(A.sum(1), 1e-300)[:, None]
    return .5*np.abs(r.mixing@(r.Bbase@K))


def static_ratios(r):
    """k_sp/k_up and k_dn/k_up for every preparation, at its bath filter energy."""
    hs, out = r.model['hs'], {}
    for c in r.cfg['bath_centers']:
        k = int(np.argmin(abs(r.grid-(c+hs))))
        k_up = r.b_dn[k]/r.tau_core[1, k]                     # reverse flips, down -> up
        out[c] = ((1-r.b_up[k])/r.tau_core[0, k]/k_up,        # spin-preserving scattering of an up state
                  r.b_up[k]/r.tau_core[0, k]/k_up)            # forward flips, up -> down
    return np.array([out[c] for c in r.states[:, 0]]).T
