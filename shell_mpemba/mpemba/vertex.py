"""Observable-resolved (vertex-corrected) impurity relaxation from static inputs.

The independent-coefficient contraction (wavefunctions.contract) relaxes a bare-diagonal impurity
population at the bare states' lifetime, the width of their strength profiles. That width counts every
coupling channel, but only channels that flip the impurity relax its population, and flipped weight can
return. Resumming these |V|^2 insertions (the ladder, or vertex, correction) gives for every bare-energy
shell a two-sector renewal equation,

    P_uu(t) = S_u(t) + int_0^t f_u(tau) [(1-b_u) P_uu + b_u P_du](t-tau) dtau
    P_du(t) =          int_0^t f_d(tau) [b_d P_uu + (1-b_d) P_du](t-tau) dtau,     f_s = -dS_s/dt,

with P_ab the probability of the up sector at t after starting in sector a. S_s is the survival kernel of
the static strength profiles; b_s, the flip share of each bare state's coupling, comes from the Golden
Rule on the bare-basis matrix elements of V broadened by the profile widths. b equal to the equilibrium
fractions sends departed weight straight to equilibrium and recovers the independent-coefficient result;
the stationary state is made microcanonical by detailed balance. The correction vanishes when
spin-preserving scattering equals the reverse flip rate. No trajectory enters any forecast.
"""
import numpy as np

SHELL_STEP, SMOOTH = 0.05, 0.25


def renewal(t, S_u, S_d, b_u, b_d):
    """Solve the two-sector renewal equations on a uniform grid t; S_*: (nshell, nt), b_*: (nshell,).

    Riemann-Stieltjes trapezoid rule, int f(tau) X(t_n - tau) ~ sum_j dS_j (X_{n-j}+X_{n-j+1})/2,
    implicit in X_n. Returns P_uu, P_du with shape (nshell, nt).
    """
    S_u, S_d = np.atleast_2d(S_u), np.atleast_2d(S_d)
    ns, nt = S_u.shape
    b_u, b_d = np.broadcast_to(b_u, (ns,)), np.broadcast_to(b_d, (ns,))
    dSu, dSd = S_u[:, :-1]-S_u[:, 1:], S_d[:, :-1]-S_d[:, 1:]
    A, B = np.zeros((ns, nt)), np.zeros((ns, nt))
    A[:, 0] = S_u[:, 0]
    for n in range(1, nt):
        j = np.arange(2, n+1)
        Am, Bm = (A[:, n-j]+A[:, n-j+1])/2, (B[:, n-j]+B[:, n-j+1])/2
        ru = S_u[:, n]+np.einsum('sj,sj->s', dSu[:, j-1], (1-b_u)[:, None]*Am+b_u[:, None]*Bm)
        rd = np.einsum('sj,sj->s', dSd[:, j-1], b_d[:, None]*Am+(1-b_d)[:, None]*Bm)
        hu, hd = dSu[:, 0]/2, dSd[:, 0]/2
        ru += hu*((1-b_u)*A[:, n-1]+b_u*B[:, n-1])
        rd += hd*(b_d*A[:, n-1]+(1-b_d)*B[:, n-1])
        a11, a12, a21, a22 = 1-hu*(1-b_u), -hu*b_u, -hd*b_d, 1-hd*(1-b_d)
        det = a11*a22-a12*a21
        A[:, n] = (ru*a22-a12*rd)/det
        B[:, n] = (a11*rd-a21*ru)/det
    return A, B


def stationary_up(tau_u, tau_d, b_u, b_d):
    """Long-time up occupation of the semi-Markov renewal; tau_s = int_0^inf S_s dt."""
    return b_d*tau_u/(b_d*tau_u+b_u*tau_d)


def detailed_balance(b_u, b_d, tau, pi_u):
    """Keep each shell's flip conductance and make the stationary state the microcanonical pi_u."""
    Wud, Wdu = b_u/tau[0], b_d/tau[1]
    G = np.sqrt(Wud*pi_u*Wdu*(1-pi_u))
    return np.clip(G/pi_u*tau[0], 0, 1), np.clip(G/(1-pi_u)*tau[1], 0, 1)


def lorentz(x, w):
    return w/(np.pi*(x*x+w*w))


def channel_rates(e_up, e_dn, Vuu, Vdd, Vud, centers, gamma, block=512):
    """Golden-Rule rates from every bare state into its own ('uu','dd') and the other sector ('ud','du').

    V blocks are bare-basis matrix elements (Vud: up rows, down columns); diagonal shifts are excluded.
    Lorentzian broadening with half-width gamma_s + gamma_s', interpolated from (centers, gamma[s]).
    """
    gu = lambda e: np.interp(e, centers, gamma[0])
    gd = lambda e: np.interp(e, centers, gamma[1])
    nb = len(e_up)
    W = {k: np.zeros(nb) for k in ('uu', 'ud', 'du', 'dd')}
    for j in range(0, nb, block):
        sl = slice(j, min(j+block, nb))
        r = np.arange(sl.stop-sl.start)
        eu, ed = e_up[sl], e_dn[sl]
        x = np.asarray(Vuu[sl])**2*lorentz(eu[:, None]-e_up[None, :], 2*gu(eu)[:, None])
        x[r, r+sl.start] = 0.
        W['uu'][sl] = 2*np.pi*x.sum(1)
        W['ud'][sl] = 2*np.pi*(np.asarray(Vud[sl])**2*lorentz(eu[:, None]-e_dn[None, :], (gu(eu)+gd(eu))[:, None])).sum(1)
        x = np.asarray(Vdd[sl])**2*lorentz(ed[:, None]-e_dn[None, :], 2*gd(ed)[:, None])
        x[r, r+sl.start] = 0.
        W['dd'][sl] = 2*np.pi*x.sum(1)
        W['du'][sl] = 2*np.pi*(np.asarray(Vud[:, sl]).T**2*lorentz(ed[:, None]-e_up[None, :], (gu(ed)+gd(ed))[:, None])).sum(1)
    return W


def to_grid(e, vals, grid):
    """Gaussian-weighted average (width SMOOTH) of per-bare-state values on the shell grid."""
    out = np.empty(len(grid))
    for i in range(0, len(grid), 200):
        K = np.exp(-.5*((grid[i:i+200, None]-e[None, :])/SMOOTH)**2)
        out[i:i+200] = (K @ vals)/np.maximum(K.sum(1), 1e-300)
    return out


def density(e, grid):
    K = np.exp(-.5*((grid[:, None]-e[None, :])/SMOOTH)**2)/(np.sqrt(2*np.pi)*SMOOTH)
    return K.sum(1)


def flip_share(W, s):
    num, other = (W['ud'], W['uu']) if s == 0 else (W['du'], W['dd'])
    return np.divide(num, num+other, out=np.ones_like(num), where=num+other > 0)


def survival_kernels(prof, kind, grid, times):
    """S_s(eps, t) and mean residence tau_s(eps) from calibrated profiles (wavefunctions.calibrate)."""
    c = prof['centers']
    S, tau = np.zeros((2, len(grid), len(times))), np.zeros((2, len(grid)))
    if kind == 'lorentzian':
        for s in (0, 1):
            g = np.interp(grid, c, prof['gamma'][s])
            S[s], tau[s] = np.exp(-2*g[:, None]*times[None, :]), 1/(2*g)
        return S, tau
    edges = prof['detuning_edges']
    x = (edges[:-1]+edges[1:])/2
    dx = x[1]-x[0]
    j = np.clip(np.searchsorted(c, grid)-1, 0, len(c)-2)
    f = np.clip((grid-c[j])/(c[j+1]-c[j]), 0, 1)
    for s in (0, 1):
        m = prof['general'][s]
        gr, gi = np.zeros((len(c), len(times))), np.zeros((len(c), len(times)))
        for k in range(0, len(times), 50):
            ts = slice(k, k+50)
            gr[:, ts] = m @ np.cos(np.outer(x, times[ts]))
            gi[:, ts] = -(m @ np.sin(np.outer(x, times[ts])))
        S[s] = ((1-f)[:, None]*gr[j]+f[:, None]*gr[j+1])**2+((1-f)[:, None]*gi[j]+f[:, None]*gi[j+1])**2
        mm = (1-f)[:, None]*m[j]+f[:, None]*m[j+1]
        tau[s] = np.pi*(mm**2).sum(1)/dx                  # Parseval: int_0^inf |g|^2 dt = pi int F^2
    return S, tau


def hat(e, w, grid):
    i = np.clip(np.searchsorted(grid, e)-1, 0, len(grid)-2)
    f = np.clip((e-grid[i])/(grid[i+1]-grid[i]), 0, 1)
    return np.bincount(i, w*(1-f), len(grid))+np.bincount(i+1, w*f, len(grid))


def predict(grid, S, tau, b_u, b_d, preps, mixing, times):
    """Signed z deviations of the mixed preparations; preps = [(bare energies, spin, weights), ...]."""
    A, B = renewal(times, S[0], S[1], b_u, b_d)
    zeq = 2*stationary_up(tau[0], tau[1], b_u, b_d)-1
    dev = [hat(e, w, grid) @ (2*(A if s == 0 else B)-1-zeq[:, None]) for e, s, w in preps]
    return mixing @ np.array(dev)


def master_inputs(e_up, e_dn, V, grid, times, iters=4):
    """Golden-Rule master equation with no profile input: self-consistent widths gamma = Gamma/2."""
    gam = np.full((2, len(grid)), .05)
    for _ in range(iters):
        W = channel_rates(e_up, e_dn, *V, grid, gam)
        Gu, Gd = to_grid(e_up, W['uu']+W['ud'], grid), to_grid(e_dn, W['dd']+W['du'], grid)
        gam = np.array([Gu, Gd])/2
    S = np.array([np.exp(-np.outer(Gu, times)), np.exp(-np.outer(Gd, times))])
    return S, np.array([1/Gu, 1/Gd]), to_grid(e_up, flip_share(W, 0), grid), to_grid(e_dn, flip_share(W, 1), grid)


def shell_grid(preps):
    occ = np.concatenate([e[w > 1e-12*w.max()] for e, s, w in preps])
    return np.arange(occ.min()-.5, occ.max()+.5+1e-9, SHELL_STEP)


def forecast(e_up, e_dn, V, prof, preps, mixing, times, centres=(), h=0.):
    """All static forecasts: 'vertex_*', 'zero_vertex_*' (general, lorentzian) and 'master', as distances.

    Also returns, at each bath centre, the flip shares and the ratio of spin-preserving scattering to the
    reverse flip rate (1 means the independent-coefficient result needs no correction).
    """
    grid = shell_grid(preps)
    W = channel_rates(e_up, e_dn, *V, prof['centers'], prof['gamma'])
    bu, bd = to_grid(e_up, flip_share(W, 0), grid), to_grid(e_dn, flip_share(W, 1), grid)
    rho_u, rho_d = density(e_up, grid), density(e_dn, grid)
    pi_u = rho_u/(rho_u+rho_d)
    Sm, taum, bum, bdm = master_inputs(e_up, e_dn, V, grid, times)
    curves = {'master': .5*np.abs(predict(grid, Sm, taum, *detailed_balance(bum, bdm, taum, pi_u), preps, mixing, times))}
    statics = []
    for kind in ('general', 'lorentzian'):
        S, tau = survival_kernels(prof, kind, grid, times)
        for label, (b1, b2) in (('vertex', detailed_balance(bu, bd, tau, pi_u)),
                                ('zero_vertex', detailed_balance(1-pi_u, pi_u, tau, pi_u))):
            curves[f'{label}_{kind}'] = .5*np.abs(predict(grid, S, tau, b1, b2, preps, mixing, times))
        if kind == 'lorentzian':
            for c in centres:
                k = int(np.argmin(abs(grid-(c+h))))
                statics.append(dict(bath_centre=float(c), flip_share_up=float(bu[k]), flip_share_down=float(bd[k]),
                                    spin_preserving_over_reverse_flip=float(((1-bu[k])/tau[0, k])/(bd[k]/tau[1, k]))))
    return curves, statics


def chain_inputs(p, eb, ub, basis, block=128):
    """Bare-basis V blocks and preparation tuples for the impurity chain of model.full_eigensystem."""
    from .model import bath_link_operators
    Ze, Le = bath_link_operators(ub, p['link'], block)
    nb, hs = len(eb), p['hs']
    preps = [((hs if i % 2 == 0 else -hs)+eb, i % 2, b[:nb] if i % 2 == 0 else b[nb:]) for i, b in enumerate(basis)]
    return (p['g']*p['eta']*Ze, -p['g']*p['eta']*Ze, p['g']*Le), preps


def notes_formula(E, U, basis, mixing, initial, times, block=1024):
    """The collaborators' formula: D_i(0)|sum_mu p_i,mu e^{iE_mu t}|^2 with exact populations, no window."""
    p = np.zeros((len(basis), len(E)))
    for j in range(0, len(E), block):
        sl = slice(j, min(j+block, len(E)))
        p[:, sl] = basis @ np.asarray(U[:, sl])**2
    p = mixing @ p
    chi = p @ np.exp(1j*np.outer(E, times))
    return initial[:, None]*np.abs(chi)**2
