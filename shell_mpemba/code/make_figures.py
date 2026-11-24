"""Draw the paper's figures from the data files in data/ (seconds; no heavy computation).
    python make_figures.py
writes, in the folder figures/ next to this one,
    fig_mechanism.pdf   Fig. 1, schematic with illustrative parameters
    fig_chain.pdf       Fig. 3, the chain at eta = 0.5        (data/fig_chain.npz, data/fig_sweep.npz)
    fig_sweep.pdf       Fig. 4, the chain at five couplings   (data/fig_sweep.npz)
    fig_ising.pdf       Fig. 5, the long-range Ising chain    (data/fig_ising.npz)
    fig_regimes.pdf     Fig. 6, the two limits                (data/fig_regimes.npz)
    sm_fig_support.pdf  Fig. S1 of the supplement             (data/fig_chain.npz, data/fig_ising.npz)
Each method keeps its colour and line style in every figure (the three uncorrected static predictions,
energy-shell, pair-resolved and the averaged propagator, share blue); states are drawn in neutral greys.
Equation numbers in the labels are those of the paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch      # noqa: E402
from scipy.ndimage import gaussian_filter1d, uniform_filter1d   # noqa: E402

from common import DATA, FIGURES, first_crossing, relaxation_time   # noqa: E402
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9,
                     'axes.labelsize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42})
# method colours (checked for colour-vision deficiency: every pair differs by >= 8 OKLab units under
# protan/deutan/tritan simulation and >= 15 for normal vision)
TYP, WIN, DET, CW, CHAN = '#e0a526', '#cb5b2e', '#239477', '#2466ae', '#8a2a6b'
METHOD = {'exact': ('Exact', 'black', '-'),
          'typicality': ('Typicality (1)', TYP, (0, (4, 2))),
          'windows': ('Window (2), fitted', WIN, (0, (1, 1.4))),
          'detuned': ('Detuned typicality (10)', DET, (0, (6, 1.5, 1.5, 1.5))),
          'shell': ('Energy-shell (12)', CW, (0, (3, 1.2))),
          'pair': ('Pair-resolved (11)', CW, (0, (3, 1.2))),
          'channel': ('Channel-resolved (15)', CHAN, (0, (5, 1.2))),
          'propagator': ('Averaged propagator (16)', CW, (0, (3, 1.2)))}
GREYS = ['#222222', '#7a7a7a', '#b0b0b0']           # states, darkest first


def panel_label(ax, label):
    """The bold (a), (b), ... in the corner; the captions describe the panels."""
    ax.set_title(f'({label})', loc='left', pad=4, fontweight='bold')


def panel(ax, label, xlabel='Time'):
    """Panel label, x label and a faint grid."""
    panel_label(ax, label)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=.13)


def line(ax, key, x, y, **kw):
    """One curve in the colour and line style of its method (METHOD)."""
    lab, c, ls = METHOD[key]
    return ax.plot(x, y, color=c, ls=ls, lw=1.6 if key == 'exact' else 1.4, label=kw.pop('label', lab), **kw)


def fig_mechanism():
    """Schematic, illustrative parameters. Preparation 1 (farther) dark, preparation 2 (nearer) mid grey."""
    P1, P2 = GREYS[0], GREYS[1]
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.75), layout='constrained', width_ratios=[1.15, 1, 1.05])
    # (a) LDOS of unperturbed states, with a width that depends on energy
    ax = axs[0]
    E = np.linspace(0, 10, 1200)
    gam = lambda e: .2+.05*e
    for ea in np.arange(.9, 10, 1.3):
        ax.plot(E, (gam(ea)/np.pi)/((E-ea)**2+gam(ea)**2), color='.8', lw=.8)
    for ea, c, k in [(2.6, P2, 2), (7.4, P1, 1)]:
        g = gam(ea)
        ax.plot(E, (g/np.pi)/((E-ea)**2+g**2), color=c, lw=1.8)
        y = .5/(np.pi*g)
        ax.annotate('', xy=(ea-g, y), xytext=(ea+g, y), arrowprops=dict(arrowstyle='<->', color=c, lw=1))
        ax.text(ea+g+.1, y+.07, rf'$2\Gamma(E_{k})$', va='bottom', fontsize=8)
    for ec, c, lab in [(2.6, P2, 'Preparation 2: nearer'), (7.4, P1, 'Preparation 1: farther')]:
        ax.fill_between(E, -.55, -.55+.42*np.exp(-(E-ec)**2/(2*.55**2)), color=c, alpha=.3, lw=0)
        ax.text(ec, -.62, lab, ha='center', va='top', fontsize=7.5)
    ax.axhline(-.55, color='.5', lw=.6)
    ax.text(.15, 1.75, r'LDOS $\Lambda(\mu,\alpha)$ of unperturbed states $|\phi_\alpha\rangle$', fontsize=7.5, va='top')
    ax.text(.15, -.08, r'Weights $(\rho_i)_{\alpha\alpha}$', fontsize=7.5, va='top')
    ax.set(xlim=(0, 10), ylim=(-.95, 1.8), xticks=[], yticks=[], xlabel=r'Unperturbed energy $E_\alpha$')
    panel_label(ax, 'a')
    # (b) two decay rates: the farther preparation decays faster and overtakes
    ax = axs[1]
    t = np.linspace(0, 6, 400)
    d1, d2, g1, g2 = .45, .36, .13, .06
    tc = np.log(d1/d2)/(2*(g1-g2))
    ax.semilogy(t, d1**2*np.exp(-4*g1*t), color=P1, lw=1.8, label=r'Preparation 1: larger $D(0)$, larger $\Gamma$')
    ax.semilogy(t, d2**2*np.exp(-4*g2*t), color=P2, lw=1.8, label=r'Preparation 2: smaller $D(0)$, smaller $\Gamma$')
    ax.axvline(tc, color='.45', lw=.8, ls=':')
    ax.text(tc+.12, .2, r'$t_c=\dfrac{\ln[D_1(0)/D_2(0)]}{2[\Gamma(E_1)-\Gamma(E_2)]}$', fontsize=8, va='top')
    ax.set(xlim=(0, 6), ylim=(.04, .3), xlabel='Time', ylabel=r'$D_i^2(t)$, log scale', yticks=[])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.legend(fontsize=6.8, loc='lower left', frameon=False)
    panel_label(ax, 'b')
    # (c) a width is a lifetime: escape channels of a spin-up unperturbed state
    ax = axs[2]
    ax.set(xlim=(0, 10), ylim=(-.75, 6.4), xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set_visible(False)
    rng = np.random.default_rng(3)
    for y0, lab in [(4.3, r'spin $\uparrow$ $\otimes$ bath states'), (1.5, r'spin $\downarrow$ $\otimes$ bath states')]:
        ax.add_patch(plt.Rectangle((.2, y0-.55), 9.6, 1.1, color='.94', lw=0))
        for x in np.linspace(.6, 9.4, 13):
            yy = y0+rng.uniform(-.3, .3)
            ax.plot([x-.18, x+.18], [yy, yy], color='.6', lw=1)
        ax.text(.25, y0+.62, lab, fontsize=7.5, va='bottom')
    start, kept, down, back = (2.0, 4.3), (5.2, 4.4), (3.8, 1.5), (7.8, 4.2)
    ax.plot(*start, 'o', color='black', ms=6, zorder=5)
    ax.text(start[0]-.15, start[1]-.25, r'$|\phi_\alpha\rangle$', ha='right', va='top', fontsize=8)
    for p, q, rad, c, ls, lab, xy, ha in [(start, kept, -.55, '.55', '--', r'$k_{\rm sp}$: spin kept', (3.6, 5.5), 'center'),
                                          (start, down, 0, 'black', '-', r'$k_\downarrow$: flipped', (3.4, 3.0), 'left'),
                                          (down, back, -.25, 'black', '-', r'$k_\uparrow$: flips back', (6.5, 2.55), 'left')]:
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=10, color=c, lw=1.3, ls=ls))
        ax.text(*xy, lab, fontsize=8, ha=ha)
    ax.text(5, .3, r'Lifetime (LDOS width): $2\Gamma=k_\downarrow+k_{\rm sp}$', ha='center', fontsize=8)
    ax.text(5, -.45, r'Relaxation rate of $Z_A$: $k_\downarrow+k_\uparrow$', ha='center', fontsize=8)
    panel_label(ax, 'c')
    fig.savefig(FIGURES/'fig_mechanism.pdf')


def fig_chain(d, s):
    t, D = d['t'], d['exact']
    pair = [2, 15]
    assert np.allclose(s['t_half'], t)
    pred = {'typicality': d['typicality'], 'windows': d['windows'], 'detuned': d['detuned'], 'shell': d['shell'],
            'channel': s['channel_half']}
    fig = plt.figure(figsize=(9.2, 5.5), layout='constrained')
    gs = fig.add_gridspec(2, 3)
    # (a) the Mpemba crossing of the displayed pair, as one ratio per method
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(1, color='.55', lw=.8)
    line(ax, 'exact', t, D[2]/D[15])
    for key in ('typicality', 'detuned', 'shell', 'channel'):
        y = pred[key]
        ok = (y[15] > .05*y[15, 0]) & (y[2] > .05*y[2, 0])            # drop the ratio once both have decayed
        line(ax, key, t[ok], y[2, ok]/y[15, ok])
    tc = first_crossing(t, D[2], D[15])
    ax.plot(tc, 1, 'o', color='black', ms=4, zorder=5)
    ax.annotate(rf'exact $t_c={tc:.2f}$', (tc, 1), (tc+1.2, 1.045), fontsize=7.5, arrowprops=dict(arrowstyle='-', lw=.6, color='.3'))
    panel(ax, 'a')
    ax.set(xlim=(0, 16), ylim=(.55, 1.1), ylabel=r'$D_2(t)/D_{15}(t)$')
    ax.legend(fontsize=6.6, loc='lower left', frameon=False)
    # (b) the typicality formulas on one state
    ax = fig.add_subplot(gs[0, 1])
    line(ax, 'exact', t, D[2]**2)
    for key in ('typicality', 'windows', 'detuned'):
        line(ax, key, t, pred[key][2]**2)
    panel(ax, 'b')
    ax.set(xlim=(0, 16), ylim=(0, .18), ylabel=r'$D_2^2(t)$')
    ax.legend(fontsize=7, loc='upper right')
    # (c) relaxation times of every preparation
    ax = fig.add_subplot(gs[0, 2])
    ex = np.array([relaxation_time(t, y) for y in D])
    x = np.arange(len(D))
    keys = ['typicality', 'windows', 'detuned', 'shell', 'channel']
    marks = dict(typicality='D', windows='s', detuned='^', shell='o', channel='o')
    for j, key in enumerate(keys):
        lab, c, _ = METHOD[key]
        r = [relaxation_time(t, yy)/e for yy, e in zip(pred[key], ex)]
        ax.plot(x+.16*(j-(len(keys)-1)/2), r, marks[key], color=c, ms=3.4, ls='', label=lab,
                mfc='white' if key == 'shell' else c, mew=1.0)
    ax.axhline(1, color='.5', lw=.7)
    for b in (3.5, 7.5, 11.5):
        ax.axvline(b, color='.85', lw=.6)
    panel(ax, 'c', 'Preparation (grouped by bath energy)')
    ax.set(ylabel='Predicted / exact relaxation time', yscale='log', ylim=(.08, 1.6), xticks=[0, 4, 8, 12, 15])
    ax.legend(fontsize=6.4, loc='center', bbox_to_anchor=(.5, .42), framealpha=.92)
    # (d) energy distributions of the displayed pair against the LDOS of one unperturbed state
    ax = fig.add_subplot(gs[1, 0])
    E, q = d['E'], d['q']
    edges = np.arange(-12, 12.01, .25)
    peak = 0.
    for i, c, xt, yt in zip(pair, GREYS, (-8.7, 5.9), (.22, .36)):
        h = np.histogram(E, edges, weights=q[i])[0]/np.diff(edges)
        peak = max(peak, h.max())
        ax.stairs(h, edges, color=c, lw=1.3)
        ax.text(xt, yt, f'State {i}', fontsize=7.5, color=c)
    fine = np.arange(-12, 12.001, .02)
    for row in d['ldos']:
        h = gaussian_filter1d(np.histogram(E, fine, weights=row)[0]/np.diff(fine), 1)
        ax.plot((fine[:-1]+fine[1:])/2, h*peak/h.max(), color=CW, lw=1.1)
    ax.text(-1.0, .53, r'LDOS $\Lambda(\mu,\alpha)$ of one unperturbed' '\n' r'state each (scaled), width $2\Gamma\approx0.15$',
            fontsize=6.8, color=CW, ha='center', va='bottom')
    m2 = q[2]@E
    s2 = np.sqrt(q[2]@E**2-m2**2)
    ax.annotate('', xy=(m2-s2, .47), xytext=(m2+s2, .47), arrowprops=dict(arrowstyle='<->', lw=1, color=GREYS[0]))
    ax.text(m2, .485, rf'$2\sigma_E={2*s2:.1f}$', ha='center', fontsize=7.5)
    panel(ax, 'd', r'Eigenenergy $E_\mu$')
    ax.set(ylabel=r'Population density $(\rho_i)_{\mu\mu}$', xlim=(-9, 9), ylim=(0, .64))
    # (e) decay rates of the shell kernels against energy, with the exact rates of all preparations
    ax = fig.add_subplot(gs[1, 1:])
    eb = d['energy_edges']
    centers = (eb[:-1]+eb[1:])/2
    rate = np.array([1/relaxation_time(t, k) if k[0] > 0 else np.nan for k in d['K']])
    ok = np.isfinite(rate) & (np.abs(centers) < 12)
    ax.plot(centers[ok], rate[ok], '-', color=CW, lw=1.5, label=r'Static shell kernels $\widehat K_b$, Eq. (12)')
    B = d['B']
    Bm = (B@centers)/B.sum(1)                                     # mean energy of the (signed) shell weights
    share = np.clip(B, 0, None)/np.clip(B, 0, None).sum(1, keepdims=True)
    Bs = np.sqrt(np.clip(share@centers**2-(share@centers)**2, 0, None))   # their spread, for the bars
    ax.plot(Bm, 1/ex, 'o', color='black', ms=3.6, label='Exact decay rate of each preparation,\nat the mean energy of its shell weights')
    for i, c in zip(pair, GREYS):
        ax.errorbar(Bm[i], 1/ex[i], xerr=Bs[i], fmt='o', mfc='white', mec=c, ecolor=c, ms=8, mew=1.4, elinewidth=1.2, capsize=2, zorder=4)
        ax.text(Bm[i]+(0 if i == 2 else .55), 1/ex[i]+(.0022 if i == 2 else .0012), f'state {i}', ha='center' if i == 2 else 'left', fontsize=7.5, color=c)
    panel(ax, 'e', r'Mean pair energy $\bar E$')
    ax.set(ylabel=r'Decay rate $1/t_i$', xlim=(-9, 9), ylim=(.12, .165))
    ax.legend(fontsize=7, loc='lower left', frameon=False)
    fig.savefig(FIGURES/'fig_chain.pdf')


def fig_sweep(s):
    """Predicted over exact relaxation time against the static k_sp/k_up, every preparation at five couplings."""
    eta, x = s['eta'], s['ksp_over_kup']
    fig, ax = plt.subplots(figsize=(3.4, 2.75), layout='constrained')
    xx = np.linspace(0, 3.8, 200)
    ax.plot(xx, 2/(1+xx), color='.6', lw=1.2, zorder=1, label=r'$2/(1+k_{\rm sp}/k_\uparrow)$, no fit')
    ax.axhline(1, color='.82', lw=.8, zorder=0)
    rng = np.random.default_rng(0)
    jit = .03*rng.standard_normal(len(x))                 # small horizontal jitter separates preparations
    ax.plot(x+jit, s['ratio_pair'], 'o', color=CW, mfc='white', ms=3.6, mew=.9, ls='', label=METHOD['pair'][0])
    ax.plot(x+jit, s['ratio_channel'], 'o', color=CHAN, ms=3.6, ls='', label=METHOD['channel'][0])
    ax.text(np.median(x[eta < .3]), 2.65, r'$\eta=0,\ 0.25$', ha='center', fontsize=6.5, color='.35')
    for e in np.unique(eta[eta > .3]):
        ax.text(np.median(x[eta == e]), 2.65, rf'$\eta={e:g}$', ha='center', fontsize=6.5, color='.35')
    ax.set(yscale='log', ylim=(.3, 3.1), xlim=(-.2, 3.8), ylabel='Predicted / exact relaxation time',
           xlabel=r'Spin-preserving / reverse-flip rate, $k_{\rm sp}/k_\uparrow$')
    ax.set_yticks([.3, .5, 1, 2]); ax.set_yticklabels(['0.3', '0.5', '1', '2'])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(fontsize=6.3, loc='lower left', frameon=False)
    ax.grid(alpha=.13)
    fig.savefig(FIGURES/'fig_sweep.pdf')


def fig_ising(d):
    t, D, P = d['t'], d['exact'], d['propagator']
    frac = d['phi_over_2pi']
    fig = plt.figure(figsize=(9.2, 2.85), layout='constrained')
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, .82, 1.18])
    # (a) the inversion D^2(0) - D^2(0.05) at early times
    ax = fig.add_subplot(gs[0])
    ax.axhline(0, color='.6', lw=.7)
    for key, y in (('exact', D), ('typicality', d['typicality']), ('windows', d['windows']), ('propagator', P)):
        line(ax, key, t, y[0]**2-y[2]**2)
    tc = first_crossing(t, D[0], D[2])
    ax.plot(tc, 0, 'o', color='black', ms=3.5, zorder=5)
    panel(ax, 'a')
    ax.set(xlim=(0, 1), ylim=(-.015, .05), ylabel=r'$D^2(0)-D^2({%g})$' % frac[2])
    ax.legend(fontsize=6, loc='upper right', frameon=False)
    # (b) it persists: running time average over unit windows, raw curves faint
    ax = fig.add_subplot(gs[1])
    ax.axhline(0, color='.6', lw=.7)
    dt = t[1]-t[0]
    late = t >= 1
    for key, y in (('exact', D), ('propagator', P)):
        diff = y[0]**2-y[2]**2
        c = METHOD[key][1]
        ax.plot(t[late], 1e3*diff[late], color=c, lw=.5, alpha=.35)
        line(ax, key, t[late], 1e3*uniform_filter1d(diff, int(round(1/dt)), mode='nearest')[late])
    panel(ax, 'b')
    ax.set(xlim=(1, 20), ylim=(-3, .8), ylabel=r'$10^3\,[D^2(0)-D^2({%g})]$' % frac[2])
    ax.legend(fontsize=6.6, loc='lower right', frameon=False)
    # (c) the two widths against tilt
    ax = fig.add_subplot(gs[2])
    g = d['phi_grid']
    sel = g <= .07
    for key, c, lab, xy in (('energy_spread_grid', TYP, 'energy spread $\\sigma_E$,\nread by typicality', (.002, 1.6)),
                            ('local_rate_grid', 'black', 'local-field rate,\nfelt by the spin', (.033, 1.07))):
        y = d[key]
        ax.plot(g[sel], y[sel]/y[0], color=c, lw=1.5)
        ax.plot(frac, np.interp(frac, g, y)/y[0], 'o', color=c, ms=3.5)
        ax.text(*xy, lab, fontsize=6.6, color=c if c == 'black' else '#9a6c00', va='center')
    panel(ax, 'c', r'Tilt $\varphi/2\pi$')
    ax.set(xlim=(0, .07), ylim=(.85, 1.75), ylabel='Width / value at $\\varphi=0$', xticks=[0, .02, .04, .06])
    # (d) the LDOS width falls with energy, where the tilted states put their weight
    sub = gs[3].subgridspec(2, 1, height_ratios=[1.1, 1], hspace=.05)
    ax = fig.add_subplot(sub[0])
    axp = fig.add_subplot(sub[1], sharex=ax)
    ok = np.isfinite(d['gamma'])
    ax.semilogy(d['gamma_centers'][ok], d['gamma'][ok], 'o-', color=CW, ms=2.5, lw=1.1)
    panel_label(ax, 'd')
    ax.set(ylabel=r'LDOS $\Gamma$', xlim=(-10, 20))
    ax.tick_params(labelbottom=False)
    pe = d['population_edges']
    for i in range(3):
        above = 1-np.r_[0, np.cumsum(d['population_hist'][i])]      # weight above each energy
        axp.plot(pe, above, color=GREYS[i], lw=1.4, label=f'{frac[i]:g}')
        axp.plot(4, np.interp(4, pe, above), 'o', color=GREYS[i], ms=3)
    axp.set(xlabel=r'Energy $E$', ylabel=r'Weight above $E$', xlim=(-10, 20), ylim=(0, 1.02), yticks=[0, .5, 1])
    axp.legend(fontsize=5.8, title=r'$\varphi/2\pi$', title_fontsize=5.8, frameon=False, loc='upper right', handlelength=1.2)
    for a in (ax, axp):
        a.axvspan(4, 20, color='.5', alpha=.1, lw=0)
        a.grid(alpha=.13)
    fig.savefig(FIGURES/'fig_ising.pdf')


def fig_regimes(r):
    """(a) relaxation-time ratios against the filter width in units of the LDOS width; (b) the featureless bath with a
    spin-preserving share p: exact over pair-resolved relaxation time against 1/[2(1-p)]."""
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.6), layout='constrained')
    ax = axs[0]
    x = r['sigma_0']/r['two_gamma']                     # spread of the occupied unperturbed energies, over the LDOS width
    rng = np.random.default_rng(1)
    jit = np.exp(.04*rng.standard_normal(len(x)))
    ax.axhline(1, color='.82', lw=.8, zorder=0)
    for key, mk, c, lab in (('r_typicality', 'D', TYP, 'Typicality (1)'), ('r_detuned', '^', DET, 'Detuned typicality (10)'),
                            ('r_shell', 'o', CW, 'Energy-shell (12)')):
        ax.plot(x*jit, r[key], mk, color=c, ms=3.4, ls='', label=lab, mfc='white' if key == 'r_shell' else c, mew=.9)
    full = r['polarization'] >= 1
    ax.plot((x*jit)[full], r['r_typicality'][full], 'D', color='black', ms=6, mfc='none', mew=.8, ls='', label=r'fully polarized, $\mu_i=1$')
    ax.set(xscale='log', yscale='log', xlim=(.08, 15), ylim=(.08, 2.3), ylabel='Predicted / exact relaxation time')
    ax.set_yticks([.1, .2, .5, 1, 2]); ax.set_yticklabels(['0.1', '0.2', '0.5', '1', '2'])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    panel(ax, 'a', r'Occupied spread / LDOS width, $\sigma_0/2\Gamma$')
    ax.legend(fontsize=6.1, loc='lower left', frameon=False)
    ax = axs[1]
    pp = np.linspace(0, .92, 200)
    ax.plot(pp, 1/(2*(1-pp)), color='.6', lw=1.2, label=r'$\kappa/2\Gamma$ inverted: $1/[2(1-p)]$, no fit')
    ax.axhline(1, color='.82', lw=.8, zorder=0)
    ratio = r['t_exact']/r['t_pair']
    ax.errorbar(r['share'], ratio, yerr=r['t_exact_sem']/r['t_pair'], fmt='o', color='black', ms=4, lw=.9, capsize=2,
                label='Exact / pair-resolved (11)')
    ax.set(yscale='log', ylim=(.4, 7), xlim=(-.03, .95), ylabel='Relaxation-time ratio')
    ax.set_yticks([.5, 1, 2, 5]); ax.set_yticklabels(['0.5', '1', '2', '5'])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    panel(ax, 'b', r'Spin-preserving share $p$ of the coupling')
    ax.legend(fontsize=6.3, loc='upper left', frameon=False)
    fig.savefig(FIGURES/'fig_regimes.pdf')


def fig_support(c, i):
    """Supplement: (a) gap spectra of chain state 2; (b) Ising level statistics by energy."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), layout='constrained')
    ax = axs[0]
    ax.plot(c['population_omega'], c['population_density'], color=TYP, lw=1.4, label='Population products, Eq. (M1)')
    ax.plot(c['signal_omega'], c['signal_density'], color=CW, lw=1.4, label=r'Pair weights $\widehat C_{i,\mu\nu}$, Eq. (M11)')
    panel(ax, 'a', r'Gap $\omega$')
    ax.set(ylabel='Density (unit signed area)', xlim=(0, 3))
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]
    ok = np.isfinite(i['r_by_bin'])
    ax.plot(i['r_edges'][ok]+.5, i['r_by_bin'][ok], 'o-', color='.25', ms=3, lw=1.2)
    ax.axhline(.5307, color='.35', lw=.8, ls='--')
    ax.axhline(.3863, color='.35', lw=.8, ls=':')
    ax.text(-9.5, .535, 'GOE', fontsize=7, va='bottom')
    ax.text(-9.5, .39, 'Poisson', fontsize=7, va='bottom')
    ax.axvspan(4, 20, color='.5', alpha=.1, lw=0)
    panel(ax, 'b', r'Eigenenergy $E_\mu$')
    ax.set(xlim=(-10, 20), ylim=(.28, .6), ylabel=r'$\langle r\rangle$')
    fig.savefig(FIGURES/'sm_fig_support.pdf')


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True)
    chain, sweep, ising, regimes = (dict(np.load(DATA/f'fig_{name}.npz')) for name in ('chain', 'sweep', 'ising', 'regimes'))
    fig_mechanism()
    fig_chain(chain, sweep)
    fig_sweep(sweep)
    fig_ising(ising)
    fig_regimes(regimes)
    fig_support(chain, ising)
    print(f'wrote the figures to {FIGURES}')
