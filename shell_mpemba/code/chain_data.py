"""Data for Fig. 3 (the impurity chain at eta = 0.5) and Fig. S1(a).

Input: the chain run selected_N14_width0.75, which holds the exact dynamics, the LDOS profiles, the binned pair
weights and the energy-shell prediction, Eq. (12). This script adds the typicality formula, Eq. (1), the best-fit
windows of Eq. (2), detuned typicality, Eq. (10), the LDOS of one unperturbed state for each displayed preparation,
and the gap spectra of Fig. S1(a).
    python chain_data.py            ->  data/fig_chain.npz
It needs the full eigensystem (cache/spin_chain) and the general Lambda (cache/lambda), both rebuilt if absent:
about 11 min and under 7 GiB the first time, a few minutes after.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

from common import CACHE, DATA, RUNS, best_window, relative_error, typicality, windowed
import chain_theory as ct
from mpemba.lorentzian import preparations
from mpemba.model import full_eigensystem

CASE = RUNS/'selected_N14_width0.75'


def main():
    cfg, numerics = ct.run_config(CASE)
    E, U, Eb, E0 = full_eigensystem(cfg['model'], CACHE/'spin_chain', numerics['block_columns'])
    basis, mixing, states = preparations(Eb, cfg['bath_centers'], cfg['bath_width'], cfg['polarizations'])
    v = mixing@basis                                    # unperturbed populations (rho_i)_alpha,alpha, 16 preparations
    q = ct.energy_populations(v, U)                     # energy populations (rho_i)_mu,mu
    del U

    stored = np.load(CASE/'wavefunction/forecast.npz')                  # the run's stored predictions
    pair_weights = np.load(CASE/'wavefunction/contraction_general.npz')
    t = stored['times']
    D = ct.exact_distances(CASE)
    shell = stored['general_factorized_raw']                            # energy-shell relaxation, Eq. (12)

    typ = typicality(D, q, E, t)
    windows = np.array([windowed(D[i, 0], q[i], E, t, *best_window(E, q[i], D[i], t)) for i in range(len(D))])
    L = ct.general_lambda(CASE, E, E0)                  # Lambda[alpha, mu]
    detuned = ct.detuned_typicality(D, v, L, E, E0, t)

    # Fig. 3(d): LDOS Lambda(mu, alpha) of the most populated unperturbed state of states 2 and 15
    top = [int(np.argmax(v[i])) for i in (2, 15)]
    ldos, ldos_energy = np.array([np.asarray(L[a]) for a in top]), E0[top]

    # Fig. S1(a): gap spectra of state 2 (spin up with weight 0.875 in the first bath filter):
    # the pair weights of Eq. (11), and the products of energy populations that enter Eq. (1)
    omega = pair_weights['omega']
    dw = omega[1]-omega[0]
    mass = (.875*pair_weights['state_spectrum'][0]+.125*pair_weights['state_spectrum'][1]).sum(0)
    signal = gaussian_filter1d(mass, .025/dw, mode='reflect')/(mass.sum()*dw)
    dx = .005
    edges = np.arange(np.floor(E.min()/dx)*dx, E.max()+2*dx, dx)
    pop = np.histogram(E, edges, weights=q[2])[0]
    auto = correlate(pop, pop, mode='full', method='fft')[len(pop)-1:]
    auto[0] -= np.sum(q[2]**2)                          # remove the diagonal mu = nu
    pmass = 2*auto
    pmass[0] = auto[0]
    population = gaussian_filter1d(pmass, .025/dx, mode='reflect')/(pmass.sum()*dx)

    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA/'fig_chain.npz', t=t, states=states, E=E, q=q, exact=D, typicality=typ,
                        windows=windows, detuned=detuned, shell=shell, B=stored['general_B'],
                        K=pair_weights['K'], energy_edges=stored['energy_edges'],
                        ldos=ldos, ldos_energy=ldos_energy, signal_omega=omega, signal_density=signal,
                        population_omega=np.arange(len(pmass))*dx, population_density=population)
    print(f'wrote {DATA/"fig_chain.npz"}; relative errors: typicality {100*relative_error(typ, D):.1f}%, '
          f'detuned typicality {100*relative_error(detuned, D):.1f}%, energy-shell relaxation {100*relative_error(shell, D):.1f}%')


if __name__ == '__main__':
    main()
