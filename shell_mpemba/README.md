# shell_mpemba

The code that makes the figures of our joint paper draft (*Quantum Mpemba effect in chaotic systems*), with its input data, and a note comparing it with your typicality code (github.com/VenelinPavlov/Quantum-Mpemba-Effect). Equation and figure numbers are those of the joint paper ("the paper"); in the note, "the draft" is your typicality draft.

## Quick start

Python 3.10 with NumPy, SciPy and Matplotlib (`requirements.txt`). From this folder:

```
python code/make_figures.py                                                  # all figures, from code/data (seconds)
python code/compare_pipelines.py --their-spectrum path/to/H_spectrum_14.npz  # every number in the note (6 min the first time)
```

The data scripts in `code/` recompute `code/data`: the chain data from the runs in `chain_runs/`, the Ising data from scratch. Each takes 3–12 min and up to 7 GiB of memory, so run them one at a time. Eigensystems and rebuilt Λ are cached in `cache/` (about 8 GB once every script has run; safe to delete).

## Files

**`CONSISTENCY_NOTE.pdf`** (and `.tex`). Four pages comparing this code with yours. It lists what agrees, the two choices that differ (the equilibrium reference and the energy window, with a drop-in replacement for `rhoA_DE`), why the typicality curves of your Fig. 5 and our Fig. 5a differ, and where the GOE region of Fig. S1(b) sits in the spectrum.

### `code/`

- **`common.py`**: the paths, and the definitions every script shares. These are the relaxation time (the first time D falls to D(0)/e), the first crossing of two curves, the relative error, the typicality formula, Eq. (1), and the windowed formula, Eq. (2), with its best-fit window.
- **`chain_theory.py`**: the predictions for the impurity chain, from static inputs only. It reads a run's inputs and computes pair-resolved (11), energy-shell (12) and channel-resolved (15) relaxation, detuned typicality (10), and the rate ratios k_sp/k_↑.
- **`chain_data.py`**: the data for Fig. 3 (the chain at η = 0.5) and Fig. S1(a). It adds typicality, the fitted windows, detuned typicality, the LDOS of two unperturbed states and the gap spectra to the stored run. Writes `data/fig_chain.npz`.
- **`sweep_data.py`**: the data for Fig. 4 and the channel-resolved curve of Fig. 3. It computes pair- and channel-resolved relaxation at the five couplings η, and the ratio of predicted to exact relaxation time of every preparation. Writes `data/fig_sweep.npz`.
- **`regime_data.py`**: the data for Fig. 6. Panel (a) tests when typicality applies, using the N = 13 chain with bath filters down to width 0.02. Panel (b) uses random-matrix baths with a spin-preserving share p of the coupling. Writes `data/fig_regimes.npz`.
- **`ising_data.py`**: the data for Fig. 5 and Fig. S1(b), from scratch. It builds and diagonalizes your N = 14 Ising Hamiltonian, then computes the exact distances, typicality, fitted windows, the averaged propagator (16), the LDOS widths and the level statistics. Writes `data/fig_ising.npz`.
- **`make_figures.py`**: draws the six figures into `figures/` from `data/`. It does no heavy computation.
- **`compare_pipelines.py`**: applies your pipeline's choices and ours to the same exact Ising dynamics. It prints every number in the note, including a test of the suggested `rhoA_DE` code.
- **`data/`**: the four data files behind the figures, as written by the scripts above.

### `figures/`

The figure PDFs:
- `fig_mechanism.pdf`: Fig. 1;
- `fig_chain.pdf`: Fig. 3;
- `fig_sweep.pdf`: Fig. 4;
- `fig_ising.pdf`: Fig. 5;
- `fig_regimes.pdf`: Fig. 6;
- `sm_fig_support.pdf`: Fig. S1.

Fig. 2 is drawn in TikZ inside the paper source.

### `mpemba/`

The simulation package that produced the chain runs. Only the modules the scripts import are included, and the scripts use a few functions of each:
- **`model.py`**: builds the impurity chain and caches its eigensystems (`bath_eigensystem`, `full_eigensystem`).
- **`lorentzian.py`**: the preparations of the chain (`preparations`) and the Lorentzian fit of an LDOS (`fit_histogram`). The rest is the original run pipeline.
- **`wavefunctions.py`**: the general Λ from the LDOS profiles (`construct_variance`) and the binned pair weights (`pair_histogram`, `fourier_moments`).
- **`vertex.py`**: the Golden-Rule rates, flip shares and residence times behind the channel correction (Sec. III F).
- **`reference.py`**: the random-matrix test bed of Fig. 6(b) (`run`).
- **`crossings.py`**: the Mpemba-crossing criteria, imported by `lorentzian.py` and `reference.py`.

Older names inside the package and the run files:
- "forecast" means a stored prediction;
- "contraction" means the pair weights of Eq. (11);
- "vertex" means the channel correction;
- the "general" Λ is the Λ built from the measured LDOS profiles.

### `chain_runs/`

The inputs of the chain figures (Figs. 3, 4 and 6): eight simulation runs of the impurity chain, read by `chain_data.py`, `sweep_data.py` and `regime_data.py` and by nothing else. Drawing the figures and checking the note do not need this folder. Each run keeps only the files the scripts use:
- the settings, `forecast_locked.json` (`forecasts_locked.json` in the `sweep_*` runs);
- the LDOS profiles, `envelopes.npz`;
- the binned pair weights, `contraction_general.npz`;
- the exact dynamics, `exact/validation.npz`;
- stored predictions.

| Run | Chain |
|---|---|
| `pure_flip_pilot_N14` | N = 14, η = 0 |
| `sweep_N14_eta0.25`, `sweep_N14_eta0.75`, `sweep_N14_eta1` | N = 14, η = 0.25, 0.75 and 1 |
| `selected_N14_width0.75` | N = 14, η = 0.5 |
| `selected_N13_width*` | N = 13, Fig. 6(a) |

The general Λ (2 GB per run) is not included; the scripts rebuild it from the profiles in 5–10 min. Your eigendecomposition `H_spectrum_14.npz` is not included either. `compare_pipelines.py` takes its path as an argument, only to check the eigenvalues.
