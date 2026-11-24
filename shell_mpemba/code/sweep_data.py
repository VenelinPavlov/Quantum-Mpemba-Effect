"""Data for Fig. 4 (the chain at five couplings) and the channel-resolved curves of Fig. 3.

For each coupling eta (run folders in chain_theory.CASES), from static inputs only:
  - pair-resolved relaxation, Eq. (11), and channel-resolved relaxation, Eq. (15);
  - the static ratios k_sp/k_up and k_dn/k_up at each preparation's bath filter energy;
  - the ratio of predicted to exact relaxation time of every preparation.
    python sweep_data.py            ->  data/fig_sweep.npz        (about 6 min for the five couplings, 3 GiB)
"""
import numpy as np

from common import DATA, RUNS, relative_error, relaxation_time
import chain_theory as ct


def main():
    rows = {k: [] for k in ('eta', 'ksp_over_kup', 'kdn_over_kup', 'ratio_pair', 'ratio_channel')}
    for eta, name in ct.CASES.items():
        r = ct.load(RUNS/name)
        F, _ = ct.channel_factor(r)
        pair, channel = ct.pair_resolved(r), ct.pair_resolved(r, F)
        ksp, kdn = ct.static_ratios(r)
        t_exact = np.array([relaxation_time(r.t, d) for d in r.exact])
        ratio_pair = np.array([relaxation_time(r.t, d) for d in pair])/t_exact
        ratio_channel = np.array([relaxation_time(r.t, d) for d in channel])/t_exact
        print(f'eta={eta}: k_sp/k_up {ksp.min():.2f}-{ksp.max():.2f}, k_dn/k_up {kdn.min():.2f}-{kdn.max():.2f} | '
              f'pair-resolved {100*relative_error(pair, r.exact):.1f}% (median ratio {np.median(ratio_pair):.2f}), '
              f'channel-resolved {100*relative_error(channel, r.exact):.1f}% (median {np.median(ratio_channel):.2f}, '
              f'range {ratio_channel.min():.2f}-{ratio_channel.max():.2f})', flush=True)
        rows['eta'] += [eta]*len(r.exact)
        rows['ksp_over_kup'] += list(ksp)
        rows['kdn_over_kup'] += list(kdn)
        rows['ratio_pair'] += list(ratio_pair)
        rows['ratio_channel'] += list(ratio_channel)
        if eta == .5:                                       # the curves drawn in Fig. 3
            curves = dict(t_half=r.t, pair_half=pair, channel_half=channel)
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA/'fig_sweep.npz', independent_test=ct.INDEPENDENT_TEST,
                        **{k: np.array(v) for k, v in rows.items()}, **curves)
    print(f'wrote {DATA/"fig_sweep.npz"}')


if __name__ == '__main__':
    main()
