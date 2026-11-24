"""One shared robust-crossing rule and complete pair accounting.

An event must pass initial separation, depth, persistence, decay, and distance
floor gates. No event means none in the observed window, not none at all times.
"""
from pathlib import Path
import itertools
import numpy as np
from .model import write_csv


def _qualifying_inversions(t, far, near, criteria):
    """Yield inversions passing every gate except the minimum depth."""
    delta = np.asarray(far)-np.asarray(near)
    if delta[0] < criteria['initial_gap']:
        return
    starts = np.flatnonzero((delta[:-1] > 0) & (delta[1:] <= 0))
    for j in starts:
        k = j+1
        while k+1 < len(t) and delta[k+1] <= 0:
            k += 1
        tc = t[j]-delta[j]*(t[j+1]-t[j])/(delta[j+1]-delta[j])
        tend = t[k]
        if k+1 < len(t):
            tend = t[k]-delta[k]*(t[k+1]-t[k])/(delta[k+1]-delta[k])
        depth = float(-np.min(delta[j+1:k+1]))
        if tend-tc < criteria['persistence']:
            continue
        fc = np.interp(tc, t, far)
        nc = np.interp(tc, t, near)
        if max(fc/max(far[0], 1e-15), nc/max(near[0], 1e-15)) > criteria['decay_fraction']:
            continue
        if fc < criteria['distance_floor']:
            continue
        yield dict(observed=True, time=float(tc), depth=depth, duration=float(tend-tc),
                   slope=float((delta[j+1]-delta[j])/(t[j+1]-t[j])))


def crossing(t, far, near, criteria):
    """First inversion passing the fixed depth, persistence, and decay gates.

    Absence means no qualifying event within the supplied horizon. The crossing
    time is the interpolated zero before the segment, not the later time at which
    the minimum depth is reached.
    """
    for event in _qualifying_inversions(t, far, near, criteria):
        if event['depth'] >= criteria['post_gap']:
            return event
    return dict(observed=False, time=np.nan, depth=0., duration=0., slope=np.nan)


def eligible_depth(t, far, near, criteria):
    """Largest depth passing the other gates; crossing() selects the first event."""
    return max(
        (event['depth'] for event in _qualifying_inversions(t, far, near, criteria)),
        default=0.,
    )


def eligible_pairs(initial, criteria):
    pairs = []
    for i, j in itertools.combinations(range(len(initial)), 2):
        far, near = (i, j) if initial[i] > initial[j] else (j, i)
        if initial[far]-initial[near] >= criteria['initial_gap']:
            pairs.append((far, near))
    return pairs


def evaluate_predictions(times, exact, models, initial, criteria, out):
    pairs = eligible_pairs(initial, criteria)
    rows = []
    summaries = []
    for name, pred in models.items():
        counts = dict(TP=0, FP=0, FN=0, TN=0)
        errors = []
        for far, near in pairs:
            ex = crossing(times, exact[far], exact[near], criteria)
            pr = crossing(times, pred[far], pred[near], criteria)
            key = ('TP' if ex['observed'] else 'FP') if pr['observed'] else (
                'FN' if ex['observed'] else 'TN')
            counts[key] += 1
            if key == 'TP':
                errors.append(abs(ex['time']-pr['time']))
            rows.append(dict(model=name, far=far, near=near, exact=ex['observed'], predicted=pr['observed'],
                        t_exact=ex['time'], t_predicted=pr['time'], exact_depth=ex['depth'], predicted_depth=pr['depth']))
        summaries.append(dict(model=name, pairs=len(pairs), **counts,
                              relative_curve_error=float(np.linalg.norm(
                                  pred-exact)/np.linalg.norm(exact)),
                              initial_relative_error=float(np.linalg.norm(
                                  pred[:, 0]-exact[:, 0])/np.linalg.norm(exact[:, 0])),
                              crossing_time_MAE=float(np.mean(errors)) if errors else float('nan')))
    write_csv(Path(out)/'pairs.csv', rows)
    write_csv(Path(out)/'summary.csv', summaries)
    return summaries
