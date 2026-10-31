"""Published resampling schemes; independent implementation, NumPy PCG64 RNG."""

import numpy as np


def paired_stratified(values, strata, *, resamples=10_000, seed=42):
    """Bootstrap paired per-image differences, preserving each stratum's size."""
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    if values.ndim != 2 or not len(values) or strata.shape != (len(values),):
        raise ValueError("Require a nonempty samples-by-metrics array and aligned strata")
    if not np.isfinite(values).all() or resamples < 1:
        raise ValueError("Require finite values and positive resamples")
    rng = np.random.default_rng(seed)
    replicates = np.zeros((resamples, values.shape[1]))
    for label in sorted(set(strata)):
        group = values[strata == label]
        for start in range(0, resamples, 256):
            stop = min(start + 256, resamples)
            indices = rng.integers(len(group), size=(stop - start, len(group)))
            replicates[start:stop] += group[indices].sum(axis=1)
    replicates /= len(values)
    return {
        "estimate": values.mean(axis=0).tolist(),
        "ci95_low": np.percentile(replicates, 2.5, axis=0).tolist(),
        "ci95_high": np.percentile(replicates, 97.5, axis=0).tolist(),
    }


def cluster_flag_interval(flags, clusters, *, resamples=10_000, seed=42):
    """Sample global cluster IDs uniformly; include all descendants on every draw."""
    flags = np.asarray(flags, dtype=float)
    clusters = np.asarray(clusters)
    if flags.ndim != 1 or not len(flags) or clusters.shape != flags.shape:
        raise ValueError("Require nonempty aligned flags and cluster IDs")
    if not np.isin(flags, [0, 1]).all() or resamples < 1:
        raise ValueError("Require binary flags and positive resamples")
    groups = [flags[clusters == key] for key in sorted(set(clusters))]
    sums = np.array([group.sum() for group in groups])
    sizes = np.array([len(group) for group in groups])
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples)
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        draws = rng.integers(len(groups), size=(stop - start, len(groups)))
        replicates[start:stop] = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {
        "estimate": float(flags.mean()),
        "ci95_low": float(np.percentile(replicates, 2.5)),
        "ci95_high": float(np.percentile(replicates, 97.5)),
        "clusters": len(groups),
    }


def paired_sign_flip(differences, *, permutations=10_000, seed=42):
    """Two-sided Monte Carlo sign-flip test with a plus-one correction."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError("Require a nonempty finite vector of paired differences")
    if permutations < 1:
        raise ValueError("Require positive permutation count")
    rng = np.random.default_rng(seed)
    observed = float(differences.mean())
    extreme = 0
    for start in range(0, permutations, 256):
        stop = min(start + 256, permutations)
        signs = rng.integers(0, 2, size=(stop - start, len(differences))) * 2 - 1
        simulated = (signs * differences).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(simulated) >= abs(observed) - 1e-15))
    return {
        "estimate": observed,
        "p_two_sided": (extreme + 1) / (permutations + 1),
        "extreme_draws": extreme,
        "permutations": permutations,
        "seed": seed,
        "method": "random sign flips, two-sided, plus-one correction, PCG64",
    }
