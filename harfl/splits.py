"""Split and partition logic. Numpy only, so it can be tested without torch."""
import numpy as np


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def holdout(d, n, rng):
    # validation set made of whole subjects or groups, never random windows or clips
    m = np.isin(d['g'], rng.choice(np.unique(d['g']), n, replace=False))
    return subset(d, ~m), subset(d, m)


def group_folds(g, k, rng):
    # boolean masks of held-out samples, one per fold, with every group in exactly one fold
    groups = rng.permutation(np.unique(g))
    return [np.isin(g, part) for part in np.array_split(groups, k)]


def partition(y, g, K, mode, alpha, rng):
    if mode == 'subject':  # whole subjects per client, a natural non-IID split
        subs = np.array_split(rng.permutation(np.unique(g)), K)
        return [np.where(np.isin(g, s))[0] for s in subs]
    while True:  # Dirichlet label skew, redrawn until no client is nearly empty
        parts = [[] for _ in range(K)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(K))) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k] += chunk.tolist()
        if min(map(len, parts)) >= 10:
            return [np.array(p) for p in parts]


def random_split(n, rng, test=0.2, val=0.1):
    # the leaky protocol of the original paper: windows or clips shuffled regardless of subject or group
    perm = rng.permutation(n)
    n_te, n_va = int(test * n), int(val * n)
    return perm[n_te + n_va:], perm[n_te:n_te + n_va], perm[:n_te]
