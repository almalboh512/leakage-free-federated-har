import numpy as np

from harfl.splits import group_folds, holdout, partition, random_split, subset


def toy(n_groups=12, per=20, n_cls=4, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(1, n_groups + 1), per)
    return dict(x=rng.normal(size=(len(g), 3)), y=rng.integers(0, n_cls, len(g)), g=g)


def test_holdout_is_group_disjoint():
    d = toy()
    tr, va = holdout(d, 3, np.random.default_rng(1))
    assert len(np.unique(va['g'])) == 3
    assert not set(tr['g']) & set(va['g'])
    assert len(tr['y']) + len(va['y']) == len(d['y'])


def test_group_folds_cover_every_group_once():
    d = toy()
    masks = group_folds(d['g'], 3, np.random.default_rng(0))
    assert np.all(sum(m.astype(int) for m in masks) == 1)
    for m in masks:
        assert not set(d['g'][m]) & set(d['g'][~m])


def test_subject_partition_keeps_subjects_together():
    d = toy(n_groups=10)
    parts = partition(d['y'], d['g'], 5, 'subject', 0.5, np.random.default_rng(0))
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(d['y'])))
    owner = {}
    for k, p in enumerate(parts):
        for s in np.unique(d['g'][p]):
            assert owner.setdefault(s, k) == k


def test_dirichlet_partition_is_a_partition_and_skewed():
    d = toy(per=60)
    parts = partition(d['y'], d['g'], 5, 'dirichlet', 0.1, np.random.default_rng(0))
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(d['y'])))
    assert min(map(len, parts)) >= 10
    shares = np.array([np.bincount(d['y'][p], minlength=4) / len(p) for p in parts])
    assert shares.max() > 0.5  # low alpha gives clients a dominant class


def test_random_split_sizes_and_overlap():
    tr, va, te = random_split(1000, np.random.default_rng(0))
    assert (len(tr), len(va), len(te)) == (700, 100, 200)
    assert not (set(tr) & set(va) | set(tr) & set(te) | set(va) & set(te))


def test_subset_keeps_keys():
    d = toy()
    s = subset(d, d['g'] <= 3)
    assert set(s) == set(d) and set(s['g']) == {1, 2, 3}
