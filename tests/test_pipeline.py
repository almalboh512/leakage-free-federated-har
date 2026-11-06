import json

import numpy as np

from harfl import experiments as E
from harfl.common import Config

HP = json.load(open('configs/best_hparams.json'))


def fake(name, per_group=12, n_cls=4, seed=0):
    # class-dependent means make the task learnable, so a working pipeline must beat chance clearly
    rng = np.random.default_rng(seed)

    def split(groups):
        g = np.repeat(groups, per_group)
        y = rng.integers(0, n_cls, len(g))
        shape = (len(g), 9, 128) if name == 'uci' else (len(g), 10, 512)
        x = rng.normal(size=shape).astype(np.float32) + y.reshape(-1, 1, 1).astype(np.float32)
        d = dict(x=x, y=y, g=g)
        if name == 'uci':
            d['f'] = rng.normal(size=(len(g), 20)).astype(np.float32) + y[:, None]
        return d
    groups = np.arange(1, 25) if name == 'uci' else np.arange(8, 26)
    test = np.arange(25, 31) if name == 'uci' else np.arange(1, 8)
    return dict(train=split(groups), test=split(test))


def tiny(name):
    # a large learning rate and no input dropout so that a few epochs are enough on the toy data
    hp = dict(HP[name]['central'], hid=16, lr=3e-3, in_drop=0.0)
    return Config(data=name, seeds=(0, 1), epochs=10, patience=4, **hp)


def test_central_runs_learn_on_both_modalities(tmp_path):
    for name in ('uci', 'ucf'):
        summ = E.run(tiny(name), fake(name), str(tmp_path))
        assert summ['acc'][0] > 0.6, (name, summ)
        r = json.load(open(tmp_path / f'{name}_central_bilstm.json'))
        assert len(r['runs']) == 2 and r['args']['data'] == name


def test_federated_variants_run(tmp_path):
    for name in ('uci', 'ucf'):
        for s in ('fedavg', 'fedprox'):
            cfg = Config(**{**tiny(name).__dict__, 'setting': s, 'local_epochs': 2})
            summ = E.run(cfg, fake(name), str(tmp_path))
            assert summ['acc'][0] > 0.4, (name, s, summ)


def test_cv_score_uses_training_split_only():
    data = fake('ucf')
    data['test']['y'][:] = 0  # destroying the test labels must not change the CV score
    a = E.cv_score(tiny('ucf'), data)
    data['test']['y'][:] = 1
    assert a == E.cv_score(tiny('ucf'), data)
    assert a > 0.5


def test_cv_score_is_repeatable():
    data = fake('ucf')
    assert E.cv_score(tiny('ucf'), data) == E.cv_score(tiny('ucf'), data)


def test_search_caches_duplicates_and_returns_best(tmp_path, monkeypatch):
    # a one-point space forces the sampler to repeat itself
    monkeypatch.setattr(E, 'SPACE', {'lr': [1e-3], 'wd': [1e-4], 'bs': [32], 'hid': [16], 'layers': [1],
                                     'drop': [0.5], 'in_drop': [0.0]})
    monkeypatch.setattr(E, 'DEFAULT', dict(lr=1e-3, wd=1e-4, bs=32, hid=16, layers=2, drop=0.5, in_drop=0.0))
    calls = []

    def fake_cv(cfg, data, folds=3, seed=0):
        calls.append(cfg)
        return 0.5 + 0.1 * len(calls)

    monkeypatch.setattr(E, 'cv_score', fake_cv)
    best, df = E.search('ucf', fake('ucf'), 4, out=str(tmp_path))
    assert len(df) == 4 and len(calls) == 2  # the default plus one distinct sampled configuration
    assert best['layers'] == 1 and isinstance(best['lr'], float)


def test_tuned_baselines_and_leakage(tmp_path):
    data = fake('uci')
    res = E.tuned_baselines('uci', data, str(tmp_path))
    assert res['svm_rbf'][0] > 0.6 and len(res['mlp']) == 5
    E.leakage('ucf', fake('ucf'), dict(HP['ucf']['central'], hid=16), str(tmp_path))
    assert (tmp_path / 'ucf_leakage.csv').exists()
