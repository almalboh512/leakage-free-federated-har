import dataclasses
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import Config, seed_all
from .data import to_t
from .models import build
from .splits import group_folds, holdout, random_split, subset
from .train import fit, predict, train_central

# Search spaces. The video head has no CNN, so ch and ks only exist for the sensor model.
SPACE = dict(lr=[3e-4, 1e-3, 3e-3], wd=[1e-4, 1e-3, 1e-2], bs=[16, 32, 64], hid=[32, 64, 128],
             layers=[1, 2], drop=[0.3, 0.5, 0.7], in_drop=[0.0, 0.2, 0.5])
SPACE_SENSOR = dict(SPACE, ch=[32, 64], ks=[5, 9])
DEFAULT = dict(lr=1e-3, wd=1e-4, bs=32, hid=128, layers=2, drop=0.5, in_drop=0.0)
DEFAULT_SENSOR = dict(DEFAULT, ch=64, ks=5)
LOCAL_EPOCHS = (1, 2, 3)
MU_GRID = (0.001, 0.01, 0.1)


def n_classes(data):
    return int(data['train']['y'].max()) + 1


def scores(t, p):
    return dict(acc=accuracy_score(t, p), f1=f1_score(t, p, average='macro'),
                prec=precision_score(t, p, average='macro', zero_division=0),
                rec=recall_score(t, p, average='macro', zero_division=0))


def summarize(runs):
    # sample standard deviation (n - 1) over seeds
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs], ddof=1)))
            for k in ('acc', 'f1', 'prec', 'rec')}


def run(cfg, data, out='results'):
    """Train on the official training split once per seed, evaluate on the official test split once."""
    n_cls = n_classes(data)
    runs, curves, cm = [], [], np.zeros((n_cls, n_cls), int)
    for seed in cfg.seeds:
        seed_all(seed)
        rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
        tr, va = holdout(data['train'], 3, rng)
        tr, va, te = to_t(tr), to_t(va), to_t(data['test'])
        model, hist = fit(cfg, tr, va, n_cls, rng, gen)
        _, p = predict(model, te['x'], te['y'])
        t = te['y'].cpu().numpy()
        runs.append(dict(seed=seed, **scores(t, p)))
        curves.append(hist)
        cm += confusion_matrix(t, p, labels=range(n_cls))
    summ = summarize(runs)
    name = f'{cfg.data}_{cfg.setting}_{cfg.head}'
    os.makedirs(out, exist_ok=True)
    json.dump(dict(args=dataclasses.asdict(cfg), runs=runs, summary=summ, curves=curves),
              open(f'{out}/{name}.json', 'w'))
    np.save(f'{out}/{name}_cm.npy', cm)
    print(name, {k: f'{m * 100:.2f} +/- {s * 100:.2f}' for k, (m, s) in summ.items()}, flush=True)
    return summ


def cv_score(cfg, data, folds=3, seed=0):
    """Group-wise cross-validation on the training split only. The test split is never touched."""
    tr, n_cls = data['train'], n_classes(data)
    rng, accs = np.random.default_rng(seed), []
    for k, held in enumerate(group_folds(tr['g'], folds, rng)):
        fit_d, hold = subset(tr, ~held), subset(tr, held)
        fit_d, va = holdout(fit_d, 3, rng)  # early stopping uses held-out groups as well
        seed_all(seed + k)
        gen = torch.Generator().manual_seed(seed + k)
        model, _ = fit(cfg, to_t(fit_d), to_t(va), n_cls, rng, gen)
        h = to_t(hold)
        _, p = predict(model, h['x'], h['y'])
        accs.append(accuracy_score(hold['y'], p))
    return float(np.mean(accs))


def search(name, data, n_trials, seed=0, out='results/tuning'):
    """Random search over the head and CNN hyperparameters. Trial 0 is the untuned default."""
    space, base = (SPACE_SENSOR, DEFAULT_SENSOR) if name == 'uci' else (SPACE, DEFAULT)
    rng = np.random.default_rng(seed)
    trials = [dict(base)] + [{k: v[int(rng.integers(len(v)))] for k, v in space.items()}
                             for _ in range(n_trials - 1)]
    rows, seen = [], {}
    for i, hp in enumerate(trials):
        key = tuple(sorted(hp.items()))
        t0 = time.time()
        if key not in seen:  # the sampler can draw a configuration twice, and training is deterministic
            seen[key] = cv_score(Config(data=name, **hp), data, seed=seed)
        rows.append(dict(trial=i, **hp, cv_acc=seen[key]))
        print(i, f'{seen[key] * 100:.2f}', hp, f'{time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows).sort_values('cv_acc', ascending=False)
    os.makedirs(out, exist_ok=True)
    df.to_csv(f'{out}/search_{name}.csv', index=False)
    best = {k: df.iloc[0][k].item() if hasattr(df.iloc[0][k], 'item') else df.iloc[0][k] for k in trials[0]}
    return best, df


def tune_fed(name, data, best, out='results/tuning'):
    """Pick local epochs for FedAvg, and local epochs plus mu for FedProx, by group-wise CV."""
    rows = []
    for le in LOCAL_EPOCHS:
        cfg = Config(data=name, setting='fedavg', local_epochs=le, **best)
        rows.append(dict(setting='fedavg', local_epochs=le, mu=0.0, cv_acc=cv_score(cfg, data)))
        print(rows[-1], flush=True)
    for le in LOCAL_EPOCHS:
        for mu in MU_GRID:
            cfg = Config(data=name, setting='fedprox', local_epochs=le, mu=mu, **best)
            rows.append(dict(setting='fedprox', local_epochs=le, mu=mu, cv_acc=cv_score(cfg, data)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    df.to_csv(f'{out}/fed_{name}.csv', index=False)
    pick = {}
    for s in ('fedavg', 'fedprox'):
        r = df[df.setting == s].sort_values('cv_acc', ascending=False).iloc[0]
        pick[s] = dict(local_epochs=int(r.local_epochs))
        if s == 'fedprox':
            pick[s]['mu'] = float(r.mu)
    return pick, df


def tuned_baselines(name, data, out='results'):
    """SVM and MLP on engineered features, tuned with the same group-wise CV and the same training split."""
    feat = (lambda d: d['f']) if name == 'uci' else (lambda d: d['x'].mean(1))  # 561 features, or mean-pooled frames
    tr, te = data['train'], data['test']
    cv = GroupKFold(3)
    grids = {
        'svm_rbf': (lambda s: SVC(), {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 1e-3, 1e-2]}),
        'mlp': (lambda s: MLPClassifier(max_iter=300, random_state=s),
                {'mlpclassifier__hidden_layer_sizes': [(128,), (256,), (256, 128)],
                 'mlpclassifier__alpha': [1e-4, 1e-3, 1e-2]}),
    }
    res = {}
    for m, (mk, grid) in grids.items():
        gs = GridSearchCV(make_pipeline(StandardScaler(), mk(0)), grid, cv=cv, n_jobs=1)
        gs.fit(feat(tr), tr['y'], groups=tr['g'])
        params = {k.split('__')[1]: v for k, v in gs.best_params_.items()}
        seeds = (0,) if m == 'svm_rbf' else range(5)  # SVC is deterministic, the MLP is not
        accs = []
        for s in seeds:
            model = make_pipeline(StandardScaler(), mk(s)).set_params(**gs.best_params_)
            accs.append(accuracy_score(te['y'], model.fit(feat(tr), tr['y']).predict(feat(te))))
        res[m] = accs
        res.setdefault('params', {})[m] = {k: str(v) for k, v in params.items()}
        print(name, m, f'{np.mean(accs) * 100:.2f}', params, flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f'{out}/{name}_baselines_tuned.json', 'w'))
    return res


def run_random_split(cfg, data):
    """Leakage demonstration: pool train and test, then split 70/10/20 at random."""
    pooled = {k: np.concatenate([data['train'][k], data['test'][k]]) for k in data['train']}
    n_cls, accs = n_classes(data), []
    for seed in cfg.seeds:
        seed_all(seed)
        rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
        i_tr, i_va, i_te = random_split(len(pooled['y']), rng)
        tr, va, te = (to_t(subset(pooled, i)) for i in (i_tr, i_va, i_te))
        model, _ = train_central(lambda: build(cfg, n_cls), tr, va, cfg, gen)
        _, p = predict(model, te['x'], te['y'])
        accs.append(accuracy_score(te['y'].cpu().numpy(), p))
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def leakage(name, data, tuned, out='results'):
    """Same model and budget under the random split and under the official split."""
    default = DEFAULT_SENSOR if name == 'uci' else DEFAULT
    rows = []
    for label, hp in (('default', default), ('tuned', tuned)):
        cfg = Config(data=name, seeds=(0, 1, 2), **hp)
        rm, rs = run_random_split(cfg, data)
        om, os_ = run(cfg, data, out=f'{out}/leakage_runs/{label}')['acc']
        rows.append(dict(data=name, config=label, random_split=f'{rm * 100:.2f} +/- {rs * 100:.2f}',
                         official_split=f'{om * 100:.2f} +/- {os_ * 100:.2f}'))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{out}/{name}_leakage.csv', index=False)


def final(name, data, hp, out='results', seeds=range(5)):
    """Final runs with the stored hyperparameters: 3 heads centralized, FedAvg and FedProx, tuned baselines."""
    base = dict(data=name, seeds=tuple(seeds), epochs=50, patience=7, **hp['central'])
    for head in ('bilstm', 'lstm', 'pool'):
        run(Config(**base, setting='central', head=head), data, out)
    for s in ('fedavg', 'fedprox'):
        run(Config(**base, setting=s, head='bilstm', **hp[s]), data, out)
    tuned_baselines(name, data, out)

