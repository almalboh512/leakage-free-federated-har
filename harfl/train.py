import copy

import numpy as np
import torch
import torch.nn.functional as F

from .common import DEV
from .models import build, trainable
from .splits import partition


def predict(model, x, y, bs=512):
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])
    return F.cross_entropy(out, y).item(), out.argmax(1).cpu().numpy()


def run_epoch(model, x, y, opt, bs, gen, ref=None, mu=0.):
    model.train()
    perm = torch.randperm(len(x), generator=gen).to(DEV)
    total, n = 0., 0
    for i in range(0, len(x), bs):
        b = perm[i:i + bs]
        if len(b) < 2:  # BatchNorm cannot train on a single sample
            continue
        loss = F.cross_entropy(model(x[b]), y[b])
        if mu:  # FedProx proximal term toward the global weights
            loss = loss + mu / 2 * sum(((p - r) ** 2).sum() for p, r in zip(trainable(model), ref))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, n = total + loss.item(), n + 1
    return total / max(n, 1)


class Best:
    """Early stopping on validation loss with best-weights restore."""

    def __init__(self, patience):
        self.loss, self.state, self.bad, self.patience = 1e9, None, 0, patience

    def step(self, val_loss, model):
        if val_loss < self.loss:
            self.loss, self.state, self.bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            self.bad += 1
        return self.bad >= self.patience


def train_central(mk, tr, va, cfg, gen):
    model, hist = mk().to(DEV), []
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best = Best(cfg.patience)
    for _ in range(cfg.epochs):
        tl = run_epoch(model, tr['x'], tr['y'], opt, cfg.bs, gen)
        vl, _ = predict(model, va['x'], va['y'])
        hist.append((tl, vl))
        sch.step(vl)
        if best.step(vl, model):
            break
    model.load_state_dict(best.state)
    return model, hist


def train_fed(mk, tr, va, cfg, gen, rng):
    mu = cfg.mu if cfg.setting == 'fedprox' else 0.
    mode = 'subject' if cfg.data == 'uci' else 'dirichlet'
    parts = partition(tr['y'].cpu().numpy(), tr['g'], cfg.clients, mode, cfg.alpha, rng)
    gm, hist, best = mk().to(DEV), [], Best(cfg.patience)
    w = [len(p) / sum(map(len, parts)) for p in parts]  # FedAvg weights follow client size
    for _ in range(cfg.epochs):  # one epoch here is one communication round
        ref = [p.detach().clone() for p in trainable(gm)]
        states, tls = [], []
        for idx in parts:
            lm = copy.deepcopy(gm)
            opt = torch.optim.Adam(lm.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
            i = torch.as_tensor(idx, device=DEV)
            for _ in range(cfg.local_epochs):
                tls.append(run_epoch(lm, tr['x'][i], tr['y'][i], opt, cfg.bs, gen, ref, mu))
            states.append(lm.state_dict())
        gm.load_state_dict({k: sum(wi * s[k].float() for wi, s in zip(w, states)).to(states[0][k].dtype)
                            for k in states[0]})
        vl, _ = predict(gm, va['x'], va['y'])
        hist.append((float(np.mean(tls)), vl))
        if best.step(vl, gm):
            break
    gm.load_state_dict(best.state)
    return gm, hist


def fit(cfg, tr, va, n_cls, rng, gen):
    mk = lambda: build(cfg, n_cls)
    if cfg.setting == 'central':
        return train_central(mk, tr, va, cfg, gen)
    return train_fed(mk, tr, va, cfg, gen, rng)
