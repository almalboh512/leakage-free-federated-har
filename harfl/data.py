import numpy as np
import pandas as pd
import torch

from .common import DEV

SIGS = [f'{s}_{a}' for s in ('body_acc', 'body_gyro', 'total_acc') for a in 'xyz']
UCI_LABELS = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
UCF_CLASSES = ['BabyCrawling', 'BodyWeightSquats', 'BoxingPunchingBag', 'BoxingSpeedBag',
               'Fencing', 'Lunges', 'Punch', 'PushUps']


def load_uci(root):
    # official subject-wise split, raw 9-channel signals (128 steps) plus the 561 engineered features
    rd = lambda p: pd.read_csv(p, sep=r'\s+', header=None).values
    out = {}
    for s in ('train', 'test'):
        d = f'{root}/{s}'
        raw = np.stack([rd(f'{d}/Inertial Signals/{n}_{s}.txt') for n in SIGS], 1)
        out[s] = dict(x=raw.astype(np.float32), f=rd(f'{d}/X_{s}.txt').astype(np.float32),
                      y=rd(f'{d}/y_{s}.txt')[:, 0] - 1, g=rd(f'{d}/subject_{s}.txt')[:, 0])
    mu = out['train']['x'].mean((0, 2), keepdims=True)
    sd = out['train']['x'].std((0, 2), keepdims=True)
    for s in out:
        out[s]['x'] = (out[s]['x'] - mu) / sd  # statistics come from the training split only
    return out


def load_ucf(path):
    # official split 1: groups 1-7 form the test set, groups 8-25 the training set
    z = torch.load(path, weights_only=False)
    g, out = z['groups'].numpy(), {}
    for s, m in (('train', g >= 8), ('test', g <= 7)):
        out[s] = dict(x=z['feats'][m].numpy(), y=z['labels'][m].numpy(), g=g[m])
    return out


def to_t(d):
    return dict(x=torch.from_numpy(d['x']).to(DEV), y=torch.from_numpy(d['y']).long().to(DEV), g=d['g'])
