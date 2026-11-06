"""Writes a tiny dataset in the exact on-disk layout of the real ones, so the loaders and CLI can be tested offline."""
import os

import numpy as np
import torch

from harfl.data import SIGS, UCF_CLASSES


def make_uci(root, per_subject=16, seed=0):
    rng = np.random.default_rng(seed)
    for split, subjects in (('train', range(1, 22)), ('test', range(22, 31))):
        d = f'{root}/UCI HAR Dataset/{split}'
        os.makedirs(f'{d}/Inertial Signals', exist_ok=True)
        subj = np.repeat(list(subjects), per_subject)
        y = rng.integers(1, 7, len(subj))
        for n in SIGS:
            x = rng.normal(size=(len(y), 128)) + y[:, None]
            np.savetxt(f'{d}/Inertial Signals/{n}_{split}.txt', x, fmt='%.5f')
        np.savetxt(f'{d}/X_{split}.txt', rng.normal(size=(len(y), 561)) + y[:, None], fmt='%.5f')
        np.savetxt(f'{d}/y_{split}.txt', y, fmt='%d')
        np.savetxt(f'{d}/subject_{split}.txt', subj, fmt='%d')


def make_ucf(root, clips_per_group=2, seed=0):
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(len(UCF_CLASSES), 1, 512))  # one random direction per class, so the toy task is not ordinal
    feats, labels, groups, names = [], [], [], []
    for c, cls in enumerate(UCF_CLASSES):
        for g in range(1, 26):
            for k in range(1, clips_per_group + 1):
                feats.append(torch.tensor(rng.normal(size=(10, 512)) + proto[c], dtype=torch.float32))
                labels.append(c)
                groups.append(g)
                names.append(f'v_{cls}_g{g:02d}_c{k:02d}.avi')
    torch.save(dict(feats=torch.stack(feats), labels=torch.tensor(labels), groups=torch.tensor(groups),
                    names=names, classes=UCF_CLASSES), f'{root}/ucf8_feats.pt')
    os.makedirs(f'{root}/ucfTrainTestlist', exist_ok=True)
    with open(f'{root}/ucfTrainTestlist/testlist01.txt', 'w') as f:
        for n, g in zip(names, groups):
            if g <= 7:
                f.write(f'{n.split("_")[1]}/{n} 1\n')


def make_all(root):
    make_uci(root)
    make_ucf(root)
