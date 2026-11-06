"""Command line entry point. See README.md for the order in which the commands are run."""
import argparse
import json
import os
import platform
import sys

import numpy as np

CONFIG = 'configs/best_hparams.json'


def load_hp(path=CONFIG):
    return json.load(open(path))


def save_hp(hp, path=CONFIG):
    json.dump(hp, open(path, 'w'), indent=2)


def load_data(name, data_dir):
    from harfl.data import load_ucf, load_uci
    if name == 'uci':
        return load_uci(f'{data_dir}/UCI HAR Dataset')
    feats = f'{data_dir}/ucf8_feats.pt'
    if not os.path.exists(feats):
        sys.exit(f'{feats} not found, run: python run.py features')
    return load_ucf(feats)


def check(data_dir):
    """Sanity checks on the splits, run before any experiment."""
    import torch
    u = load_data('uci', data_dir)
    print('UCI-HAR train/test windows:', len(u['train']['y']), len(u['test']['y']),
          '| train subjects:', len(np.unique(u['train']['g'])), '| test subjects:', len(np.unique(u['test']['g'])))
    assert not set(u['train']['g']) & set(u['test']['g']), 'subjects overlap between train and test'
    v = load_data('ucf', data_dir)
    print('UCF8 train/test clips:', len(v['train']['y']), len(v['test']['y']))
    print('per-class train:', np.bincount(v['train']['y']).tolist(), '| test:', np.bincount(v['test']['y']).tolist())
    assert not set(v['train']['g']) & set(v['test']['g']), 'groups overlap between train and test'
    z = torch.load(f'{data_dir}/ucf8_feats.pt', weights_only=False)
    lst = f'{data_dir}/ucfTrainTestlist/testlist01.txt'
    official = {l.split()[0].split('/')[-1] for l in open(lst) if l.strip()}
    ours = {n for n, g in zip(z['names'], z['groups'].tolist()) if g <= 7}
    diff = ours ^ (official & set(z['names']))
    assert not diff, f'{len(diff)} clips differ from the official testlist01.txt'
    print('UCF101 split 1 matches the official testlist01.txt')


def environment(out):
    import sklearn
    import torch
    info = dict(python=platform.python_version(), torch=torch.__version__, sklearn=sklearn.__version__,
                numpy=np.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    os.makedirs(out, exist_ok=True)
    json.dump(info, open(f'{out}/environment.json', 'w'), indent=1)
    print(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'features', 'search', 'tune-fed', 'final', 'leakage',
                                    'latency', 'summary', 'figures', 'reproduce'])
    ap.add_argument('--data', choices=['uci', 'ucf', 'both'], default='both')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out', default='results')
    ap.add_argument('--trials', type=int, help='search trials (the paper used 12 for uci and 30 for ucf)')
    ap.add_argument('--gray', action='store_true', help='grayscale figures')
    a = ap.parse_args()
    names = ['uci', 'ucf'] if a.data == 'both' else [a.data]

    if a.cmd == 'check':
        return check(a.data_dir)
    if a.cmd == 'features':
        from harfl.features import extract
        return extract(f'{a.data_dir}/UCF-101', f'{a.data_dir}/ucf8_feats.pt')
    if a.cmd == 'summary':
        from harfl.report import summary_table
        return summary_table(a.out)
    if a.cmd == 'figures':
        from harfl.report import figures
        return figures(a.out, a.gray)

    hp = load_hp()
    if a.cmd == 'latency':
        from harfl.bench import benchmark
        return benchmark(hp, a.out)

    from harfl import experiments as E
    if a.cmd == 'reproduce':
        environment(a.out)
        for n in names:
            data = load_data(n, a.data_dir)
            E.final(n, data, hp[n], a.out)
            E.leakage(n, data, hp[n]['central'], a.out)
        from harfl.bench import benchmark
        from harfl.report import figures, summary_table
        benchmark(hp, a.out)
        summary_table(a.out)
        figures(a.out, a.gray)
        return

    for n in names:
        data = load_data(n, a.data_dir)
        if a.cmd == 'search':
            best, _ = E.search(n, data, a.trials or (12 if n == 'uci' else 30), out=f'{a.out}/tuning')
            hp.setdefault(n, {})['central'] = best
            save_hp(hp)
            print('best', n, best)
        elif a.cmd == 'tune-fed':
            pick, _ = E.tune_fed(n, data, hp[n]['central'], out=f'{a.out}/tuning')
            hp[n].update(pick)
            save_hp(hp)
            print('picks', n, pick)
        elif a.cmd == 'final':
            E.final(n, data, hp[n], a.out)
        elif a.cmd == 'leakage':
            E.leakage(n, data, hp[n]['central'], a.out)


if __name__ == '__main__':
    main()
