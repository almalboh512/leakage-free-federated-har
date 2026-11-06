import json

import numpy as np

import run as cli
from fakedata import make_all
from harfl import experiments as E
from harfl.common import Config
from harfl.report import figures, summary_table

HP = json.load(open('configs/best_hparams.json'))


def test_loaders_checks_and_reports(tmp_path, capsys):
    root = str(tmp_path / 'data')
    make_all(root)
    cli.check(root)  # asserts the splits are disjoint and match testlist01.txt
    assert 'matches the official' in capsys.readouterr().out

    u = cli.load_data('uci', root)
    assert u['train']['x'].shape[1:] == (9, 128) and u['train']['f'].shape[1] == 561
    assert abs(float(u['train']['x'].mean())) < 1e-3  # normalised with training statistics
    v = cli.load_data('ucf', root)
    assert v['train']['x'].shape[1:] == (10, 512) and set(v['train']['g']).isdisjoint(v['test']['g'])

    out = str(tmp_path / 'results')
    for name, data in (('uci', u), ('ucf', v)):
        hp = dict(HP[name]['central'], hid=16, lr=3e-3, in_drop=0.0)
        for s in ('central', 'fedavg'):
            E.run(Config(data=name, setting=s, seeds=(0, 1), epochs=3, patience=2, **hp), data, out)
    E.tuned_baselines('uci', u, out)
    table = summary_table(out)
    assert {'uci_central_bilstm', 'ucf_fedavg_bilstm', 'uci_baselines_tuned:svm_rbf'} <= set(table.run)
    figures(out)
    assert (tmp_path / 'results' / 'uci_central_bilstm_cm.pdf').exists()
    assert np.load(f'{out}/ucf_central_bilstm_cm.npy').sum() == len(v['test']['y']) * 2
