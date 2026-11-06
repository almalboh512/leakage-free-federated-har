import json

import torch

from harfl.common import Config
from harfl.models import Head, SensorNet, build, n_trainable

HP = json.load(open('configs/best_hparams.json'))


def test_shapes():
    assert SensorNet('bilstm', 6)(torch.randn(4, 9, 128)).shape == (4, 6)
    for kind in ('bilstm', 'lstm', 'pool'):
        assert Head(kind, 512, 8)(torch.randn(4, 10, 512)).shape == (4, 8)


def test_parameter_counts_match_the_paper_tables():
    # these counts are reported in the manuscript, so a change to the architecture must fail here
    expected = {('uci', 'bilstm'): 236_806, ('uci', 'lstm'): 129_286, ('uci', 'pool'): 25_862,
                ('ucf', 'bilstm'): 674_376, ('ucf', 'lstm'): 337_480, ('ucf', 'pool'): 33_352}
    for (d, head), n in expected.items():
        m = build(Config(data=d, head=head, **HP[d]['central']), 6 if d == 'uci' else 8)
        assert n_trainable(m) == n, (d, head, n_trainable(m))


def test_bilstm_uses_both_final_states():
    # the backward direction must see the whole sequence, so reversing the input must change the output
    torch.manual_seed(0)
    h = Head('bilstm', 8, 3, hid=16, layers=1, drop=0.0).eval()
    x = torch.randn(2, 10, 8)
    assert not torch.allclose(h(x), h(x.flip(1)), atol=1e-5)


def test_eval_is_deterministic():
    m = build(Config(data='uci', **HP['uci']['central']), 6).eval()
    x = torch.randn(3, 9, 128)
    assert torch.equal(m(x), m(x))
