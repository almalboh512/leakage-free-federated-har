import torch
import torch.nn as nn


class Head(nn.Module):
    """Temporal head over a sequence of encoder vectors (B x T x D)."""

    def __init__(self, kind, d_in, n_cls, hid=128, layers=2, drop=0.5, in_drop=0.0):
        super().__init__()
        assert kind in ('bilstm', 'lstm', 'pool')
        self.kind = kind
        self.in_drop = nn.Dropout(in_drop)
        if kind != 'pool':
            self.rnn = nn.LSTM(d_in, hid, num_layers=layers, batch_first=True,
                               bidirectional=(kind == 'bilstm'), dropout=drop if layers > 1 else 0.)
            d_in = hid * (2 if kind == 'bilstm' else 1)
        self.fc = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_cls))

    def forward(self, x):
        x = self.in_drop(x)
        if self.kind == 'pool':
            return self.fc(x.mean(1))
        self.rnn.flatten_parameters()  # weights are no longer contiguous after deepcopy in the federated loop
        h = self.rnn(x)[1][0]  # h_n has shape (layers * directions) x B x H
        # lstm_out[:, -1] would give the backward state after a single step, so h_n is used instead
        h = torch.cat([h[-2], h[-1]], 1) if self.kind == 'bilstm' else h[-1]
        return self.fc(h)


class SensorNet(nn.Module):
    """1-D CNN over raw inertial signals (B x 9 x 128), then a temporal head over the 32 time steps."""

    def __init__(self, kind, n_cls, ch=64, ks=5, **head_kw):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(9, ch, ks, padding=ks // 2), nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch, 2 * ch, ks, padding=ks // 2), nn.BatchNorm1d(2 * ch), nn.ReLU(), nn.MaxPool1d(2))
        self.head = Head(kind, 2 * ch, n_cls, **head_kw)

    def forward(self, x):  # the convolution runs over time, so the LSTM sees time steps
        return self.head(self.cnn(x).transpose(1, 2))


def build(cfg, n_cls):
    kw = dict(hid=cfg.hid, layers=cfg.layers, drop=cfg.drop, in_drop=cfg.in_drop)
    if cfg.data == 'uci':
        return SensorNet(cfg.head, n_cls, ch=cfg.ch, ks=cfg.ks, **kw)
    return Head(cfg.head, 512, n_cls, **kw)  # 512 = ResNet-18 feature size


def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]


def n_trainable(m):
    return sum(p.numel() for p in trainable(m))
