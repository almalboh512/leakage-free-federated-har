import random
from dataclasses import dataclass

import numpy as np
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    # one place that seeds python, numpy and torch so a run repeats on the same hardware
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    data: str = 'uci'            # uci | ucf
    setting: str = 'central'     # central | fedavg | fedprox
    head: str = 'bilstm'         # bilstm | lstm | pool
    seeds: tuple = (0, 1, 2)
    epochs: int = 40
    patience: int = 6
    lr: float = 1e-3
    wd: float = 1e-4
    bs: int = 32
    hid: int = 128
    layers: int = 2
    drop: float = 0.5
    in_drop: float = 0.0
    ch: int = 64                 # sensor CNN width, ignored for video
    ks: int = 5                  # sensor CNN kernel size, ignored for video
    clients: int = 5
    local_epochs: int = 1
    alpha: float = 0.5           # Dirichlet concentration for the video partition
    mu: float = 0.01             # FedProx proximal weight
