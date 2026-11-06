import json
import os
import time

import torch
import torch.nn as nn
import torchvision

from .common import Config
from .models import Head, build, n_trainable


def ms(fn, x, dev, reps=300, warm=30):
    # warm-up first so allocator setup and kernel selection are not timed
    with torch.no_grad():
        for _ in range(warm):
            fn(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / reps


def benchmark(hp, out='results'):
    """Batch-size-1 latency and parameter counts for the tuned models, plus federated payload size."""
    report = dict(batch_size=1, torch=torch.__version__, devices={})
    sizes = {}
    for d, n_cls in (('uci', 6), ('ucf', 8)):
        for head in ('bilstm', 'lstm', 'pool'):
            m = build(Config(data=d, head=head, **hp[d]['central']), n_cls)
            sizes[f'{d}_{head}'] = n_trainable(m)
    backbone = torchvision.models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    report['trainable_params'] = sizes
    report['frozen_backbone_params'] = sum(p.numel() for p in backbone.parameters())
    # a federated round sends the trainable weights only, as float32, in each direction
    report['payload_MB_per_client_per_direction'] = {k: v * 4 / 1e6 for k, v in sizes.items()}

    for name in ('cpu', 'cuda'):
        if name == 'cuda' and not torch.cuda.is_available():
            continue
        dev = torch.device(name)
        label = torch.cuda.get_device_name(0) if name == 'cuda' else 'cpu'
        r = {'name': label, 'sensor_ms': {}, 'video_head_ms': {}}
        for head in ('bilstm', 'lstm', 'pool'):
            s = build(Config(data='uci', head=head, **hp['uci']['central']), 6).to(dev).eval()
            r['sensor_ms'][head] = ms(s, torch.randn(1, 9, 128, device=dev), dev)
            h = build(Config(data='ucf', head=head, **hp['ucf']['central']), 8).to(dev).eval()
            r['video_head_ms'][head] = ms(h, torch.randn(1, 10, 512, device=dev), dev)
        net = backbone.to(dev).eval()
        r['backbone_ms_per_clip'] = ms(net, torch.randn(10, 3, 224, 224, device=dev), dev, reps=50, warm=10)
        r['video_end_to_end_ms'] = r['backbone_ms_per_clip'] + r['video_head_ms']['bilstm']
        report['devices'][name] = r

    os.makedirs(out, exist_ok=True)
    json.dump(report, open(f'{out}/latency.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))
    return report
