import glob
import os
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .common import DEV
from .data import UCF_CLASSES

T = 10  # frames sampled per clip, evenly spaced


def extract(ucf_root, out):
    """Cache frozen ImageNet ResNet-18 features for the eight UCF101 classes. Run once."""
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.to(DEV).eval()  # eval mode keeps BatchNorm statistics at their ImageNet values

    @torch.no_grad()
    def clip_features(path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = np.linspace(0, max(total - 1, 0), T).astype(int)
        keep, frames, i = set(wanted.tolist()), {}, 0
        while i <= wanted.max():
            ok, f = cap.read()
            if not ok:
                break
            if i in keep:
                frames[i] = tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        if not frames:
            raise RuntimeError(f'no frames decoded from {path}')  # fail loudly instead of zero padding
        last = frames[max(frames)]
        return net(torch.stack([frames.get(j, last) for j in wanted]).to(DEV)).cpu()

    feats, labels, groups, names = [], [], [], []
    for c, cls in enumerate(UCF_CLASSES):
        files = sorted(glob.glob(os.path.join(ucf_root, cls, '*.avi')))
        assert files, f'class folder missing or empty: {cls}'
        for f in files:
            feats.append(clip_features(f))
            labels.append(c)
            groups.append(int(re.search(r'_g(\d+)_', f).group(1)))
            names.append(os.path.basename(f))
        print(cls, len(files), flush=True)
    feats = torch.stack(feats)
    torch.save(dict(feats=feats, labels=torch.tensor(labels), groups=torch.tensor(groups),
                    names=names, classes=UCF_CLASSES), out)
    print('saved', out, tuple(feats.shape))
