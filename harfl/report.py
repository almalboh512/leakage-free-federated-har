import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .data import UCF_CLASSES, UCI_LABELS  # noqa: E402

LABELS = {'uci': UCI_LABELS, 'ucf': UCF_CLASSES}
FIGURE_RUNS = ('central', 'fedavg')


def summary_table(out='results'):
    """Mean and sample std over seeds on the official test split, one row per run."""
    rows = []
    for f in sorted(glob.glob(f'{out}/u*_*.json')):
        n = os.path.basename(f)[:-5]
        r = json.load(open(f))
        if 'summary' in r:
            rows.append(dict(run=n, **{k: f'{r["summary"][k][0] * 100:.2f} +/- {r["summary"][k][1] * 100:.2f}'
                                       for k in ('acc', 'f1', 'prec', 'rec')}))
        else:  # tuned baselines: {model: [accuracies]}, plus the chosen parameters
            for m, accs in r.items():
                if m == 'params':
                    continue
                sd = f' +/- {np.std(accs, ddof=1) * 100:.2f}' if len(accs) > 1 else ''
                rows.append(dict(run=f'{n}:{m}', acc=f'{np.mean(accs) * 100:.2f}{sd}'))
    table = pd.DataFrame(rows)
    table.to_csv(f'{out}/summary.csv', index=False)
    print(table.to_string(index=False))
    return table


def figures(out='results', gray=False):
    """Confusion matrices and mean loss curves. Matplotlib defaults, with a grayscale switch."""
    cmap = 'Greys' if gray else 'Blues'
    for d in ('uci', 'ucf'):
        for s in FIGURE_RUNS:
            name = f'{d}_{s}_bilstm'
            if not os.path.exists(f'{out}/{name}.json'):
                continue
            labels, cm = LABELS[d], np.load(f'{out}/{name}_cm.npy')
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, cm[i, j], ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')
            ax.set_xticks(range(len(labels)), labels, rotation=90)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            fig.tight_layout()
            fig.savefig(f'{out}/{name}_cm.pdf')
            plt.close(fig)

            curves = json.load(open(f'{out}/{name}.json'))['curves']
            n = min(len(c) for c in curves)  # early stopping gives runs of different length
            c = np.array([c[:n] for c in curves])
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            for k, (lab, ls, mk) in enumerate((('Training loss', '-', 'o'), ('Validation loss', '--', 's'))):
                m, sd = c[:, :, k].mean(0), c[:, :, k].std(0, ddof=1)
                color = 'black' if gray else f'C{k}'
                ax.plot(range(1, n + 1), m, ls, marker=mk, color=color, markersize=3, label=lab)
                ax.fill_between(range(1, n + 1), m - sd, m + sd, color=color, alpha=0.15, linewidth=0)
            ax.set_xlabel('Communication round' if s != 'central' else 'Epoch')
            ax.set_ylabel('Loss')
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(f'{out}/{name}_loss.pdf')
            plt.close(fig)
    print('figures written to', out)
