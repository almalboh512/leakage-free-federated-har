# Federated human activity recognition with a shared CNN-BiLSTM design

Code for the MVIP2026 camera-ready experiments: federated human activity recognition (HAR) on inertial sensor data (UCI-HAR) and on video (an eight-class UCF101 subset), under a single leakage-free evaluation protocol.

## Protocol

- UCI-HAR: official subject-wise split, 21 training and 9 test subjects (7,352 and 2,947 windows). The model reads the raw 9-channel signals (128 steps) after a 1-D CNN.
- UCF8: classes BabyCrawling, BodyWeightSquats, BoxingPunchingBag, BoxingSpeedBag, Fencing, Lunges, Punch, PushUps (1,041 clips). UCF101 split 1 is used: groups 1 to 7 form the test set (291 clips), groups 8 to 25 the training set (750 clips). Ten frames per clip pass through a frozen ImageNet ResNet-18, and the features are cached once.
- Validation for early stopping is made of three held-out subjects or groups per seed, never random windows or clips.
- Hyperparameters are chosen by group-wise 3-fold cross-validation on the training split only (random search, with trial 0 fixed to the untuned default). The test split is evaluated once per seed.
- Federated learning: FedAvg with client-size weights and FedProx, five clients. UCI-HAR clients hold whole subjects. UCF8 clients hold Dirichlet label-skewed partitions (alpha 0.5). Local epochs, and mu for FedProx, are chosen by the same group-wise cross-validation.
- Baselines (SVM and MLP) use the 561 engineered features for UCI-HAR and mean-pooled frame features for UCF8. They are tuned with the same group-wise folds.
- Results are the mean and sample standard deviation (n - 1) over five seeds. Precision, recall and F1 are macro-averaged.
- `leakage` repeats the same model under a random 70/10/20 split of the pooled data, to measure how much that protocol inflates accuracy.

## Setup

```
pip install -r requirements.txt
bash scripts/download_data.sh          # about 7 GB with the UCF101 archive
python run.py features                 # caches ResNet-18 features, run once (GPU recommended)
python run.py check                    # split sanity checks
```

`check` verifies that train and test subjects and groups are disjoint and that the group rule reproduces the official `testlist01.txt` exactly.

## Reproducing the reported results

The tuned hyperparameters are stored in `configs/best_hparams.json`, and the search logs they came from are in `configs/search_logs/`. One command reruns every experiment from those stored settings:

```
python run.py reproduce
```

This runs the final five-seed experiments (three temporal heads centralized, FedAvg, FedProx, tuned baselines), the leakage comparison, latency and parameter counts, the summary table and the figures. Everything lands in `results/`, with `results/summary.csv` as the table and `results/environment.json` recording the software and GPU used.

The stages can also be run separately:

```
python run.py final --data uci         # final runs and tuned baselines
python run.py leakage --data ucf       # random split versus official split
python run.py latency                  # batch-size-1 latency, parameter counts, federated payload
python run.py summary
python run.py figures [--gray]
```

## Rerunning the hyperparameter search

```
python run.py search                   # 12 trials for uci, 30 for ucf (override with --trials)
python run.py tune-fed
```

Both commands overwrite `configs/best_hparams.json` with their picks, and write the full search tables to `results/tuning/`. The random search uses a fixed seed, so the same command draws the same candidates on every run of this code. The tables in `configs/search_logs/` record the search that produced the stored hyperparameters. The sampler can draw a configuration twice (this happened once in the 30 UCF8 trials), and a repeat is not trained again.

## Determinism

Each run seeds Python, NumPy and PyTorch and sets cuDNN to deterministic mode. Results should repeat exactly on the same hardware and library versions. Across GPU models or PyTorch versions the last digit of an accuracy can differ.

## Layout

```
harfl/splits.py        split, validation and federated partition logic (numpy only)
harfl/data.py          dataset loaders
harfl/models.py        temporal head (BiLSTM, LSTM, mean-pool) and sensor CNN
harfl/train.py         centralized and federated training
harfl/experiments.py   runs, cross-validation, search, baselines, leakage demo
harfl/features.py      frozen ResNet-18 feature caching
harfl/bench.py         latency, parameter counts, federated payload size
harfl/report.py        summary table and figures
run.py                 command line entry point
tests/                 unit tests and an offline end-to-end test on synthetic data
```

## Tests

```
python -m pytest tests
```

The tests need no dataset. They check the split logic, the parameter counts reported in the paper, that training learns a toy task, that cross-validation never reads the test split, and the full pipeline on synthetic data written in the real on-disk layout.

## License

MIT, see `LICENSE`.
