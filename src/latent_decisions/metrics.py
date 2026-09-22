"""Binary Brier is (p-y)^2, not the sum over two classes. ECE is descriptive."""

import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(logits):
    x = np.clip(np.asarray(logits, dtype=float), -50, 50)
    return 1 / (1 + np.exp(-x))


def metrics(y, p, bins=10):
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    if len(y) != len(p) or not len(y):
        raise ValueError("Expected nonempty aligned labels and probabilities")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels or probabilities")
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    index = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            reliability.append(
                dict(
                    bin=b,
                    n=int(mask.sum()),
                    probability=float(p[mask].mean()),
                    frequency=float(y[mask].mean()),
                )
            )
    ece = sum(r["n"] * abs(r["probability"] - r["frequency"]) for r in reliability) / len(y)
    return dict(
        n=len(y),
        accuracy=float(((p >= 0.5) == y).mean()),
        brier=float(np.mean((p - y) ** 2)),
        nll=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        ece=float(ece),
        reliability=reliability,
    )


def fit_temperature(logits, y):
    """Fit only on the calibration split. Bounded, deterministic NLL grid search."""
    temperatures = np.geomspace(0.1, 10.0, 161)
    losses = [metrics(y, sigmoid(logits / t))["nll"] for t in temperatures]
    return float(temperatures[np.argmin(losses)])


def cluster_mean_ci(values, groups, seed=0, repeats=1000):
    """Percentile interval over whole scenario groups; not individual query rows."""
    values, groups = np.asarray(values), np.asarray(groups)
    unique = np.unique(groups)
    means = np.array([values[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    samples = rng.choice(means, (repeats, len(means)), replace=True).mean(axis=1)
    return dict(
        mean=float(values.mean()),
        low=float(np.quantile(samples, 0.025)),
        high=float(np.quantile(samples, 0.975)),
        groups=len(unique),
    )
