import numpy as np
import pandas as pd


def budget_table(p, y_true, budgets):
    """Precision and recall when only the top-b scored rows are flagged."""
    order = np.argsort(-p)
    n_pos = int(y_true.sum())
    rows = []
    for b in budgets:
        hits = int(y_true[order[:b]].sum())
        rows.append({"alerts": b, "precision": hits / b, "recall": hits / n_pos})
    return pd.DataFrame(rows)
def _weighted_ap(ys, ws):
    pos = ws * ys
    prec = np.cumsum(pos) / np.maximum(np.cumsum(ws), 1)
    return (prec * pos).sum() / pos.sum()


def _interval(values):
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_ci(p, y_true, k=500, reps=200, seed=42):
    """Poisson bootstrap: mean and 95% interval for PR-AUC and precision@k."""
    rng = np.random.default_rng(seed)
    ys = y_true[np.argsort(-p)].astype("float64")
    ap, pk = [], []
    for _ in range(reps):
        ws = rng.poisson(1.0, len(ys)).astype("float64")
        ap.append(_weighted_ap(ys, ws))
        pk.append((ws[:k] * ys[:k]).sum() / ws[:k].sum())
    return {"pr_auc": _interval(ap), f"precision@{k}": _interval(pk)}