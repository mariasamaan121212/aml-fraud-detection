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