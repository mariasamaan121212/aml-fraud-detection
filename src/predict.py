import json
from functools import lru_cache

import lightgbm as lgb
import pandas as pd

from src.data import ROOT
from src.features import make_features

MODELS = ROOT / "models"

RAW_SOURCE = {
    "log_amount": "amount",
    "log_time_since_last": "time_since_last_transaction",
    "log_velocity": "velocity_score",
    "hour_sin": "hour",
    "hour_cos": "hour",
}


@lru_cache(maxsize=1)
def load_model(meta_name: str = "lgbm_v1_meta.json"):
    """Load the saved booster and its metadata once."""
    meta = json.loads((MODELS / meta_name).read_text())
    booster = lgb.Booster(model_file=str(MODELS / meta["model_file"]))
    return booster, meta


def _label(feature: str) -> str:
    if feature.startswith("channel_"):
        return "payment_channel"
    return RAW_SOURCE.get(feature, feature)


def _py(value):
    return value.item() if hasattr(value, "item") else value


def score(df_raw: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Risk score plus the top contributing inputs for each raw transaction row."""
    booster, meta = load_model()
    cols = meta["feature_columns"]
    X = make_features(df_raw)[cols]
    base_rate = meta["n_train_fraud"] / meta["n_train_rows"]

    prob = booster.predict(X)
    contrib = booster.predict(X, pred_contrib=True)
    base_value = contrib[:, -1]
    grouped = (
        pd.DataFrame(contrib[:, :-1], columns=[_label(c) for c in cols])
        .T.groupby(level=0).sum().T
    )

    results = []
    for i in range(len(X)):
        row = grouped.iloc[i]
        top = row.reindex(row.abs().sort_values(ascending=False).index).iloc[:top_n]
        factors = []
        for label, value in top.items():
            if label in df_raw.columns:
                shown = _py(df_raw.iloc[i][label])
            else:
                shown = _py(X.iloc[i][label])
            factors.append(
                {"feature": label, "value": shown, "contribution": round(float(value), 4)}
            )
        results.append(
            {
                "risk_score": float(prob[i]),
                "times_train_base_rate": float(prob[i] / base_rate),
                "base_value": float(base_value[i]),
                "top_factors": factors,
                "other_contribution": round(float(row.sum() - top.sum()), 4),
            }
        )
    return results