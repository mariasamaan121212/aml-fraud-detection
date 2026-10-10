import numpy as np
import pandas as pd

CHANNELS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
BINARY_FLAGS = [
    "is_first_transaction",
    "is_weekend",
    "is_cross_bank_transfer",
    "is_cross_currency_transfer",
    "is_new_receiver",
    "is_new_bank",
    "is_new_payment_format",
]

FEATURE_COLUMNS = (
    [
        "log_amount",
        "log_time_since_last",
        "log_velocity",
        "spending_deviation_score",
        "burst",
        "has_history",
        "hour_sin",
        "hour_cos",
        "day_of_week",
    ]
    + BINARY_FLAGS
    + [f"channel_{c}" for c in CHANNELS]
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model features from raw transaction columns."""
    out = pd.DataFrame(index=df.index)

    out["log_amount"] = np.log1p(df["amount"]).astype("float32")
    out["log_time_since_last"] = np.log1p(df["time_since_last_transaction"]).astype("float32")
    out["log_velocity"] = np.log1p(df["velocity_score"]).astype("float32")
    out["spending_deviation_score"] = df["spending_deviation_score"].astype("float32")

    out["burst"] = (
        (df["time_since_last_transaction"] == 0) & (df["is_first_transaction"] == 0)
    ).astype("int8")
    out["has_history"] = (df["velocity_score"] >= 2).astype("int8")

    angle = 2 * np.pi * df["hour"] / 24
    out["hour_sin"] = np.sin(angle).astype("float32")
    out["hour_cos"] = np.cos(angle).astype("float32")
    out["day_of_week"] = df["day_of_week"].astype("int8")

    for col in BINARY_FLAGS:
        out[col] = df[col].astype("int8")

    for ch in CHANNELS:
        out[f"channel_{ch}"] = (df["payment_channel"] == ch).astype("int8")

    return out[FEATURE_COLUMNS]