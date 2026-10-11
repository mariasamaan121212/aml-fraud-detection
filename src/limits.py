# Ranges copied from describe() on benchmark/train.parquet.
HARD = {
    "amount": (0.0, 3.645e12),
    "time_since_last_transaction": (0.0, 1_194_480.0),
    "velocity_score": (0, 222_036),
    "spending_deviation_score": (-7.15e7, 2.02e8),
}

# 0.1% and 99.9% points of the training data. Outside them we only warn.
USUAL = {
    "amount": (0.0, 3.66e8),
    "time_since_last_transaction": (0.0, 749_460.0),
    "velocity_score": (0, 209_915),
    "spending_deviation_score": (-3.94, 1543.93),
}