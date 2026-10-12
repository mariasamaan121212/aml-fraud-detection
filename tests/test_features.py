import numpy as np
import pandas as pd

from src.features import CHANNELS, FEATURE_COLUMNS, make_features

BASE = dict(
    amount=1000.0, payment_channel="ACH", time_since_last_transaction=60.0,
    velocity_score=5, spending_deviation_score=0.1, is_first_transaction=0,
    hour=10, day_of_week=2, is_weekend=0,
    is_cross_bank_transfer=1, is_cross_currency_transfer=0,
    is_new_receiver=0, is_new_bank=0, is_new_payment_format=0,
)


def raw_frame(*overrides):
    return pd.DataFrame([{**BASE, **o} for o in overrides])


def test_columns_come_in_the_expected_order():
    out = make_features(raw_frame({}))
    assert list(out.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 23


def test_each_known_channel_sets_exactly_one_column():
    out = make_features(raw_frame(*[{"payment_channel": c} for c in CHANNELS]))
    channel_cols = [f"channel_{c}" for c in CHANNELS]
    assert (out[channel_cols].sum(axis=1) == 1).all()
    for i, c in enumerate(CHANNELS):
        assert out.loc[i, f"channel_{c}"] == 1


def test_unknown_channel_sets_no_channel_column():
    out = make_features(raw_frame({"payment_channel": "Carrier pigeon"}))
    assert out[[f"channel_{c}" for c in CHANNELS]].to_numpy().sum() == 0


def test_burst_needs_zero_gap_and_a_known_sender():
    out = make_features(raw_frame(
        {"time_since_last_transaction": 0.0, "is_first_transaction": 0},
        {"time_since_last_transaction": 0.0, "is_first_transaction": 1},
        {"time_since_last_transaction": 60.0, "is_first_transaction": 0},
    ))
    assert out["burst"].tolist() == [1, 0, 0]


def test_has_history_starts_at_two_previous_transactions():
    out = make_features(raw_frame({"velocity_score": 0}, {"velocity_score": 1}, {"velocity_score": 2}))
    assert out["has_history"].tolist() == [0, 0, 1]


def test_amount_is_log_transformed():
    out = make_features(raw_frame({"amount": 0.0}, {"amount": 1000.0}))
    assert np.allclose(out["log_amount"], np.log1p([0.0, 1000.0]), atol=1e-4)


def test_hour_is_encoded_as_a_circle():
    out = make_features(raw_frame({"hour": 0}, {"hour": 23}))
    gap = np.hypot(out.loc[0, "hour_sin"] - out.loc[1, "hour_sin"],
                   out.loc[0, "hour_cos"] - out.loc[1, "hour_cos"])
    assert gap < 0.3