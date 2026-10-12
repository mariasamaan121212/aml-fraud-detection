import math

import pytest
from pydantic import ValidationError

from api.main import Transaction, score_transaction
from src.features import FEATURE_COLUMNS
from src.predict import load_model

PAYLOAD = dict(
    amount=11139.43, payment_channel="ACH", time_since_last_transaction=0.0,
    velocity_score=61, spending_deviation_score=0.5, hour=14, day_of_week=2,
    is_cross_bank_transfer=1, is_cross_currency_transfer=0,
    is_new_receiver=1, is_new_bank=1, is_new_payment_format=0,
)


def make(**overrides):
    return Transaction(**{**PAYLOAD, **overrides})


def test_saved_model_expects_the_columns_the_code_builds():
    _, meta = load_model()
    assert meta["feature_columns"] == FEATURE_COLUMNS


def test_valid_transaction_gets_a_score_between_0_and_1():
    result = score_transaction(make())
    assert 0.0 < result["risk_score"] < 1.0
    assert len(result["top_factors"]) == 5
    assert result["warnings"] == []


def test_contributions_add_up_to_the_model_score():
    result = score_transaction(make())
    p = result["risk_score"]
    log_odds = math.log(p / (1 - p))
    total = (result["base_value"]
             + sum(f["contribution"] for f in result["top_factors"])
             + result["other_contribution"])
    assert abs(total - log_odds) < 1e-3


def test_first_transaction_must_have_new_receiver_bank_and_format():
    with pytest.raises(ValidationError):
        make(velocity_score=0, time_since_last_transaction=0.0, spending_deviation_score=0.0,
             is_new_receiver=0, is_new_bank=1, is_new_payment_format=1)


def test_first_transaction_must_have_zero_gap():
    with pytest.raises(ValidationError):
        make(velocity_score=0, time_since_last_transaction=30.0, spending_deviation_score=0.0,
             is_new_receiver=1, is_new_bank=1, is_new_payment_format=1)


def test_deviation_score_must_be_zero_for_senders_without_history():
    with pytest.raises(ValidationError):
        make(velocity_score=1, spending_deviation_score=0.5)


def test_values_beyond_the_training_range_are_rejected():
    with pytest.raises(ValidationError):
        make(amount=4e12)


def test_unusual_but_possible_amount_is_scored_with_a_warning():
    result = score_transaction(make(amount=5e9))
    assert any("amount" in w for w in result["warnings"])