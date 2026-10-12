from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from src.limits import HARD, USUAL
from src.predict import load_model, score

Channel = Literal["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
Flag = Literal[0, 1]

NOTE = (
    "risk_score is a model score learned from a synthetic dataset, not a verified fraud "
    "probability. Contributions are log-odds differences relative to an average transaction "
    "(base_value); they describe what the model used, not what caused fraud."
)


class Transaction(BaseModel):
    amount: float = Field(ge=HARD["amount"][0], le=HARD["amount"][1])
    payment_channel: Channel
    time_since_last_transaction: float = Field(
        ge=HARD["time_since_last_transaction"][0], le=HARD["time_since_last_transaction"][1],
        description="Seconds since the sender's previous transaction (0 if none).",
    )
    velocity_score: int = Field(
        ge=HARD["velocity_score"][0], le=HARD["velocity_score"][1],
        description="Number of transactions the sender made before this one.",
    )
    spending_deviation_score: float = Field(
        ge=HARD["spending_deviation_score"][0], le=HARD["spending_deviation_score"][1]
    )
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="0 = Monday, 6 = Sunday")
    is_cross_bank_transfer: Flag
    is_cross_currency_transfer: Flag
    is_new_receiver: Flag
    is_new_bank: Flag
    is_new_payment_format: Flag

    @model_validator(mode="after")
    def check_consistency(self):
        if self.velocity_score == 0:
            if not (self.is_new_receiver == 1 and self.is_new_bank == 1
                    and self.is_new_payment_format == 1):
                raise ValueError(
                    "velocity_score = 0 means a first transaction: is_new_receiver, "
                    "is_new_bank and is_new_payment_format must all be 1"
                )
            if self.time_since_last_transaction != 0:
                raise ValueError("velocity_score = 0 requires time_since_last_transaction = 0")
        if self.velocity_score <= 1 and self.spending_deviation_score != 0:
            raise ValueError(
                "spending_deviation_score must be 0 when velocity_score is 0 or 1 "
                "(no usable history in the training data)"
            )
        return self


def usual_range_warnings(t: Transaction) -> list[str]:
    out = []
    for name, (lo, hi) in USUAL.items():
        value = getattr(t, name)
        if value < lo or value > hi:
            out.append(
                f"{name}={value} is outside the range of 99.8% of training rows ({lo} to {hi})"
            )
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="AML fraud scoring API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    _, meta = load_model()
    return {
        "status": "ok",
        "model_file": meta["model_file"],
        "trained_at_utc": meta["trained_at_utc"],
        "n_features": len(meta["feature_columns"]),
    }
def score_transaction(t: Transaction) -> dict:
    row = t.model_dump()
    row["is_weekend"] = int(t.day_of_week >= 5)
    row["is_first_transaction"] = int(t.velocity_score == 0)
    result = score(pd.DataFrame([row]))[0]
    result["warnings"] = usual_range_warnings(t)
    result["note"] = NOTE
    return result


@app.post("/predict")
def predict(t: Transaction):
    return score_transaction(t)