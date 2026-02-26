from pydantic import BaseModel
from typing import Optional


class TransactionInput(BaseModel):
    TransactionAmt: float
    card1: Optional[int] = 0
    card2: Optional[float] = 0.0
    card3: Optional[float] = 0.0
    card5: Optional[float] = 0.0
    addr1: Optional[float] = 0.0
    addr2: Optional[float] = 0.0
    dist1: Optional[float] = 0.0
    P_emaildomain: Optional[int] = 0
    R_emaildomain: Optional[int] = 0
    hour: Optional[int] = 0
    day: Optional[int] = 0
    is_night: Optional[int] = 0
    amt_log: Optional[float] = 0.0
    amt_cents: Optional[int] = 0
    is_round_amount: Optional[int] = 0
    card1_amt_mean: Optional[float] = 0.0
    card1_amt_std: Optional[float] = 0.0
    card1_amt_max: Optional[float] = 0.0
    card1_count: Optional[int] = 0
    card1_amt_diff: Optional[float] = 0.0

    class Config:
        extra = "allow"


class PredictionOutput(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float = 0.5