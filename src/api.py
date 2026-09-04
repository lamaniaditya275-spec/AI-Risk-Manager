"""
FastAPI service.

Loads a trained model (pickled by train.py) and exposes:
  POST /score  -> takes one raw order (JSON), returns the full decision record

Run with: uvicorn api:app --reload --port 8000
"""

import pickle
import sys
from pathlib import Path

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
project_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from features import build_features
from model import score_orders
from risk_engine import decide

app = FastAPI(title="AI Risk Manager — Scoring API")

MODEL_PATH = project_root / "artifacts" / "trained_model.pkl"
_trained = None


def get_trained_model():
    global _trained
    if _trained is None:
        with open(MODEL_PATH, "rb") as f:
            _trained = pickle.load(f)
    return _trained


class OrderEvent(BaseModel):
    order_id: str
    customer_id: str
    device_id: str
    phone: str
    address_id: str
    promo_code: Optional[str] = None
    order_amount: float
    payment_method: str
    order_timestamp: str
    account_created_at: str


@app.post("/score")
def score_order(order: OrderEvent):
    """
    NOTE: for a single incoming order, velocity/graph-lite features need
    recent history to compute correctly. In this prototype we score against
    a small trailing-history buffer you'd maintain in production (a feature
    store). For the buildathon demo, pass in a short list of recent orders
    for the same entities alongside the new one — see /score_batch below,
    or wire this endpoint to a real feature store for production use.
    """
    trained = get_trained_model()
    df = pd.DataFrame([order.model_dump()])
    feat_df = build_features(df)
    scored = score_orders(trained, feat_df)
    record = decide(scored.iloc[0], trained.feature_columns, trained.model_version)
    return record


@app.post("/score_batch")
def score_batch(orders: list[OrderEvent]):
    """
    Score a batch together so velocity/graph-lite features have context.
    This is the more realistic way to demo the system: feed in a day's worth
    of orders (including the planted fraud patterns) and get back decisions
    for all of them with correct rolling-window features.
    """
    trained = get_trained_model()
    df = pd.DataFrame([o.model_dump() for o in orders])
    feat_df = build_features(df)
    scored = score_orders(trained, feat_df)
    records = [
        decide(row, trained.feature_columns, trained.model_version)
        for _, row in scored.iterrows()
    ]
    return records


@app.get("/health")
def health():
    return {"status": "ok"}
