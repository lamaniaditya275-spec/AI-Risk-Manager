"""
Risk decision engine.

Combines three signals into one decision:
  1. Rules engine (deterministic, can force escalation regardless of score)
  2. Calibrated ML risk_score (probability of fraud/abuse)
  3. Anomaly score (catches patterns not in training labels)

Produces a single audit-ready JSON record per order — this record is what
your reviewer dashboard reads and what you'd show a judge as "here's exactly
why the system made this call."

Actions include STEP_UP_VERIFICATION for high-risk orders (customer gets
a chance to verify themselves via OTP before being blocked/escalated),
plus a recovery_path field describing what the customer can do next.

Trust decay: customers who previously went through flagging but were NOT
confirmed fraudulent receive a small downward risk-score adjustment.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from config import RISK_BANDS, RISK_BAND_ACTIONS, RECOVERY_PATHS, MODEL_VERSION, TRUST_DECAY
from rules import evaluate_rules, max_severity


def _risk_band(score: float) -> str:
    if score <= RISK_BANDS["low_max"]:
        return "low"
    elif score <= RISK_BANDS["medium_max"]:
        return "medium"
    elif score <= RISK_BANDS["high_max"]:
        return "high"
    return "very_high"


def _feature_snapshot_hash(row: pd.Series, feature_columns: List[str]) -> str:
    payload = {c: (None if pd.isna(row.get(c)) else float(row.get(c))) for c in feature_columns}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]


def _apply_trust_decay(risk_score: float, trust_adjustment: float) -> float:
    """
    Subtract the trust adjustment from the raw risk score.
    Clamp to [0, 1] so the score stays valid.
    """
    adjusted = risk_score - trust_adjustment
    return max(0.0, min(1.0, adjusted))


def decide(row: pd.Series, feature_columns: List[str], model_version: str = MODEL_VERSION) -> Dict:
    """
    row must already have 'risk_score' and 'anomaly_score' columns (i.e. run
    score_orders() from model.py first). Returns the full audit-record dict.
    """
    triggered_rules = evaluate_rules(row)
    rule_severity = max_severity(triggered_rules)

    raw_risk_score = row.get("risk_score", 0.0)
    trust_adj = row.get("trust_adjustment", 0.0)
    adjusted_risk_score = _apply_trust_decay(raw_risk_score, trust_adj)

    ml_band = _risk_band(adjusted_risk_score)

    # Rules escalate proportionally — bump up by one band, not straight to max.
    # This way a low-risk order with a rule firing goes to monitoring, not
    # straight to block. Only if the ML ALSO says high risk does it reach
    # very_high. Deterministic controls still override ML, just proportionally.
    _ESCALATION = {
        "low": "medium",
        "medium": "high",
        "high": "very_high",
        "very_high": "very_high",
    }
    if rule_severity == "high":
        final_band = _ESCALATION[ml_band]
    elif rule_severity == "medium":
        # Medium-severity rules only escalate if ML already flags some risk
        if ml_band in ("medium", "high"):
            final_band = _ESCALATION[ml_band]
        else:
            final_band = ml_band
    else:
        final_band = ml_band

    action = RISK_BAND_ACTIONS[final_band]
    recovery_path = RECOVERY_PATHS.get(action)

    # Build human-readable top reasons: rule reasons first, then a note on
    # the ML score itself, then anomaly if notably high.
    top_reasons = [r["reason"] for r in triggered_rules]
    if trust_adj > 0:
        top_reasons.append(
            f"Model risk score: {raw_risk_score:.2f} → {adjusted_risk_score:.2f} "
            f"(trust adjustment: -{trust_adj:.2f} from {int(row.get('prior_cleared_flags', 0))} "
            f"prior cleared flags)"
        )
    else:
        top_reasons.append(f"Model risk score: {adjusted_risk_score:.2f}")
    if row.get("anomaly_score", 0.0) >= 0.8:
        top_reasons.append(f"Unusual pattern not well-represented in training data (anomaly score {row.get('anomaly_score'):.2f})")

    record = {
        "decision_id": f"dec_{row.get('order_id', 'unknown')}",
        "order_id": row.get("order_id"),
        "customer_id": row.get("customer_id"),
        "action": action,
        "recovery_path": recovery_path,
        "risk_band": final_band,
        "risk_score": round(float(adjusted_risk_score), 4),
        "risk_score_raw": round(float(raw_risk_score), 4),
        "trust_adjustment": round(float(trust_adj), 4),
        "anomaly_score": round(float(row.get("anomaly_score", 0.0)), 4),
        "model_version": model_version,
        "triggered_rules": triggered_rules,
        "top_reasons": top_reasons,
        "feature_snapshot_hash": _feature_snapshot_hash(row, feature_columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return record


def decide_batch(scored_df: pd.DataFrame, feature_columns: List[str], model_version: str = MODEL_VERSION) -> List[Dict]:
    """Run decide() across an entire scored dataframe."""
    return [decide(row, feature_columns, model_version) for _, row in scored_df.iterrows()]
