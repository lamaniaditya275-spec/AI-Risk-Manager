"""
Central configuration. If your dataset (from Kaggle / ChatGPT / wherever) uses
different column names than SCHEMA.md, change COLUMN_MAP here — nothing else
in the pipeline needs to change.

Left side  = internal name used everywhere in this codebase.
Right side = the column name in YOUR raw CSV/dataframe.

If your dataset already matches SCHEMA.md exactly, leave this as identity mapping.
"""

COLUMN_MAP = {
    "order_id": "order_id",
    "customer_id": "customer_id",
    "device_id": "device_id",
    "phone": "phone",
    "address_id": "address_id",
    "promo_code": "promo_code",
    "order_amount": "order_amount",
    "payment_method": "payment_method",
    "order_timestamp": "order_timestamp",
    "account_created_at": "account_created_at",
    "label_fraud": "label_fraud",
}

# Rolling window sizes (in hours) used for velocity features
VELOCITY_WINDOWS_HOURS = [1, 24, 24 * 7]

# Cost model (in ₹) — used by evaluate.py for threshold selection.
# Edit these to match whatever numbers you want to defend in the pitch.
COST_MODEL = {
    "missed_fraud": 1000,     # cost of a false negative (fraud/abuse we approved)
    "false_decline": 250,     # cost of a false positive (legit order we blocked/held)
    "manual_review": 20,      # cost of routing an order to human review
    "rto_cost": 180,          # cost of a COD return-to-origin event
}

# Risk band thresholds on the final calibrated risk score (0-1).
# Tune these after looking at the threshold table evaluate.py produces.
RISK_BANDS = {
    "low_max": 0.35,
    "medium_max": 0.65,
    "high_max": 0.85,
    # anything above high_max => "very_high"
}

RISK_BAND_ACTIONS = {
    "low": "APPROVE",
    "medium": "APPROVE_WITH_MONITORING",
    "high": "STEP_UP_VERIFICATION",
    "very_high": "BLOCK_OR_FORCE_PREPAID",
}

# What the customer can do next after each action
RECOVERY_PATHS = {
    "APPROVE": None,
    "APPROVE_WITH_MONITORING": None,
    "STEP_UP_VERIFICATION": "Verify via OTP to proceed",
    "MANUAL_REVIEW": "Order held for manual review, resolution within 24h",
    "BLOCK_OR_FORCE_PREPAID": None,
}

# Trust decay: customers previously flagged but confirmed NOT fraudulent
# get a small downward risk-score adjustment on future orders.
TRUST_DECAY = {
    "adjustment_per_cleared_flag": 0.05,  # subtracted per prior cleared flag
    "max_adjustment": 0.20,               # cap so it doesn't zero out risk
}

MODEL_VERSION = "risk_model_v1"
RANDOM_SEED = 42

# --- Currency & Reputation Metrics ---
USD_TO_INR_RATE = 83.0

# Reputation impact metrics for false declines
REPUTATION_MODEL = {
    "false_decline_churn_rate": 0.39,      # Citing Javelin Strategy & Research 2015
    "avg_customer_ltv_multiplier": 3.5     # Conservative industry midpoint
}

