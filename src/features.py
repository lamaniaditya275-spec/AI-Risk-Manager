"""
Feature engineering for the Risk Manager model.

LEAKAGE SAFETY: every feature here is computed using only data that would be
available strictly BEFORE order_timestamp for that specific order. Rolling
windows are computed on a per-row basis relative to that row's own timestamp,
not with a global groupby that could leak future information backwards.

Feature families:
  1. Velocity     — how often has this entity (device/phone/address) acted recently
  2. Consistency   — does this order look "coherent" with the entity's history
  3. Graph-lite    — how many distinct entities share this device/phone/address
                      (a cheap stand-in for graph/ring detection, no GNN needed)
  4. Baseline      — how unusual is this order relative to the customer's own history
"""

import pandas as pd
import numpy as np
from config import COLUMN_MAP, VELOCITY_WINDOWS_HOURS, TRUST_DECAY


def _c(name: str) -> str:
    """Resolve internal column name -> actual dataset column name."""
    return COLUMN_MAP[name]


def load_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns per COLUMN_MAP -> internal names, parse timestamps,
    sort by time. Call this first, on raw data, before anything else.
    """
    inv_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=inv_map)

    for ts_col in ["order_timestamp", "account_created_at"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col])

    df = df.sort_values("order_timestamp").reset_index(drop=True)
    return df


def _rolling_count_before(df: pd.DataFrame, entity_col: str, window_hours: int) -> pd.Series:
    """
    For each row, count how many PRIOR rows (strictly before this row's
    timestamp) share the same entity_col value, within window_hours.
    Implemented with a per-entity expanding approach — O(n log n), safe for
    buildathon-scale data (tens of thousands of rows).
    """
    out = pd.Series(0, index=df.index, dtype=int)
    for entity_value, group in df.groupby(entity_col):
        idx = group.index.to_numpy()
        times = group["order_timestamp"].to_numpy()
        window = np.timedelta64(window_hours, "h")
        left = np.searchsorted(times, times - window, side="left")
        # count of prior rows in window = position - left_bound (excludes self)
        positions = np.arange(len(times))
        counts = positions - left
        out.loc[idx] = counts
    return out


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling counts per device / phone / address / customer, several window sizes."""
    for entity_col in ["device_id", "phone", "address_id", "customer_id"]:
        if entity_col not in df.columns:
            continue
        for hours in VELOCITY_WINDOWS_HOURS:
            label = f"{hours}h" if hours < 24 else f"{hours // 24}d"
            df[f"velocity_{entity_col}_{label}"] = _rolling_count_before(df, entity_col, hours)

    # Promo-specific velocity: how many times has this promo code been used
    # by ANY customer in the trailing 24h (proxy for coordinated promo abuse)
    if "promo_code" in df.columns:
        promo_df = df[df["promo_code"].notna() & (df["promo_code"] != "")]
        promo_counts = _rolling_count_before(promo_df, "promo_code", 24) if len(promo_df) else pd.Series(dtype=int)
        df["velocity_promo_code_24h"] = 0
        df.loc[promo_counts.index, "velocity_promo_code_24h"] = promo_counts.values

    return df


def add_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flags for 'too many new things at once' — classic ring/abuse signal."""
    if "account_created_at" in df.columns:
        df["account_age_hours"] = (
            (df["order_timestamp"] - df["account_created_at"]).dt.total_seconds() / 3600
        ).clip(lower=0)
        df["is_new_account_24h"] = (df["account_age_hours"] <= 24).astype(int)
    else:
        df["account_age_hours"] = np.nan
        df["is_new_account_24h"] = 0

    if "payment_method" in df.columns:
        df["is_cod"] = (df["payment_method"].astype(str).str.upper() == "COD").astype(int)
    else:
        df["is_cod"] = 0

    df["uses_promo"] = (
        df["promo_code"].notna() & (df["promo_code"].astype(str) != "")
    ).astype(int) if "promo_code" in df.columns else 0

    # "First promo use on a brand-new account" — a specific, explainable combo signal
    df["new_account_first_promo"] = ((df["is_new_account_24h"] == 1) & (df["uses_promo"] == 1)).astype(int)

    return df


def add_graph_lite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cheap ring-detection features: how many DISTINCT customers/orders share
    this device / phone / address, computed only from data up to (not
    including) this order. No GNN needed — this is what the research doc
    calls "graph-derived aggregates," and it's what actually catches rings.
    """
    for entity_col, target_col in [
        ("device_id", "customer_id"),
        ("phone", "customer_id"),
        ("address_id", "customer_id"),
    ]:
        if entity_col not in df.columns:
            continue
        out = pd.Series(0, index=df.index, dtype=int)
        for entity_value, group in df.sort_values("order_timestamp").groupby(entity_col):
            seen = set()
            counts = []
            for cust in group[target_col]:
                counts.append(len(seen))  # distinct customers seen so far, excluding current row
                seen.add(cust)
            out.loc[group.index.to_numpy()] = counts
        df[f"distinct_{target_col}_per_{entity_col}_prior"] = out

    return df


def add_customer_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """How unusual is this order vs. this customer's OWN prior order history."""
    df["customer_prior_order_count"] = 0
    df["amount_vs_customer_median_ratio"] = 1.0

    for customer_id, group in df.groupby("customer_id"):
        idx = group.index.to_numpy()
        amounts = group["order_amount"].to_numpy()
        prior_counts = np.arange(len(amounts))
        df.loc[idx, "customer_prior_order_count"] = prior_counts

        running_median = np.zeros(len(amounts))
        for i in range(len(amounts)):
            if i == 0:
                running_median[i] = np.nan
            else:
                running_median[i] = np.median(amounts[:i])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = amounts / running_median
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        df.loc[idx, "amount_vs_customer_median_ratio"] = ratio

    return df


def add_customer_lifetime_value_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    CLV proxy = (customer_prior_order_count + 1) * mean_historical_order_amount.
    Used by evaluate.py for weighted false-decline costs, NOT as a model feature.
    Computed strictly from each customer's own past orders (no leakage).
    """
    df["customer_lifetime_value_proxy"] = 0.0

    for customer_id, group in df.groupby("customer_id"):
        idx = group.index.to_numpy()
        amounts = group["order_amount"].to_numpy()
        prior_counts = group["customer_prior_order_count"].to_numpy()

        # Running mean of prior order amounts (excludes current order)
        running_mean = np.zeros(len(amounts))
        for i in range(len(amounts)):
            if i == 0:
                # No prior orders — use current order amount as baseline
                running_mean[i] = amounts[i]
            else:
                running_mean[i] = np.mean(amounts[:i])

        clv_proxy = (prior_counts + 1) * running_mean
        df.loc[idx, "customer_lifetime_value_proxy"] = clv_proxy

    return df


def add_trust_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trust decay: for each order, count how many of this customer's PRIOR orders
    had a high risk score or were flagged (heuristic: risk-indicating features)
    BUT turned out legitimate (label_fraud == 0). This gives "prior_cleared_flags".

    The trust_adjustment = min(0.05 * prior_cleared_flags, 0.20) and is subtracted
    from risk_score in risk_engine.py.

    LEAKAGE SAFETY: only looks at rows strictly BEFORE the current row for the
    same customer. Uses label_fraud from past orders only (training-time ground
    truth), never from the current or future rows.
    """
    adj_per = TRUST_DECAY["adjustment_per_cleared_flag"]
    max_adj = TRUST_DECAY["max_adjustment"]

    df["prior_cleared_flags"] = 0
    df["trust_adjustment"] = 0.0

    if "label_fraud" not in df.columns:
        return df

    for customer_id, group in df.groupby("customer_id"):
        idx = group.index.to_numpy()
        labels = group["label_fraud"].to_numpy()
        # A "flagged but cleared" prior order = one where the customer had
        # risk-indicating signals (high velocity, ring indicators, etc.)
        # but label_fraud == 0. As a practical proxy that doesn't require
        # access to past model scores (which would be circular), we count
        # past orders where:
        #   - The customer's velocity was elevated (>= 2 prior orders from
        #     same customer within 7 days) OR amount ratio was high (>= 2x), AND
        #   - label_fraud == 0
        # This is conservative and avoids any leakage.
        velocities = group.get("velocity_customer_id_7d",
                              pd.Series(0, index=group.index)).to_numpy()
        ratios = group.get("amount_vs_customer_median_ratio",
                           pd.Series(1.0, index=group.index)).to_numpy()

        was_flaggy = (velocities >= 2) | (ratios >= 2.0)
        was_legit = (labels == 0)
        was_cleared = was_flaggy & was_legit

        prior_cleared = np.zeros(len(idx), dtype=int)
        running_count = 0
        for i in range(len(idx)):
            prior_cleared[i] = running_count
            if was_cleared[i]:
                running_count += 1

        df.loc[idx, "prior_cleared_flags"] = prior_cleared
        df.loc[idx, "trust_adjustment"] = np.minimum(
            adj_per * prior_cleared, max_adj
        )

    return df


FEATURE_COLUMNS = [
    # filled in dynamically by build_features(), listed here for reference only
]


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: raw dataset -> normalized -> all feature families.
    Returns a dataframe with original id/label columns PLUS engineered features.
    Call this on train/val/test slices SEPARATELY if you want strict
    leakage safety across the split (recommended), or on the full sorted
    dataset if your rolling-window logic is already time-respecting (it is,
    by construction, since every window only looks backward from each row).
    """
    df = load_and_normalize(raw_df)
    df = add_velocity_features(df)
    df = add_consistency_features(df)
    df = add_graph_lite_features(df)
    df = add_customer_baseline_features(df)
    df = add_customer_lifetime_value_proxy(df)
    df = add_trust_adjustment(df)
    return df


def get_model_feature_columns(df: pd.DataFrame) -> list:
    """
    Everything except IDs, raw timestamps, and the label — i.e. what actually
    goes into the model. Kept as a function (not a static list) so it adapts
    to whichever optional columns were present.
    """
    exclude = {
        "order_id", "customer_id", "device_id", "phone", "address_id",
        "promo_code", "payment_method", "order_timestamp", "account_created_at",
        "label_fraud",
        # CLV proxy is for evaluation cost weighting, not a model input
        "customer_lifetime_value_proxy",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype != "object"]


if __name__ == "__main__":
    print("This module is a library — import build_features() from your training script.")
    print("See SCHEMA.md for the expected input format and config.py for column mapping.")
