"""
Deterministic rules engine.

Rules exist for cases where you want a guaranteed, explainable, non-probabilistic
control — not "the model was 87% sure," but "this exact condition was true."
Rules run independently of the ML model and can force a decision regardless
of model score (see risk_engine.py).

Each rule takes a single feature-row (pandas Series, i.e. one order after
build_features() has run) and returns either None (rule did not fire) or a
dict describing the triggered rule.
"""

from typing import Optional, List, Dict


def rule_device_ring(row) -> Optional[Dict]:
    val = row.get("distinct_customer_id_per_device_id_prior", 0)
    if val >= 3:
        return {
            "rule_id": "DEVICE_RING_001",
            "severity": "high",
            "reason": f"Device linked to {int(val)} other customer accounts prior to this order",
        }
    return None


def rule_address_ring(row) -> Optional[Dict]:
    val = row.get("distinct_customer_id_per_address_id_prior", 0)
    if val >= 3:
        return {
            "rule_id": "ADDRESS_RING_001",
            "severity": "high",
            "reason": f"Shipping address linked to {int(val)} other customer accounts prior to this order",
        }
    return None


def rule_promo_velocity(row) -> Optional[Dict]:
    val = row.get("velocity_promo_code_24h", 0)
    if val >= 5:
        return {
            "rule_id": "PROMO_VELOCITY_001",
            "severity": "high",
            "reason": f"This promo code used {int(val)} times across all customers in the last 24h",
        }
    return None


def rule_new_account_promo_ring(row) -> Optional[Dict]:
    is_new_promo = row.get("new_account_first_promo", 0)
    device_ring = row.get("distinct_customer_id_per_device_id_prior", 0)
    if is_new_promo == 1 and device_ring >= 1:
        return {
            "rule_id": "NEW_ACCOUNT_PROMO_RING_001",
            "severity": "medium",
            "reason": "New account's first order uses a promo code from a device already linked to another account",
        }
    return None


def rule_cod_device_velocity(row) -> Optional[Dict]:
    is_cod = row.get("is_cod", 0)
    device_1h = row.get("velocity_device_id_1h", 0)
    if is_cod == 1 and device_1h >= 3:
        return {
            "rule_id": "COD_DEVICE_VELOCITY_001",
            "severity": "medium",
            "reason": f"{int(device_1h)} COD-eligible orders from this device in the last hour",
        }
    return None


def rule_extreme_amount_new_account(row) -> Optional[Dict]:
    ratio = row.get("amount_vs_customer_median_ratio", 1.0)
    is_new = row.get("is_new_account_24h", 0)
    if is_new == 1 and ratio >= 5:
        return {
            "rule_id": "EXTREME_AMOUNT_NEW_ACCOUNT_001",
            "severity": "medium",
            "reason": f"Order amount is {ratio:.1f}x this new account's baseline",
        }
    return None


ALL_RULES = [
    rule_device_ring,
    rule_address_ring,
    rule_promo_velocity,
    rule_new_account_promo_ring,
    rule_cod_device_velocity,
    rule_extreme_amount_new_account,
]


def evaluate_rules(row) -> List[Dict]:
    """Run every rule against one feature-row, return list of triggered rules."""
    triggered = []
    for rule_fn in ALL_RULES:
        result = rule_fn(row)
        if result is not None:
            triggered.append(result)
    return triggered


def max_severity(triggered_rules: List[Dict]) -> Optional[str]:
    if not triggered_rules:
        return None
    order = {"low": 0, "medium": 1, "high": 2}
    return max(triggered_rules, key=lambda r: order.get(r["severity"], 0))["severity"]
