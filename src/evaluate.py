"""
Cost-based evaluation.

A bare accuracy/precision number doesn't tell a judge (or a merchant) what
threshold to actually use. This module produces a threshold table showing
the real tradeoff in rupees, using the cost model in config.py, plus
standard metrics (PR-AUC, calibration).

Expected cost formula (per the research brief):
  expected_cost = FN * C_missed_fraud
                 + SUM_FP[ base_false_decline * (1 + log(1 + CLV_proxy)) ]
                 + reviews * C_manual_review

The false-decline cost is now weighted by customer lifetime value proxy
so that wrongly blocking a repeat/high-value customer costs more — but
the log scaling prevents unbounded growth.

Insult rate = FP / (FP + TN) = fraction of legitimate customers wrongly flagged.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from config import COST_MODEL


def compute_pr_auc(y_true, y_score) -> float:
    return average_precision_score(y_true, y_score)


def threshold_table(
    y_true: pd.Series,
    y_score: pd.Series,
    clv_proxy: pd.Series = None,
    thresholds=None,
) -> pd.DataFrame:
    """
    For each threshold, treat score >= threshold as "flagged" (review/block)
    and compute precision, recall, insult rate, false declines, and expected
    cost (with customer-value-weighted false-decline cost).

    NOTE: this is a simplified two-outcome view (flag vs approve). Your
    actual risk_engine.py has four bands — use this table to help you decide
    where the medium/high/very_high cutoffs in config.py should sit.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    y_true_np = y_true.astype(int).to_numpy()
    y_score_np = y_score.to_numpy()

    # CLV proxy for weighted false-decline cost
    if clv_proxy is not None:
        clv_np = clv_proxy.to_numpy()
    else:
        clv_np = np.zeros(len(y_true_np))

    base_fd_cost = COST_MODEL["false_decline"]

    for t in thresholds:
        flagged = (y_score_np >= t).astype(int)
        tp = int(((flagged == 1) & (y_true_np == 1)).sum())
        fp = int(((flagged == 1) & (y_true_np == 0)).sum())
        fn = int(((flagged == 0) & (y_true_np == 1)).sum())
        tn = int(((flagged == 0) & (y_true_np == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        insult_rate = fp / (fp + tn) if (fp + tn) else float("nan")
        reviews = tp + fp  # everything flagged goes to review/block

        # Weighted false-decline cost: each FP costs
        #   base_false_decline * (1 + log(1 + customer_lifetime_value_proxy))
        fp_mask = (flagged == 1) & (y_true_np == 0)
        if fp_mask.any():
            fp_clv = clv_np[fp_mask]
            weighted_fd_cost = float(
                (base_fd_cost * (1 + np.log(1 + fp_clv))).sum()
            )
        else:
            weighted_fd_cost = 0.0

        expected_cost = (
            fn * COST_MODEL["missed_fraud"]
            + weighted_fd_cost
            + reviews * COST_MODEL["manual_review"]
        )

        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(precision, 4) if precision == precision else None,
            "recall": round(recall, 4) if recall == recall else None,
            "insult_rate": round(insult_rate, 4) if insult_rate == insult_rate else None,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "reviews_triggered": reviews,
            "expected_cost_rupees": int(expected_cost),
        })

    df = pd.DataFrame(rows)
    return df


def best_threshold_by_cost(table: pd.DataFrame) -> dict:
    """Pick the threshold that minimizes expected cost."""
    best_row = table.loc[table["expected_cost_rupees"].idxmin()]
    return best_row.to_dict()


def calibration_table(y_true: pd.Series, y_score: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """
    Bins predictions and compares mean predicted score vs actual fraud rate
    per bin — shows whether risk_score is a genuine probability (well
    calibrated) or just a ranking.
    """
    df = pd.DataFrame({"y_true": y_true.astype(int).to_numpy(), "y_score": y_score.to_numpy()})
    df["bin"] = pd.qcut(df["y_score"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("y_score", "mean"),
        actual_fraud_rate=("y_true", "mean"),
        count=("y_true", "size"),
    ).reset_index(drop=True)
    return grouped


def full_report(
    y_true: pd.Series,
    y_score: pd.Series,
    clv_proxy: pd.Series = None,
) -> dict:
    """Convenience wrapper: run everything and print a summary."""
    pr_auc = compute_pr_auc(y_true, y_score)
    table = threshold_table(y_true, y_score, clv_proxy=clv_proxy)
    best = best_threshold_by_cost(table)
    calib = calibration_table(y_true, y_score)

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"\nBest threshold by expected cost: {best['threshold']} "
          f"(precision={best['precision']}, recall={best['recall']}, "
          f"expected_cost=INR {best['expected_cost_rupees']})")

    # Insult rate — the key metric for the pitch
    insult = best.get("insult_rate")
    if insult is not None:
        print(f"\n*** Insult rate (legit customers wrongly flagged): {insult:.4f} "
              f"({insult * 100:.2f}%) at threshold {best['threshold']} ***")

    print("\nThreshold table:")
    print(table.to_string(index=False))
    print("\nCalibration (predicted vs actual fraud rate per bin):")
    print(calib.to_string(index=False))

    return {
        "pr_auc": pr_auc,
        "threshold_table": table,
        "best_threshold": best,
        "calibration": calib,
    }
