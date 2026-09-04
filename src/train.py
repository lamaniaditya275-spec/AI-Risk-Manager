"""
End-to-end training script.

Usage:
    python train.py path/to/your_dataset.csv

Your CSV should match SCHEMA.md (or have config.py's COLUMN_MAP pointed at
its actual column names). This script will:
  1. Load + build features
  2. Time-based train/val/test split
  3. Train the model
  4. Run the ablation comparison (rules vs ML vs ML+graph)
  5. Score the test set and run cost-based evaluation
  6. Save the trained model to trained_model.pkl for api.py to load
  7. Save a sample of decision records to decisions_sample.json for the dashboard
"""

import sys
import json
import pickle
from pathlib import Path

# Ensure src directory is in sys.path regardless of working directory
src_dir = Path(__file__).resolve().parent
project_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd

from features import build_features, get_model_feature_columns
from model import time_based_split, train_model, score_orders, run_ablation
from evaluate import full_report
from risk_engine import decide_batch


def main(csv_path: str):
    print(f"Loading dataset from {csv_path}")
    raw_df = pd.read_csv(csv_path)
    print(f"Loaded {len(raw_df)} rows")

    print("\nBuilding features...")
    feat_df = build_features(raw_df)

    print("\nSplitting by time (train/val/test)...")
    train_df, val_df, test_df = time_based_split(feat_df)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    print("\nTraining model...")
    trained = train_model(train_df, val_df)

    print("\nRunning ablation (rules-only vs ML vs ML+graph)...")
    run_ablation(train_df, val_df)

    print("\nScoring test set...")
    scored_test = score_orders(trained, test_df)

    print("\n=== Cost-based evaluation on TEST set ===")
    # Pass CLV proxy for customer-value-weighted false-decline costs
    clv_proxy = scored_test["customer_lifetime_value_proxy"] if "customer_lifetime_value_proxy" in scored_test.columns else None
    report = full_report(scored_test["label_fraud"], scored_test["risk_score"], clv_proxy=clv_proxy)

    # ---------- Generate decisions for full test set ----------
    print("\nGenerating decision records for ALL test orders (this may take a while)...")
    all_records = decide_batch(scored_test, trained.feature_columns, trained.model_version)

    # ---------- Build a representative sample for the dashboard ----------
    # Instead of head(50) which skews to low-risk, take a stratified sample
    # so the dashboard shows a realistic mix of all action types.
    records_by_action = {}
    for rec in all_records:
        records_by_action.setdefault(rec["action"], []).append(rec)

    sample_records = []
    target_total = 50
    # Prioritize rare/interesting actions so they're visible in the dashboard
    action_priority = [
        "BLOCK_OR_FORCE_PREPAID",
        "STEP_UP_VERIFICATION",
        "MANUAL_REVIEW",
        "APPROVE",
        "APPROVE_WITH_MONITORING",
    ]
    remaining = target_total
    for action_name in action_priority:
        bucket = records_by_action.get(action_name, [])
        # Take up to 15 from rare actions, fill the rest with common ones
        take = min(len(bucket), max(5, remaining // max(1, len(action_priority))))
        sample_records.extend(bucket[:take])
        remaining = target_total - len(sample_records)
        if remaining <= 0:
            break
    # Fill any remaining slots with APPROVE_WITH_MONITORING (most common)
    if remaining > 0:
        filler = records_by_action.get("APPROVE_WITH_MONITORING", [])
        sample_records.extend(filler[:remaining])

    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = artifacts_dir / "decisions_sample.json"

    with open(decisions_path, "w") as f:
        json.dump(sample_records[:target_total], f, indent=2, default=str)
    print(f"Saved {decisions_path} ({len(sample_records[:target_total])} stratified sample decisions)")

    # ---------- Print summary of new features ----------
    print("\n=== Step-up Verification & Action Distribution (full test set) ===")
    actions = pd.Series([r["action"] for r in all_records])
    action_counts = actions.value_counts()
    for action_name, count in action_counts.items():
        print(f"  {action_name}: {count}")

    stepup_count = int(action_counts.get("STEP_UP_VERIFICATION", 0))
    block_count = int(action_counts.get("BLOCK_OR_FORCE_PREPAID", 0))
    print(f"\n  STEP_UP_VERIFICATION: {stepup_count}  vs  BLOCK_OR_FORCE_PREPAID: {block_count}")

    # Trust adjustment stats
    if "trust_adjustment" in scored_test.columns:
        adj = scored_test["trust_adjustment"]
        n_adjusted = (adj > 0).sum()
        print(f"\n=== Trust Adjustment Stats ===")
        print(f"  Orders with trust adjustment > 0: {n_adjusted} / {len(scored_test)}")
        if n_adjusted > 0:
            print(f"  Mean adjustment (where > 0): {adj[adj > 0].mean():.4f}")
            print(f"  Max adjustment: {adj.max():.4f}")
        print("  (trust_adjustment uses only PAST orders per customer — no leakage)")

    model_path = artifacts_dir / "trained_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained, f)
    print(f"\nSaved {model_path}")

    print("\nDone. Next: run `streamlit run src/dashboard.py` or `uvicorn src.api:app --reload`")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train.py path/to/your_dataset.csv")
        sys.exit(1)
    main(sys.argv[1])
