"""
Preprocesses the IEEE-CIS Fraud Detection dataset (train_transaction.csv +
optionally train_identity.csv) into the schema our pipeline expects
(see SCHEMA.md). This dataset does NOT have clean customer_id/device_id/
phone/promo_code columns, so this script constructs proxies:

  order_id           <- TransactionID
  customer_id (proxy) <- constructed "uid": card1+card2+card3+card4+addr1
                          (a well-known trick from the original Kaggle
                          competition for approximating a real customer,
                          since no true customer ID is provided)
  device_id           <- DeviceInfo from train_identity.csv if available,
                          else falls back to card1 (weaker proxy)
  address_id          <- addr1 + addr2 combined
  phone                <- NOT AVAILABLE in this dataset. Left blank —
                          the pipeline gracefully skips phone-based
                          features when the column is absent.
  promo_code           <- NOT AVAILABLE in this dataset (it's a payment-
                          fraud dataset, not a promo-abuse one). Two
                          options, see bottom of this file.
  order_amount         <- TransactionAmt
  payment_method        <- card6 ("credit"/"debit") if present, else ProductCD
  order_timestamp        <- TransactionDT converted to a synthetic calendar
                          date (it's originally just seconds-since-reference)
  account_created_at   <- approximated as order_timestamp - D1 days
                          (D1 = "days since customer's account/card first seen",
                          a real feature in this dataset)
  label_fraud           <- isFraud

Usage:
    python preprocess_ieee.py train_transaction.csv output.csv
    python preprocess_ieee.py train_transaction.csv train_identity.csv output.csv

If you don't have train_identity.csv, device_id will fall back to card1
(weaker, but the pipeline still runs).
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

REFERENCE_DATE = datetime(2026, 1, 1)  # arbitrary anchor for TransactionDT -> real date


def main(transaction_path: str, identity_path: str = None, output_path: str = "data/processed/output.csv"):
    print(f"Loading {transaction_path} ...")
    tx = pd.read_csv(transaction_path)
    print(f"Loaded {len(tx)} transactions")

    if identity_path:
        print(f"Loading {identity_path} and merging on TransactionID ...")
        identity = pd.read_csv(identity_path)
        tx = tx.merge(identity[["TransactionID", "DeviceInfo", "DeviceType"]], on="TransactionID", how="left")
        print(f"Matched identity info for {tx['DeviceInfo'].notna().sum()} / {len(tx)} rows")
    else:
        tx["DeviceInfo"] = np.nan
        print("No identity file provided — device_id will fall back to card1 for all rows.")

    out = pd.DataFrame()
    out["order_id"] = tx["TransactionID"].astype(str)

    # --- customer_id proxy: the well-known IEEE-CIS "uid" trick ---
    out["customer_id"] = (
        tx["card1"].astype(str) + "_" +
        tx["card2"].fillna(-1).astype(str) + "_" +
        tx["card3"].fillna(-1).astype(str) + "_" +
        tx["card4"].fillna("na").astype(str) + "_" +
        tx["addr1"].fillna(-1).astype(str)
    )

    # --- device_id: real device info where available, else card1 fallback ---
    out["device_id"] = tx["DeviceInfo"].fillna("card_" + tx["card1"].astype(str))

    # --- address_id: combine addr1+addr2 ---
    out["address_id"] = tx["addr1"].fillna(-1).astype(str) + "_" + tx["addr2"].fillna(-1).astype(str)

    # --- phone: not available in this dataset ---
    out["phone"] = ""

    # --- promo_code: not available — see note at bottom of this file ---
    out["promo_code"] = ""

    out["order_amount"] = tx["TransactionAmt"]

    out["payment_method"] = tx["card6"].fillna(tx["ProductCD"]).fillna("unknown")

    # --- order_timestamp: TransactionDT is seconds offset, not a real date ---
    out["order_timestamp"] = tx["TransactionDT"].apply(
        lambda seconds: (REFERENCE_DATE + timedelta(seconds=int(seconds))).isoformat()
    )

    # --- account_created_at: approximate using D1 ("days since account first seen") ---
    d1_days = tx["D1"].fillna(0).clip(lower=0)
    out["account_created_at"] = [
        (REFERENCE_DATE + timedelta(seconds=int(dt)) - timedelta(days=float(d1)))
        .isoformat()
        for dt, d1 in zip(tx["TransactionDT"], d1_days)
    ]

    out["label_fraud"] = tx["isFraud"].astype(int)

    out.to_csv(output_path, index=False)
    print(f"\nWrote {output_path} with {len(out)} rows, {out['label_fraud'].sum()} fraud ({out['label_fraud'].mean()*100:.2f}%)")
    print("\nIMPORTANT: promo_code and phone are empty in this output — see the note")
    print("at the bottom of preprocess_ieee.py for how to handle the promo-abuse story.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_ieee.py train_transaction.csv [train_identity.csv] [output.csv]")
        print("  2 args: transaction.csv output.csv")
        print("  3 args: transaction.csv identity.csv output.csv")
        sys.exit(1)

    transaction_path = sys.argv[1]
    identity_path = None
    output_path = "ieee_mapped.csv"

    if len(sys.argv) == 3:
        # transaction.csv output.csv  (no identity file)
        output_path = sys.argv[2]
    elif len(sys.argv) >= 4:
        # transaction.csv identity.csv output.csv
        identity_path = sys.argv[2]
        output_path = sys.argv[3]

    main(transaction_path, identity_path, output_path)


# ============================================================================
# NOTE on promo_code: this dataset has no promo/coupon field at all, because
# it's a general payment-fraud dataset, not an e-commerce-promo dataset.
# You have two honest options:
#
# OPTION 1 (recommended, more defensible): Pivot your pitch narrative slightly
# — from "COD/RTO + promo abuse" to "card/payment fraud detection", since
# that's literally what this real dataset measures. Your rules.py /
# features.py graph-ring detection (shared card/address across many
# "customers") still tells the exact same fraud-ring story, just framed as
# payment fraud instead of promo abuse. This keeps 100% real labels.
#
# OPTION 2: Keep the promo-abuse story by synthetically injecting a
# promo_code column on top of this real data — e.g. for a random 5% of rows,
# assign a shared promo_code to transactions that already share a
# customer_id proxy or device_id, so the promo velocity / ring rules have
# something real to catch. This is common in security ML demos (real base
# data + a synthetically injected attack pattern) but be upfront about it
# in your pitch: "we validated ring-detection on injected promo-abuse
# patterns over real IEEE-CIS fraud data."
# ============================================================================
