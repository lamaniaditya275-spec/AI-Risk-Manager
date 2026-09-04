"""
Generates a synthetic dataset (~2000 rows) to smoke-test the full pipeline.
Fraud rings are spread across the entire time window so train/val/test splits
each get enough positive examples for meaningful PR-AUC.

Usage: python make_smoke_test_data.py  -> writes smoke_test_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

rng = np.random.default_rng(42)
N_LEGIT_CUSTOMERS = 800
START = datetime(2025, 6, 1)
DAYS_SPAN = 60  # two months of data

rows = []
order_id = 0

# --- Legit customers ---
for cust_i in range(N_LEGIT_CUSTOMERS):
    cust_id = f"cust_legit_{cust_i}"
    device_id = f"dev_legit_{cust_i}"
    phone = f"phone_legit_{cust_i}"
    address_id = f"addr_legit_{cust_i}"

    # 5% household sharing (legit, not fraud)
    if cust_i > 0 and rng.random() < 0.05:
        device_id = f"dev_legit_{cust_i - 1}"
        address_id = f"addr_legit_{cust_i - 1}"

    account_created = START + timedelta(days=int(rng.integers(0, DAYS_SPAN - 5)))
    n_orders = int(rng.integers(1, 4))

    for order_idx in range(n_orders):
        order_time = account_created + timedelta(
            hours=int(rng.integers(6, 72 * (order_idx + 1)))
        )
        order_id += 1
        rows.append({
            "order_id": f"ord_{order_id}",
            "customer_id": cust_id,
            "device_id": device_id,
            "phone": phone,
            "address_id": address_id,
            "promo_code": "WELCOME10" if rng.random() < 0.12 else "",
            "order_amount": float(np.clip(rng.normal(900, 350), 100, 5000)),
            "payment_method": "COD" if rng.random() < 0.40 else "prepaid",
            "order_timestamp": order_time.isoformat(),
            "account_created_at": account_created.isoformat(),
            "label_fraud": 0,
        })

# --- Fraud rings: 15 rings x 8 members, spread evenly across time ---
N_RINGS = 15
MEMBERS_PER_RING = 8
for ring_i in range(N_RINGS):
    shared_device = f"dev_ring_{ring_i}"
    shared_address = f"addr_ring_{ring_i}"
    # Spread rings evenly across the full time window
    ring_day = int((ring_i / N_RINGS) * (DAYS_SPAN - 3))
    ring_start = START + timedelta(days=ring_day)

    for member_i in range(MEMBERS_PER_RING):
        cust_id = f"cust_ring_{ring_i}_{member_i}"
        phone = f"phone_ring_{ring_i}_{member_i}"
        account_created = ring_start + timedelta(hours=int(rng.integers(0, 36)))
        order_time = account_created + timedelta(hours=int(rng.integers(1, 8)))
        order_id += 1
        rows.append({
            "order_id": f"ord_{order_id}",
            "customer_id": cust_id,
            "device_id": shared_device,
            "phone": phone,
            "address_id": shared_address,
            "promo_code": f"PROMO{ring_i}",
            "order_amount": float(np.clip(rng.normal(1500, 400), 300, 4000)),
            "payment_method": "COD",
            "order_timestamp": order_time.isoformat(),
            "account_created_at": account_created.isoformat(),
            "label_fraud": 1,
        })

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "smoke_test_data.csv"

df = pd.DataFrame(rows).sort_values("order_timestamp")
df.to_csv(out_path, index=False)
n_fraud = int(df["label_fraud"].sum())
n_legit = int((df["label_fraud"] == 0).sum())
print(f"Wrote {out_path} with {len(df)} rows ({n_fraud} fraud, {n_legit} legit)")
