# Expected Data Schema

This system is schema-driven. Whatever dataset you bring in (from ChatGPT, Kaggle,
or elsewhere), map its columns to this schema. The pipeline only ever needs one
core table — `orders` — with the rest as optional lookup context. Missing optional
columns degrade gracefully (fewer features get built), so you don't need a perfect
match to get started.

## Core table: `orders` (one row per order/transaction event)

| column              | type      | required | notes |
|---------------------|-----------|----------|-------|
| order_id            | string    | yes      | unique per order |
| customer_id         | string    | yes      | |
| device_id           | string    | yes      | device fingerprint / session ID |
| phone               | string    | yes      | can be hashed/tokenized |
| address_id          | string    | yes      | shipping address ID (not raw text) |
| promo_code          | string    | no       | null/empty if none used |
| order_amount        | float     | yes      | |
| payment_method      | string    | yes      | e.g. "COD", "prepaid" |
| order_timestamp     | datetime  | yes      | ISO 8601, used for time-based split & rolling windows |
| account_created_at  | datetime  | yes      | used for "account age" feature |
| label_fraud         | int (0/1) | yes*     | 1 = confirmed COD refusal / promo abuse / RTO fraud. *Required for training, not for scoring new orders. |

## Optional lookup tables (improve feature quality if present)

### `customers`
| column | type | notes |
|---|---|---|
| customer_id | string | |
| signup_country | string | |
| kyc_status | string | |

### `devices`
| column | type | notes |
|---|---|---|
| device_id | string | |
| device_type | string | mobile/desktop/emulator flag if known |

### `outcomes` (post-hoc — NEVER join into training features directly, only into labels)
| column | type | notes |
|---|---|---|
| order_id | string | |
| outcome_type | string | "delivered", "rto", "refused", "promo_abuse_confirmed", "chargeback" |
| outcome_timestamp | datetime | must be AFTER order_timestamp — this is what makes it a label, not a feature |

## Column mapping

If your dataset uses different column names, edit `config.py` → `COLUMN_MAP` to
point our internal names at your actual column names. Nothing else needs to change.

## Critical rule: leakage safety

Any column that is only known AFTER the order is placed (delivery outcome, refund
status, chargeback status, manual review verdict) must live in `outcomes`, never in
`orders`, and must never be used as a model input feature — only as the training
label. The pipeline enforces this by only ever reading `label_fraud` from outcomes
data during training, and refusing to accept outcome-derived columns as features.
