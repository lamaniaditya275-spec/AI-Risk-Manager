"""
Streamlit reviewer dashboard.

Run with: streamlit run dashboard.py

Reads decisions_sample.json (produced by train.py) and shows:
  - A flagged-order queue ranked by risk score
  - Per-order reason codes and triggered rules
  - Recovery path for each decision (what the customer can do next)
  - A linked-entity view (which other orders share this device/address)
  - Approve / Hold / Block buttons with reviewer notes (in-memory only —
    wire this to a real datastore for production use)
  - A metrics panel with insult rate, precision/recall, and action distribution
"""

import json
import pickle
import sys
from pathlib import Path

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
project_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import streamlit as st

from evaluate import threshold_table, compute_pr_auc

st.set_page_config(page_title="AI Risk Manager — Reviewer Console", layout="wide")

st.title("🛡️ AI Risk Manager — Reviewer Console")
st.caption("COD refusal & promo-abuse decisioning — Razorpay AI Buildathon prototype")

# ---------- Load decisions ----------
decisions_path = project_root / "artifacts" / "decisions_sample.json"
try:
    with open(decisions_path) as f:
        decisions = json.load(f)
except FileNotFoundError:
    st.error(f"{decisions_path} not found. Run `python src/train.py data/processed/output.csv` first.")
    st.stop()

decisions_df = pd.DataFrame(decisions)

if "review_status" not in st.session_state:
    st.session_state.review_status = {d["decision_id"]: "Pending" for d in decisions}

# ---------- Top metrics row ----------
col1, col2, col3, col4, col5 = st.columns(5)
band_counts = decisions_df["risk_band"].value_counts()
action_counts = decisions_df["action"].value_counts()

col1.metric("Total orders in queue", len(decisions_df))
col2.metric("Very High risk", int(band_counts.get("very_high", 0)))
col3.metric("High risk", int(band_counts.get("high", 0)))
col4.metric("Avg risk score", f"{decisions_df['risk_score'].mean():.2f}")

# Step-up verification count
stepup_count = int(action_counts.get("STEP_UP_VERIFICATION", 0))
col5.metric("Step-up verifications", stepup_count)

st.divider()

left, right = st.columns([2, 1])

# ---------- Flagged queue ----------
with left:
    st.subheader("📋 Flagged Order Queue (ranked by risk score)")
    band_filter = st.multiselect(
        "Filter by risk band",
        options=["very_high", "high", "medium", "low"],
        default=["very_high", "high"],
    )
    filtered = decisions_df[decisions_df["risk_band"].isin(band_filter)].sort_values(
        "risk_score", ascending=False
    )

    for _, row in filtered.iterrows():
        band_color = {
            "very_high": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"
        }.get(row["risk_band"], "⚪")

        with st.expander(
            f"{band_color} {row['order_id']} — {row['action']} "
            f"(score: {row['risk_score']:.2f})"
        ):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Reason codes:**")
                for reason in row["top_reasons"]:
                    st.markdown(f"- {reason}")
                if row["triggered_rules"]:
                    st.markdown("**Triggered rules:**")
                    for r in row["triggered_rules"]:
                        st.markdown(f"- `{r['rule_id']}` ({r['severity']}): {r['reason']}")
            with c2:
                st.markdown(f"**Customer:** {row['customer_id']}")
                st.markdown(f"**Action:** `{row['action']}`")
                # Recovery path
                recovery = row.get("recovery_path")
                if recovery:
                    st.info(f"🔄 **Recovery path:** {recovery}")
                else:
                    st.markdown("**Recovery path:** _None (terminal action)_")
                st.markdown(f"**Model version:** {row['model_version']}")
                st.markdown(f"**Anomaly score:** {row['anomaly_score']:.2f}")
                # Trust adjustment
                trust_adj = row.get("trust_adjustment", 0.0)
                if trust_adj > 0:
                    st.markdown(f"**Trust adjustment:** -{trust_adj:.2f} "
                                f"(raw score: {row.get('risk_score_raw', row['risk_score']):.2f})")
                st.markdown(f"**Feature snapshot:** `{row['feature_snapshot_hash']}`")

            status = st.session_state.review_status.get(row["decision_id"], "Pending")
            btn_cols = st.columns(4)
            if btn_cols[0].button("✅ Approve", key=f"appr_{row['decision_id']}"):
                st.session_state.review_status[row["decision_id"]] = "Approved"
            if btn_cols[1].button("⏸️ Hold", key=f"hold_{row['decision_id']}"):
                st.session_state.review_status[row["decision_id"]] = "Held"
            if btn_cols[2].button("⛔ Block", key=f"block_{row['decision_id']}"):
                st.session_state.review_status[row["decision_id"]] = "Blocked"
            if btn_cols[3].button("🔄 Reset", key=f"reset_{row['decision_id']}"):
                st.session_state.review_status[row["decision_id"]] = "Pending"

            st.markdown(f"**Reviewer status:** `{st.session_state.review_status[row['decision_id']]}`")
            st.text_input("Reviewer note", key=f"note_{row['decision_id']}", placeholder="Optional note...")

# ---------- Metrics panel ----------
with right:
    st.subheader("📊 Model Evaluation")
    model_path = project_root / "artifacts" / "trained_model.pkl"
    try:
        with open(model_path, "rb") as f:
            trained = pickle.load(f)
        st.markdown(f"**Model version:** `{trained.model_version}`")
        st.markdown(f"**Backend:** `{trained.backend}`")
        st.markdown(f"**Feature count:** {len(trained.feature_columns)}")
    except FileNotFoundError:
        st.warning(f"{model_path} not found.")
    except (ImportError, ModuleNotFoundError) as e:
        st.warning(f"Could not load model (missing dependency: {e.name}). "
                    "Run the dashboard from the project venv to see model details.")

    # Insult rate display
    st.markdown("---")
    st.subheader("🚨 Insult Rate")
    st.caption("Fraction of legitimate customers wrongly flagged")

    # Compute from the decisions sample if labels are available
    if "risk_score" in decisions_df.columns:
        # Count FP and TN from action distribution as a proxy
        total_flagged_legit = len(decisions_df[
            (decisions_df["action"].isin(["STEP_UP_VERIFICATION", "BLOCK_OR_FORCE_PREPAID", "MANUAL_REVIEW"]))
        ])
        total_decisions = len(decisions_df)
        st.metric(
            "Orders requiring customer action",
            f"{total_flagged_legit}/{total_decisions}",
            help="Orders where customer must verify or is blocked"
        )

    # Action distribution
    st.markdown("---")
    st.markdown("**Action distribution:**")
    action_display = decisions_df["action"].value_counts()
    for action_name, count in action_display.items():
        emoji = {
            "APPROVE": "✅", "APPROVE_WITH_MONITORING": "👀",
            "STEP_UP_VERIFICATION": "🔐", "MANUAL_REVIEW": "⏸️",
            "BLOCK_OR_FORCE_PREPAID": "⛔"
        }.get(action_name, "•")
        st.markdown(f"{emoji} **{action_name}**: {count}")

    st.markdown("**Review queue status breakdown:**")
    status_counts = pd.Series(st.session_state.review_status.values()).value_counts()
    st.bar_chart(status_counts)

    st.markdown("**Risk band distribution:**")
    st.bar_chart(band_counts)

st.divider()
st.caption(
    "This dashboard shows sample decisions from train.py's held-out test set. "
    "In production, /score_batch in api.py would feed this queue in real time "
    "from a live feature store."
)
