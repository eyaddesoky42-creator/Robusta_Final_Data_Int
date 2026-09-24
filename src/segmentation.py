"""
src/segmentation.py
Task 7 — Customer segmentation with RFM.

RFM = three numbers per customer, computed from delivered orders:
  Recency   : days since the customer's last order (smaller = better)
  Frequency : how many orders (or merged shopping trips) they placed
  Monetary  : how much they spent (sum of item prices, same definition as eda.py)

Each customer gets a 1-5 score for R and M (5 = best, quintiles). Frequency is
NOT quintiled: ~97% of Olist customers ordered exactly once, so quintiles would
be meaningless — it is capped at 3 (1, 2, 3+ orders) instead.

merge_gap_hours: if set, orders by the same customer within this many hours
are merged into one shopping trip before frequency is counted (see
src/data_prep.py) — orders placed within a checkout split into multiple rows
should not be counted as separate repeat purchases. Recency and monetary are
unaffected by merging: they depend on total spend and last-purchase date,
not on how orders are grouped.

Two views of the same RFM table:
  * rule-based segments  -> named, explainable, used by Task 8 (promos)
  * KMeans clusters      -> a data-driven cross-check that the rules aren't arbitrary

Depends on: Task 1 (olist.db), the customer_unique_id fix from Task 2, and
            src/data_prep.py (trip merging).
Feeds:      Task 8 (each promo is attached to a segment).
"""
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_SQL = """
    SELECT c.customer_unique_id,
           MAX(o.order_purchase_timestamp)  AS last_purchase,
           COUNT(DISTINCT o.order_id)       AS frequency,
           ROUND(SUM(oi.price), 2)          AS monetary
    FROM orders o
    JOIN customers c    ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
      {date_filter}
    GROUP BY c.customer_unique_id
"""

# Display order = roughly "best customers first"
SEGMENT_ORDER = ["Champions", "Loyal repeat", "High-value new",
                 "At-risk high-value", "Recent one-timers", "Lapsed one-timers"]


# --------------------------------------------------------------------------
# 1. Compute RFM
# --------------------------------------------------------------------------
def compute_rfm(conn: sqlite3.Connection, snapshot_date=None, merge_gap_hours: float = None) -> pd.DataFrame:
    """RFM table, one row per customer_unique_id.

    snapshot_date=None -> "today" is the day after the last order in the data.
    snapshot_date='2018-05-31' -> only orders BEFORE that date are used
    (needed by Task 8 to back-test what segments did in the following 90 days).

    merge_gap_hours=None -> frequency counts raw order rows (old behavior).
    merge_gap_hours=1.0  -> frequency counts merged shopping trips instead;
    recency and monetary are untouched by this, since they depend on total
    spend and the last-purchase timestamp, not on how orders are grouped.
    """
    date_filter = ""
    params = ()
    if snapshot_date is not None:
        date_filter = "AND o.order_purchase_timestamp < ?"
        params = (str(snapshot_date),)
    df = pd.read_sql_query(RFM_SQL.format(date_filter=date_filter), conn, params=params)
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])

    if snapshot_date is None:
        snapshot = df["last_purchase"].max().normalize() + pd.Timedelta(days=1)
    else:
        snapshot = pd.Timestamp(snapshot_date)
    df["recency_days"] = (snapshot - df["last_purchase"]).dt.days

    if merge_gap_hours is not None:
        from src.data_prep import get_order_timestamps, assign_trip_ids
        ts = get_order_timestamps(conn)
        if snapshot_date is not None:
            ts = ts[ts["ts"] < pd.Timestamp(snapshot_date)]
        trips = assign_trip_ids(ts, gap_hours=merge_gap_hours)
        trip_counts = trips.groupby("customer_unique_id")["trip_id"].nunique()
        df["frequency"] = df["customer_unique_id"].map(trip_counts).fillna(df["frequency"]).astype(int)

    df["avg_order_value"] = (df["monetary"] / df["frequency"]).round(2)
    df.attrs["snapshot"] = snapshot
    return df


# --------------------------------------------------------------------------
# 2. Score + rule-based segments
# --------------------------------------------------------------------------
def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add r_score, f_score, m_score. rank(method='first') breaks ties so qcut never
    fails on duplicate bin edges (many customers share the same price)."""
    out = rfm.copy()
    out["r_score"] = pd.qcut(out["recency_days"].rank(method="first"), 5,
                             labels=[5, 4, 3, 2, 1]).astype(int)      # recent = 5
    out["m_score"] = pd.qcut(out["monetary"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)      # big spender = 5
    out["f_score"] = out["frequency"].clip(upper=3).astype(int)       # 1, 2, 3+
    return out


def assign_segment(r: int, f: int, m: int) -> str:
    """Business rules — first match wins."""
    if f >= 2 and r >= 4 and m >= 4:
        return "Champions"
    if f >= 2:
        return "Loyal repeat"
    if m >= 4 and r >= 3:
        return "High-value new"
    if m >= 4:
        return "At-risk high-value"
    if r >= 3:
        return "Recent one-timers"
    return "Lapsed one-timers"


def add_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    scored = score_rfm(rfm)
    scored["segment"] = [assign_segment(r, f, m) for r, f, m in
                         zip(scored["r_score"], scored["f_score"], scored["m_score"])]
    scored["segment"] = pd.Categorical(scored["segment"], categories=SEGMENT_ORDER, ordered=True)
    return scored


def segment_profile(segmented: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: size, share of customers, averages, share of revenue."""
    g = segmented.groupby("segment", observed=True)
    prof = g.agg(customers=("customer_unique_id", "count"),
                 avg_recency_days=("recency_days", "mean"),
                 avg_orders=("frequency", "mean"),
                 avg_spend=("monetary", "mean"),
                 avg_order_value=("avg_order_value", "mean"),
                 total_revenue=("monetary", "sum"))
    prof["pct_customers"] = prof["customers"] / prof["customers"].sum() * 100
    prof["pct_revenue"] = prof["total_revenue"] / prof["total_revenue"].sum() * 100
    return prof.round(1)


# --------------------------------------------------------------------------
# 3. KMeans cross-check
# --------------------------------------------------------------------------
def kmeans_segments(rfm: pd.DataFrame, k: int = None, k_range=range(3, 7),
                    seed: int = 42, sample: int = 10000):
    """Cluster on log-scaled, standardised R/F/M.
    If k is None, choose it by silhouette score (computed on a sample for speed).
    Returns (rfm with 'cluster' column, chosen k, {k: silhouette})."""
    X = np.log1p(rfm[["recency_days", "frequency", "monetary"]].to_numpy())
    X = StandardScaler().fit_transform(X)
    sil = {}
    if k is None:
        for kk in k_range:
            labels = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(X)
            sil[kk] = float(silhouette_score(X, labels, sample_size=min(sample, len(X)),
                                             random_state=seed))
        k = max(sil, key=sil.get)
    out = rfm.copy()
    out["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    return out, k, sil


def cluster_vs_segment(clustered_with_segments: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab: how each KMeans cluster is composed of rule-based segments (row %)."""
    ct = pd.crosstab(clustered_with_segments["cluster"], clustered_with_segments["segment"],
                     normalize="index") * 100
    return ct.round(1)


# --------------------------------------------------------------------------
# 4. Chart (saved, never shown — so it also works inside Streamlit later)
# --------------------------------------------------------------------------
def save_segment_chart(profile: pd.DataFrame, out_dir: str = "reports"):
    import matplotlib
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rfm_segments.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    profile["pct_customers"].plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("% of customers")
    profile["pct_revenue"].plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_title("% of revenue")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    from src.data_loader import get_connection

    conn = get_connection()
    seg = add_segments(compute_rfm(conn, merge_gap_hours=1.0))
    print(segment_profile(seg).to_string())
