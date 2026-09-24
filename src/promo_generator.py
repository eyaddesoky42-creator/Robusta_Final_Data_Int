"""
src/promo_generator.py
Task 8 — Promo-code generator with margin logic and a human-approval gate.

The idea in one paragraph
-------------------------
For each RFM segment (Task 7) we propose ONE offer (a % discount on the next
order) and compute what it would do to margin. A discount has two sides:
  gain : customers who would NOT have ordered but do because of the code
         -> extra margin  = extra_orders * AOV * (margin_rate - discount)
  cost : customers who WOULD have ordered anyway and now use the code
         -> lost margin   = baseline_orders * AOV * discount
So   net per targeted customer = dp * AOV * (m - d)  -  p0 * AOV * d
     break-even uplift dp*     = p0 * d / (m - d)
where p0 = baseline chance the customer orders in the next 90 days, MEASURED
by back-testing the segments on history (no guessing); m = margin rate and
dp = extra order probability caused by the promo are ASSUMPTIONS (Olist has no
cost data and no promo history) — so results are shown as low/base/high
scenarios plus the break-even uplift, which is what an A/B test must beat.

merge_gap_hours (default 1.0): orders by the same customer within this many
hours are treated as one shopping trip, not separate repeat purchases, when
computing segments and the back-tested baseline (see src/data_prep.py). This
is now a tracked assumption on Assumptions, since it changes segment sizes
and p0 the same way margin_rate and uplift do.

Nothing is issued automatically: build_offers() only PROPOSES (status
'pending_approval'); export_customer_codes() refuses to produce codes for any
offer a human has not approved with approve_offers().

Depends on: Task 7 (segments), Task 6 (suggested category per customer),
            Task 2/3 (orders, customers, order_items in olist.db),
            src/data_prep.py (trip merging).
Feeds:      Task 9 (the offers table is what the LLM narrates).
"""
import hashlib
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from src.segmentation import SEGMENT_ORDER, add_segments, compute_rfm


# --------------------------------------------------------------------------
# Assumptions — everything NOT measured from data lives here, in one place
# --------------------------------------------------------------------------
@dataclass
class Assumptions:
    margin_rate: float = 0.20        # contribution margin on item price (ASSUMED — no cost data in Olist)
    margin_floor: float = 0.05       # guardrail: after the discount, keep >= 5% margin on the order
    window_days: int = 90            # offer horizon; also the back-test window for p0
    valid_days: int = 30             # how long a code is valid once issued
    merge_gap_hours: float = 1.0     # orders within this gap = one trip, not separate repeat purchases
    # extra probability (absolute) that a targeted customer orders because of the promo (ASSUMED)
    uplift: dict = field(default_factory=lambda: {"low": 0.005, "base": 0.015, "high": 0.030})
    # candidate discount per segment (proposal; the maths below decides if it is worth sending)
    discounts: dict = field(default_factory=lambda: {
        "Champions": 0.10, "Loyal repeat": 0.10, "High-value new": 0.10,
        "At-risk high-value": 0.15, "Recent one-timers": 0.10, "Lapsed one-timers": 0.15})


CODE_PREFIX = {"Champions": "VIP", "Loyal repeat": "LOYAL", "High-value new": "SECOND",
               "At-risk high-value": "COMEBACK", "Recent one-timers": "NEXT",
               "Lapsed one-timers": "WINBACK"}


# --------------------------------------------------------------------------
# 1. MEASURED: baseline reorder probability per segment (back-test)
# --------------------------------------------------------------------------
def baseline_reorder_rates(conn: sqlite3.Connection, window_days: int = 90,
                           merge_gap_hours: float = None) -> pd.DataFrame:
    """Freeze history `window_days` before the end of the data, segment customers as
    they looked THEN, and count who actually ordered in the window that followed.
    p0 = share of the segment that ordered = what happens with NO promo.

    merge_gap_hours is forwarded to compute_rfm so segments here are computed
    the same way as everywhere else in the pipeline."""
    last = pd.read_sql_query(
        "SELECT MAX(order_purchase_timestamp) AS m FROM orders WHERE order_status='delivered'",
        conn)["m"].iloc[0]
    end = pd.Timestamp(last).normalize() + pd.Timedelta(days=1)
    snapshot = end - pd.Timedelta(days=window_days)

    seg = add_segments(compute_rfm(conn, snapshot_date=snapshot.strftime("%Y-%m-%d"),
                                    merge_gap_hours=merge_gap_hours))
    future = pd.read_sql_query("""
        SELECT DISTINCT c.customer_unique_id
        FROM orders o JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
          AND o.order_purchase_timestamp >= ? AND o.order_purchase_timestamp < ?
    """, conn, params=(snapshot.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    seg["ordered_in_window"] = seg["customer_unique_id"].isin(future["customer_unique_id"])

    out = seg.groupby("segment", observed=True).agg(
        customers_at_snapshot=("customer_unique_id", "count"),
        ordered_in_window=("ordered_in_window", "sum"))
    out["p0"] = out["ordered_in_window"] / out["customers_at_snapshot"]
    out.attrs["snapshot"], out.attrs["window_end"] = snapshot, end
    return out


# --------------------------------------------------------------------------
# 2. Margin maths (pure functions — easy to unit-test and to explain)
# --------------------------------------------------------------------------
def net_per_customer(p0: float, aov: float, discount: float, margin: float, uplift: float) -> float:
    """Expected margin change per targeted customer over the window."""
    gain = uplift * aov * (margin - discount)
    cost = p0 * aov * discount
    return gain - cost


def break_even_uplift(p0: float, discount: float, margin: float) -> float:
    """Extra order probability the promo must create just to pay for itself.
    Infinite if the discount eats the whole margin."""
    if margin - discount <= 0:
        return float("inf")
    return p0 * discount / (margin - discount)


# --------------------------------------------------------------------------
# 3. Build proposals
# --------------------------------------------------------------------------
def build_offers(conn: sqlite3.Connection, a: Assumptions = None) -> pd.DataFrame:
    """One proposed offer per segment, with expected margin impact. status is always
    'pending_approval' (or 'not_recommended') — a human decides."""
    a = a or Assumptions()
    seg = add_segments(compute_rfm(conn, merge_gap_hours=a.merge_gap_hours))
    size = seg.groupby("segment", observed=True).agg(
        customers=("customer_unique_id", "count"), avg_order_value=("avg_order_value", "mean"))
    p0 = baseline_reorder_rates(conn, a.window_days, merge_gap_hours=a.merge_gap_hours)["p0"]

    max_disc = round(a.margin_rate - a.margin_floor, 4)
    rows = []
    for s in SEGMENT_ORDER:
        n, aov, base = int(size.loc[s, "customers"]), float(size.loc[s, "avg_order_value"]), float(p0.loc[s])
        d = min(a.discounts[s], max_disc)
        net = {k: net_per_customer(base, aov, d, a.margin_rate, u) for k, u in a.uplift.items()}
        be = break_even_uplift(base, d, a.margin_rate)

        if net["low"] > 0:
            rec, status = "send", "pending_approval"
        elif net["base"] > 0:
            rec, status = "send as A/B test", "pending_approval"
        else:
            rec, status = "do not discount", "not_recommended"

        rows.append({
            "segment": s, "customers": n, "avg_order_value": round(aov, 2),
            "baseline_p0": base, "discount": d, "break_even_uplift": be,
            "net_per_customer_low": net["low"], "net_per_customer_base": net["base"],
            "net_per_customer_high": net["high"],
            "total_net_low": net["low"] * n, "total_net_base": net["base"] * n,
            "total_net_high": net["high"] * n,
            "recommendation": rec, "status": status,
            "code_prefix": f"{CODE_PREFIX[s]}{int(round(d * 100))}",
        })
    offers = pd.DataFrame(rows)
    offers.attrs["assumptions"] = a
    return offers


def margin_sensitivity(conn: sqlite3.Connection, margins=(0.20, 0.25, 0.30, 0.40)) -> pd.DataFrame:
    """Total base-scenario net impact per segment for different margin assumptions —
    shows how much the conclusion depends on the one number we had to assume."""
    cols = {}
    for m in margins:
        o = build_offers(conn, Assumptions(margin_rate=m))
        cols[f"margin {int(m * 100)}%"] = o.set_index("segment")["total_net_base"].round(0)
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------
# 4. Human approval gate + code export
# --------------------------------------------------------------------------
def approve_offers(offers: pd.DataFrame, segments: list, approver: str, force: bool = False) -> pd.DataFrame:
    """Mark offers as approved by a named person. Refuses 'not_recommended' offers
    unless force=True (an explicit, visible override)."""
    out = offers.copy()
    for s in segments:
        i = out.index[out["segment"] == s]
        if len(i) == 0:
            raise ValueError(f"unknown segment: {s}")
        if out.loc[i[0], "status"] == "not_recommended" and not force:
            raise ValueError(f"'{s}' is not recommended (net impact <= 0). Use force=True to override.")
        out.loc[i[0], "status"] = "approved"
        out.loc[i[0], "approved_by"] = approver
        out.loc[i[0], "approved_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return out


def make_code(prefix: str, customer_unique_id: str, salt: str = "robusta-p9", n: int = 8) -> str:
    """Deterministic, per-customer, hard to guess: PREFIX + n hex chars (8 -> 4.3 billion combos)."""
    h = hashlib.sha1(f"{salt}|{prefix}|{customer_unique_id}".encode()).hexdigest()[:n].upper()
    return f"{prefix}-{h}"


def _unique_codes(prefix: str, customer_ids, taken: set) -> list:
    """make_code for every customer, lengthening the hash on the (rare) collision so a
    code can never be shared by two customers, also across segments."""
    codes = []
    for cid in customer_ids:
        n = 8
        code = make_code(prefix, cid, n=n)
        while code in taken:
            n += 2
            code = make_code(prefix, cid, n=n)
        taken.add(code)
        codes.append(code)
    return codes


def export_customer_codes(conn: sqlite3.Connection, offers: pd.DataFrame,
                          valid_days: int = 30, merge_gap_hours: float = 1.0) -> pd.DataFrame:
    """One row per customer in every APPROVED offer: unique code, discount, and the
    category to feature (Task 6 next-order recommender, top-1). Pending offers produce nothing.

    merge_gap_hours is forwarded to both the segment lookup and the order
    history used by the recommender, so this stays consistent with however
    build_offers() computed the segments being exported."""
    from src.recommender import NextOrderRecommender, customer_histories, load_orders

    approved = offers[offers["status"] == "approved"]
    if approved.empty:
        return pd.DataFrame(columns=["customer_unique_id", "segment", "code", "discount",
                                     "suggested_category", "valid_until"])
    seg = add_segments(compute_rfm(conn, merge_gap_hours=merge_gap_hours))
    orders = load_orders(conn, merge_gap_hours=merge_gap_hours)
    hist_all = customer_histories(orders, min_orders=1)
    model = NextOrderRecommender().fit([h for h in hist_all.values() if len(h) >= 2])
    valid_until = (pd.Timestamp.now().normalize() + pd.Timedelta(days=valid_days)).strftime("%Y-%m-%d")

    parts, taken = [], set()
    for _, o in approved.iterrows():
        cust = seg.loc[seg["segment"] == o["segment"], ["customer_unique_id"]].copy()
        cust["segment"] = o["segment"]
        cust["code"] = _unique_codes(o["code_prefix"], cust["customer_unique_id"], taken)
        cust["discount"] = o["discount"]
        cust["suggested_category"] = [
            model.recommend(hist_all.get(c, []), k=1)[0]   # no history -> popularity
            for c in cust["customer_unique_id"]]
        cust["valid_until"] = valid_until
        parts.append(cust)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    from src.data_loader import get_connection

    conn = get_connection()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(build_offers(conn).round(4).to_string(index=False))
