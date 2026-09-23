"""
src/data_prep.py
Shared "shopping trip" logic for Tasks 5-8.

Problem: consecutive orders by the same customer placed within a short
window (e.g. 45 minutes) are very likely one checkout split into multiple
order rows, not two separate purchase decisions. Counting them as separate
"repeat purchases" inflates frequency, repeat-rate, and any downstream
metric built on order counts.

Fix: merge orders by the same customer_unique_id into a "trip" whenever the
gap since their previous order is <= merge_gap_hours. Every task that counts
orders (Task 5 baskets, Task 6 histories, Task 7 frequency, Task 8's
back-test) should count trips instead of raw order rows.

Only ORDER TIMING is touched here — monetary totals and "last purchase"
date are unaffected by how orders are grouped, since the same items and
timestamps exist either way; only the *count* of distinct purchase events
changes.
"""
import pandas as pd
import sqlite3

ORDER_TIMESTAMPS_SQL = """
    SELECT o.order_id, c.customer_unique_id, o.order_purchase_timestamp AS ts
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
"""


def get_order_timestamps(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per delivered order: order_id, customer_unique_id, ts."""
    df = pd.read_sql_query(ORDER_TIMESTAMPS_SQL, conn)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values(["customer_unique_id", "ts", "order_id"]).reset_index(drop=True)


def assign_trip_ids(order_timestamps: pd.DataFrame, gap_hours: float = 1.0) -> pd.DataFrame:
    """
    Adds a 'trip_id' column. A new trip starts whenever the gap since the
    same customer's previous order exceeds gap_hours (or it's their first
    order). Orders within the gap get the same trip_id.
    """
    df = order_timestamps.sort_values(["customer_unique_id", "ts"]).copy()
    gap = df.groupby("customer_unique_id")["ts"].diff()
    new_trip = gap.isna() | (gap > pd.Timedelta(hours=gap_hours))
    trip_num = new_trip.groupby(df["customer_unique_id"]).cumsum()
    df["trip_id"] = df["customer_unique_id"] + "_trip" + trip_num.astype(str)
    return df[["order_id", "customer_unique_id", "ts", "trip_id"]]


def get_order_to_trip_map(conn: sqlite3.Connection, gap_hours: float = 1.0) -> dict:
    """order_id -> trip_id, for delivered orders only. Orders not in this map
    (e.g. non-delivered) should be left keyed by their own order_id by callers."""
    ts = get_order_timestamps(conn)
    trips = assign_trip_ids(ts, gap_hours=gap_hours)
    return dict(zip(trips["order_id"], trips["trip_id"]))


def trip_summary(conn: sqlite3.Connection, gap_hours: float = 1.0) -> dict:
    """Before/after counts — use this in the demo to show the effect of merging."""
    ts = get_order_timestamps(conn)
    trips = assign_trip_ids(ts, gap_hours=gap_hours)

    raw_repeat = (ts.groupby("customer_unique_id")["order_id"].nunique() > 1).sum()
    trip_repeat = (trips.groupby("customer_unique_id")["trip_id"].nunique() > 1).sum()

    return {
        "gap_hours": gap_hours,
        "raw_orders": ts["order_id"].nunique(),
        "merged_trips": trips["trip_id"].nunique(),
        "orders_merged_away": ts["order_id"].nunique() - trips["trip_id"].nunique(),
        "repeat_customers_raw": int(raw_repeat),
        "repeat_customers_trips": int(trip_repeat),
    }


if __name__ == "__main__":
    from src.data_loader import get_connection

    conn = get_connection()
    print(trip_summary(conn, gap_hours=1.0))
