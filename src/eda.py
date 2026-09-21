"""
src/eda.py
Task 3 — Exploratory Data Analysis: basket composition, repeat-purchase
behavior, reorder timing, customer value.
"""
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os


def compute_basket_sizes(conn: sqlite3.Connection) -> pd.DataFrame:
    """3.1 — Basket composition: items per order."""
    return pd.read_sql_query("""
        SELECT order_id, COUNT(order_item_id) AS num_items
        FROM order_items
        GROUP BY order_id
    """, conn)


def compute_category_volume(conn: sqlite3.Connection) -> pd.DataFrame:
    """3.2 — Top categories by volume."""
    return pd.read_sql_query("""
        SELECT ct.product_category_name_english AS category, COUNT(*) AS times_purchased
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        GROUP BY category
        ORDER BY times_purchased DESC
    """, conn)


def compute_repeat_rate(conn: sqlite3.Connection):
    """3.3 — Repeat-purchase behavior. Returns (rate, full breakdown)."""
    repeat_customers = pd.read_sql_query("""
        SELECT customer_unique_id, COUNT(DISTINCT o.order_id) AS num_orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
        GROUP BY customer_unique_id
    """, conn)
    repeat_rate = (repeat_customers["num_orders"] > 1).mean()
    return repeat_rate, repeat_customers


def compute_reorder_gaps(conn: sqlite3.Connection) -> pd.Series:
    """3.4 — Reorder timing: days between orders, for repeat customers."""
    order_dates = pd.read_sql_query("""
        SELECT c.customer_unique_id, o.order_purchase_timestamp
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
        ORDER BY c.customer_unique_id, o.order_purchase_timestamp
    """, conn)
    order_dates["order_purchase_timestamp"] = pd.to_datetime(order_dates["order_purchase_timestamp"])
    order_dates["gap_days"] = order_dates.groupby("customer_unique_id")["order_purchase_timestamp"].diff().dt.days
    return order_dates["gap_days"].dropna()


def compute_customer_value(conn: sqlite3.Connection) -> pd.DataFrame:
    """3.5 — Customer lifetime value snapshot."""
    return pd.read_sql_query("""
        SELECT c.customer_unique_id,
               COUNT(DISTINCT o.order_id) AS num_orders,
               ROUND(SUM(oi.price), 2) AS total_spent
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY c.customer_unique_id
        ORDER BY total_spent DESC
    """, conn)


def compute_revenue_concentration(customer_value: pd.DataFrame) -> float:
    """3.6 — Revenue concentration check (80/20 pattern)."""
    sorted_df = customer_value.sort_values("total_spent", ascending=False).reset_index(drop=True)
    sorted_df["cumulative_pct"] = sorted_df["total_spent"].cumsum() / sorted_df["total_spent"].sum()
    top_20pct_count = int(len(sorted_df) * 0.2)
    return sorted_df.iloc[top_20pct_count - 1]["cumulative_pct"]


def save_basket_size_chart(basket_sizes: pd.DataFrame, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "basket_size_distribution.png")
    plt.figure(figsize=(8, 5))
    basket_sizes["num_items"].value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Basket Size (items per order)")
    plt.xlabel("Number of items")
    plt.ylabel("Number of orders")
    plt.savefig(path)
    plt.show()
    return path


def save_category_volume_chart(category_volume: pd.DataFrame, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "top_categories.png")
    plt.figure(figsize=(10, 6))
    category_volume.head(10).plot(kind="barh", x="category", y="times_purchased", legend=False)
    plt.title("Top 10 Categories by Volume")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    return path


def save_reorder_gap_chart(gaps: pd.Series, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "reorder_gap_histogram.png")
    plt.figure(figsize=(8, 5))
    gaps.hist(bins=30)
    plt.title("Days Between Repeat Orders")
    plt.xlabel("Days")
    plt.ylabel("Frequency")
    plt.savefig(path)
    plt.show()
    return path
