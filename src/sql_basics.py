"""
src/sql_basics.py
Task 2 — Core SQL query functions, reused throughout the project.
"""
import pandas as pd
import sqlite3


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Generic helper — run any SQL string, get a DataFrame back."""
    return pd.read_sql_query(sql, conn)


def get_delivered_orders(conn: sqlite3.Connection) -> pd.DataFrame:
    """Basic SELECT + WHERE (filtering)."""
    return run_query(conn, """
        SELECT order_id, order_status, order_purchase_timestamp
        FROM orders
        WHERE order_status = 'delivered'
        LIMIT 10;
    """)


def get_orders_with_items(conn: sqlite3.Connection) -> pd.DataFrame:
    """JOIN for connecting tables."""
    return run_query(conn, """
        SELECT o.order_id, o.order_status, oi.product_id, oi.price
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        LIMIT 10;
    """)


def get_orders_per_customer(conn):
    return run_query(conn, """
        SELECT c.customer_unique_id, COUNT(o.order_id) AS total_orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.customer_unique_id
        ORDER BY total_orders DESC
        LIMIT 10;
    """)


def get_category_revenue(conn: sqlite3.Connection) -> pd.DataFrame:
    """Multi-table JOIN + aggregate for real analytical questions."""
    return run_query(conn, """
        SELECT ct.product_category_name_english AS category,
               COUNT(oi.order_item_id) AS items_sold,
               ROUND(SUM(oi.price), 2) AS total_revenue
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        GROUP BY category
        ORDER BY total_revenue DESC
        LIMIT 10;
    """)


def get_repeat_customers(conn: sqlite3.Connection) -> pd.DataFrame:
    """Subquery / HAVING — finding repeat customers.
    Olist has no native reorder flag, so this derives repeat-purchase
    signal via customer_unique_id."""
    return run_query(conn, """
        SELECT customer_unique_id, COUNT(DISTINCT order_id) AS num_orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY customer_unique_id
        HAVING num_orders > 1
        ORDER BY num_orders DESC
        LIMIT 10;
    """)


def get_customer_activity_span(conn: sqlite3.Connection) -> pd.DataFrame:
    """Date functions — julianday() enables date math in SQLite."""
    return run_query(conn, """
        SELECT customer_unique_id,
               MIN(order_purchase_timestamp) AS first_order,
               MAX(order_purchase_timestamp) AS last_order,
               julianday(MAX(order_purchase_timestamp)) - julianday(MIN(order_purchase_timestamp)) AS days_active
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY customer_unique_id
        ORDER BY days_active DESC
        LIMIT 10;
    """)
