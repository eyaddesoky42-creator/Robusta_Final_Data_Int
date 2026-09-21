"""
src/data_loader.py
Task 1 — Load Olist's 9 CSVs into a single SQLite database.
"""
import sqlite3
import pandas as pd
import os

DATA_DIR = "data"
DB_PATH = "olist.db"

CSV_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_data_to_sql(data_dir: str = DATA_DIR, db_path: str = DB_PATH) -> None:
    """Read all 9 Olist CSVs and load them into one SQLite database."""
    conn = sqlite3.connect(db_path)

    for table_name, filename in CSV_FILES.items():
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded {table_name}: {df.shape}")

    conn.close()
    print("\nDatabase created at:", os.path.abspath(db_path))


def verify_tables(db_path: str = DB_PATH) -> list:
    """Confirm all 9 tables exist in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()

    print("Tables in database:")
    for t in tables:
        print(" -", t[0])

    assert len(tables) == 9, "Expected 9 tables — check for errors above."
    print("\nall 9 tables present.")
    return [t[0] for t in tables]


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a fresh connection to the database — used by every other module."""
    return sqlite3.connect(db_path)


if __name__ == "__main__":
    load_data_to_sql()
    verify_tables()
