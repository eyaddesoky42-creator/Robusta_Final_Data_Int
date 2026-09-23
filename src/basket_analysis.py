#Comments generated via Ai (Claude)

"""
src/basket_analysis.py
Task 5 — Market basket analysis (category level): support, confidence, lift.

Why this module looks the way it does (measured on the real Olist data):
  * 9.9% of orders have more than one item.
  * Only ~0.75% of orders span 2+ *categories* (726 of ~97k orders).
  * So over ALL orders, no category pair exceeds ~0.07% support. mlxtend's
    default min_support (0.01) returns nothing, and lift comes out < 1 for
    almost every pair (categories look "mutually exclusive" because nearly
    every order is single-category).

So rules are mined on the multi-category baskets only. Read the resulting
lift as: "among customers who DO buy across categories, how much more often
do A and B appear together than chance?" — never as a store-wide claim.

For pairs, plain pandas counting gives exactly the same support/confidence/
lift that Apriori would; mlxtend only becomes necessary for 3+ item sets,
which this data cannot support anyway.
"""
##describtion generated via Ai
import itertools
import os
import sqlite3
from collections import Counter

import pandas as pd

BASKET_SQL = """
    SELECT oi.order_id, ct.product_category_name_english AS category
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN category_translation ct ON p.product_category_name = ct.product_category_name
"""


def load_category_baskets(conn: sqlite3.Connection) -> pd.Series:
    """Items with no category translation (~1.4%) are dropped."""
    df = pd.read_sql_query(BASKET_SQL, conn)
    return df.groupby("order_id")["category"].agg(frozenset)


def basket_coverage(conn: sqlite3.Connection, baskets: pd.Series) -> dict:
    """The numbers that justify the scoping decision — show these in the demo."""
    sizes = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM order_items GROUP BY order_id", conn
    )["n"]
    multi_cat = int((baskets.map(len) >= 2).sum())
    return {
        "orders_with_items": int(len(sizes)),
        "multi_item_orders": int((sizes > 1).sum()),
        "multi_item_pct": round(float((sizes > 1).mean()) * 100, 2),
        "categorized_orders": int(len(baskets)),
        "multi_category_orders": multi_cat,
        "multi_category_pct": round(multi_cat / len(baskets) * 100, 2),
    }


def mine_category_rules(baskets: pd.Series, min_count: int = 10,
                        multi_category_only: bool = True) -> pd.DataFrame:
    """Return A -> B rules (both directions) for every category pair seen in
    at least `min_count` baskets.

    Columns: antecedent, consequent, pair_count, support, confidence, lift.
    Sorted by lift, then pair_count.

    multi_category_only=True (default) mines only baskets with 2+ categories;
    False mines all orders (expect lift < 1 almost everywhere — see module doc).
    """
    if multi_category_only:
        baskets = baskets[baskets.map(len) >= 2]
    n = len(baskets)
    if n == 0:
        return pd.DataFrame(columns=["antecedent", "consequent", "pair_count",
                                     "support", "confidence", "lift"])

    item_counts, pair_counts = Counter(), Counter()
    for basket in baskets:
        item_counts.update(basket)
        pair_counts.update(itertools.combinations(sorted(basket), 2))

    rows = []
    for (a, b), c in pair_counts.items():
        if c < min_count:
            continue
        lift = (c / n) / ((item_counts[a] / n) * (item_counts[b] / n))
        for ant, con in ((a, b), (b, a)):
            rows.append({
                "antecedent": ant,
                "consequent": con,
                "pair_count": c,
                "support": c / n,
                "confidence": c / item_counts[ant],
                "lift": lift,
            })
    out = pd.DataFrame(rows, columns=["antecedent", "consequent", "pair_count",
                                      "support", "confidence", "lift"])
    return out.sort_values(["lift", "pair_count"], ascending=False).reset_index(drop=True)


def recommend_for_category(rules: pd.DataFrame, category: str, top_n: int = 3) -> pd.DataFrame:
    """'Frequently bought together' for one category: strongest rules where it is the antecedent."""
    return rules[rules["antecedent"] == category].head(top_n)


def save_rules_chart(rules: pd.DataFrame, out_dir: str = "reports", top_n: int = 8) -> str:
    """Bar chart of the strongest category pairs by lift, labelled with how many
    baskets back each one (small counts = weak evidence). Saved, never shown."""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "basket_rules_lift.png")
    pairs = rules[rules["antecedent"] < rules["consequent"]].head(top_n).iloc[::-1]
    labels = [f"{a} + {b}  (n={n})" for a, b, n in
              zip(pairs["antecedent"], pairs["consequent"], pairs["pair_count"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, pairs["lift"], color="#4C72B0")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("lift (1.0 = no better than chance)")
    ax.set_title("Category pairs bought together — multi-category baskets only")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    from src.data_loader import get_connection

    conn = get_connection()
    baskets = load_category_baskets(conn)
    print(basket_coverage(conn, baskets))
    rules = mine_category_rules(baskets, min_count=10)
    print(rules.round(3).head(15).to_string(index=False))
