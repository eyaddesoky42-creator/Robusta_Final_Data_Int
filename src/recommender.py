"""
src/recommender.py
Task 6 — Recommendation engine + honest evaluation against a popularity baseline.

Two recommenders, both at CATEGORY level (product level is far too sparse:
~32k products, most bought once):

  1. "Frequently bought together"  -> given the categories already in a basket,
     which category should be added?      (BoughtTogetherRecommender)
  2. "Next order"                  -> given a customer's past orders, which
     categories will their next order contain?   (NextOrderRecommender)

Every model is compared with baselines under cross-validation, and the result
carries a bootstrap confidence interval so we can say "not distinguishable from
popularity" when that is the truth.

Depends on: Task 5 (same pair-counting idea, but here counts are learned on a
train split only so the evaluation is leak-free).
Feeds:      Task 8 (promo generator uses NextOrderRecommender for the
            suggested category per customer).
"""
import itertools
import sqlite3
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ORDER_SQL = """
    SELECT c.customer_unique_id,
           o.order_id,
           o.order_purchase_timestamp AS ts,
           ct.product_category_name_english AS category
    FROM orders o
    JOIN customers c            ON o.customer_id = c.customer_id
    JOIN order_items oi         ON o.order_id = oi.order_id
    JOIN products p             ON oi.product_id = p.product_id
    JOIN category_translation ct ON p.product_category_name = ct.product_category_name
    WHERE o.order_status = 'delivered'
"""


def load_orders(conn: sqlite3.Connection, merge_gap_hours: float = None) -> pd.DataFrame:
    """One row per delivered order (or trip, if merge_gap_hours is set):
    customer_unique_id, order_id, ts, categories (frozenset).
    Sorted by customer then time."""
    df = pd.read_sql_query(ORDER_SQL, conn)
    orders = (df.groupby(["customer_unique_id", "order_id", "ts"])["category"]
                .agg(frozenset).reset_index().rename(columns={"category": "categories"}))
    orders["ts"] = pd.to_datetime(orders["ts"])
    orders = orders.sort_values(["customer_unique_id", "ts", "order_id"]).reset_index(drop=True)

    if merge_gap_hours is not None:
        from src.data_prep import get_order_to_trip_map
        trip_map = get_order_to_trip_map(conn, gap_hours=merge_gap_hours)
        orders["order_id"] = orders["order_id"].map(trip_map).fillna(orders["order_id"])
        orders = (orders.groupby(["customer_unique_id", "order_id"])
                        .agg(ts=("ts", "min"),
                             categories=("categories", lambda s: frozenset().union(*s)))
                        .reset_index()
                        .sort_values(["customer_unique_id", "ts", "order_id"])
                        .reset_index(drop=True))
    return orders

def customer_histories(orders: pd.DataFrame, min_orders: int = 1) -> dict:
    """customer_unique_id -> chronological list of category sets."""
    hist = orders.groupby("customer_unique_id")["categories"].agg(list)
    return hist[hist.map(len) >= min_orders].to_dict()


class PopularityRecommender:
    """Baseline: most frequently purchased categories (order-level counts)."""

    def fit(self, baskets):
        self.counts = Counter()
        for b in baskets:
            self.counts.update(b)
        self.ranked = [c for c, _ in self.counts.most_common()]
        return self

    def recommend(self, exclude=(), k: int = 5) -> list:
        exclude = set(exclude)
        return [c for c in self.ranked if c not in exclude][:k]


class BoughtTogetherRecommender:
    """'Frequently bought together': score(c | basket) = sum over a in basket of
    P(c in basket | a in basket), learned from multi-category baskets only.
    If the rules give fewer than k candidates, the rest is filled from popularity."""

    def fit(self, multi_baskets, popularity: PopularityRecommender):
        self.pop = popularity
        self.item = Counter()
        self.pair = defaultdict(Counter)
        for b in multi_baskets:
            if len(b) < 2:
                continue
            self.item.update(b)
            for a, c in itertools.permutations(b, 2):
                self.pair[a][c] += 1
        return self

    def recommend(self, basket, k: int = 5) -> list:
        basket = set(basket)
        scores = Counter()
        for a in basket:
            n_a = self.item.get(a, 0)
            if n_a:
                for c, n in self.pair[a].items():
                    if c not in basket:
                        scores[c] += n / n_a
        recs = [c for c, _ in scores.most_common(k)]
        if len(recs) < k:  # fill with popularity
            recs += [c for c in self.pop.recommend(exclude=basket | set(recs), k=k - len(recs))]
        return recs


class NextOrderRecommender:
    """Predict the categories of a customer's NEXT order.

    strategies
      'popularity'   : same list for everyone (baseline)
      'repeat_last'  : categories from the customer's own history, most recent
                       order first (baseline — "they'll buy the same kind of thing")
      'transition'   : learned P(category in next order | category in previous order)
      'hybrid'       : transition scores + a bonus for categories the customer
                       already bought (recency-weighted)
    Each strategy is padded with popularity if it produces fewer than k items.
    """

    def __init__(self, own_bonus: float = 0.15):
        self.own_bonus = own_bonus

    def fit(self, histories, popularity_baskets=None):
        """histories: iterable of chronological lists of category sets (train customers)."""
        self.trans = defaultdict(Counter)
        self.n_from = Counter()
        pop_baskets = []
        for orders in histories:
            pop_baskets.extend(orders)
            for prev, nxt in zip(orders[:-1], orders[1:]):
                for a in prev:
                    self.n_from[a] += 1
                    for c in nxt:
                        self.trans[a][c] += 1
        self.pop = PopularityRecommender().fit(popularity_baskets or pop_baskets)
        return self

    def _transition_scores(self, last_order) -> Counter:
        scores = Counter()
        for a in last_order:
            n = self.n_from.get(a, 0)
            if n:
                for c, cnt in self.trans[a].items():
                    scores[c] += cnt / n
        return scores

    def recommend(self, history, k: int = 5, strategy: str = "hybrid",
                  exclude_owned: bool = False) -> list:
        """exclude_owned=True -> 'discovery' mode: never suggest a category the customer
        already bought (what a cross-sell / new-category promo actually needs)."""
        owned = set().union(*history) if (history and exclude_owned) else set()
        if strategy == "popularity" or not history:
            return self.pop.recommend(exclude=owned, k=k)

        own = Counter()  # recency-weighted ownership: latest order weight 1, previous 0.5, ...
        for i, order in enumerate(reversed(history)):
            for c in order:
                own[c] = max(own[c], 1.0 / (2 ** i))

        if strategy == "repeat_last":
            ranked = [c for c, _ in own.most_common()]
        elif strategy == "transition":
            ranked = [c for c, _ in self._transition_scores(history[-1]).most_common()]
        elif strategy == "hybrid":
            scores = self._transition_scores(history[-1])
            for c, w in own.items():
                scores[c] += self.own_bonus * w
            ranked = [c for c, _ in scores.most_common()]
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        ranked = [c for c in ranked if c not in owned]
        recs = ranked[:k]
        if len(recs) < k:
            recs += self.pop.recommend(exclude=owned | set(recs), k=k - len(recs))
        return recs



def _folds(n: int, n_folds: int, seed: int):
    idx = np.random.RandomState(seed).permutation(n)
    return np.array_split(idx, n_folds)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% CI for the mean of `values` (resampling cases)."""
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_bought_together(orders: pd.DataFrame, ks=(1, 3, 5), n_folds: int = 5, seed: int = 42):
    """Leave-one-category-out on multi-category orders.
    For each test basket and each category in it: hide that category, give the
    rest to the recommender, hit if the hidden one is in the top-k.
    Models are fit on the training folds only.

    Returns (summary DataFrame, per-case DataFrame)."""
    baskets = orders["categories"].tolist()
    multi_idx = [i for i, b in enumerate(baskets) if len(b) >= 2]
    single = [baskets[i] for i, b in enumerate(baskets) if len(b) < 2]
    multi = [baskets[i] for i in multi_idx]

    cases = []  # one row per (basket, hidden category, model)
    for f, test_ids in enumerate(_folds(len(multi), n_folds, seed)):
        test_set = set(test_ids.tolist())
        train_multi = [b for i, b in enumerate(multi) if i not in test_set]
        pop_all = PopularityRecommender().fit(single + train_multi)
        pop_multi = PopularityRecommender().fit(train_multi)
        model = BoughtTogetherRecommender().fit(train_multi, pop_all)
        for i in test_ids:
            b = multi[i]
            for hidden in b:
                given = b - {hidden}
                kmax = max(ks)
                recs = {
                    "bought_together": model.recommend(given, kmax),
                    "popularity_all_orders": pop_all.recommend(given, kmax),
                    "popularity_multi_cat": pop_multi.recommend(given, kmax),
                }
                for name, r in recs.items():
                    row = {"basket": i, "model": name}
                    for k in ks:
                        row[f"hit@{k}"] = float(hidden in r[:k])
                    cases.append(row)
    cases = pd.DataFrame(cases)
    return _summarize(cases, ks, unit="basket", n_units=len(multi)), cases


def evaluate_next_order(orders: pd.DataFrame, ks=(1, 3, 5), n_folds: int = 5, seed: int = 42,
                        strategies=("popularity", "repeat_last", "transition", "hybrid"),
                        discovery: bool = False):
    """Repeat customers only (>=2 delivered orders). Hide each customer's LAST order;
    the earlier orders are the history. Transition statistics are learned from
    training customers only.

    discovery=True: the harder, more useful question — can we predict NEW categories
    (ones the customer never bought before)? Owned categories are excluded from the
    recommendations and from the truth; customers whose last order has no new
    category are skipped. ('repeat_last' is meaningless here and is dropped.)

    Returns (summary DataFrame, per-case DataFrame)."""
    hist = customer_histories(orders, min_orders=2)
    custs = list(hist)
    cases = []
    for test_ids in _folds(len(custs), n_folds, seed):
        test_set = set(test_ids.tolist())
        train = [hist[c] for i, c in enumerate(custs) if i not in test_set]
        model = NextOrderRecommender().fit(train)
        for i in test_ids:
            h = hist[custs[i]]
            history, truth = h[:-1], set(h[-1])
            if discovery:
                truth = truth - set().union(*history)
                if not truth:
                    continue
            for s in strategies:
                if discovery and s == "repeat_last":
                    continue
                r = model.recommend(history, max(ks), strategy=s, exclude_owned=discovery)
                row = {"customer": custs[i], "model": s}
                for k in ks:
                    top = set(r[:k])
                    row[f"hit@{k}"] = float(len(top & truth) > 0)
                    row[f"recall@{k}"] = len(top & truth) / len(truth)
                cases.append(row)
    cases = pd.DataFrame(cases)
    n_units = cases["customer"].nunique() if discovery else len(custs)
    return _summarize(cases, ks, unit="customer", n_units=n_units, extra_metric="recall"), cases


def _summarize(cases: pd.DataFrame, ks, unit: str, n_units: int, extra_metric: str = None) -> pd.DataFrame:
    metrics = [f"hit@{k}" for k in ks] + ([f"{extra_metric}@{k}" for k in ks] if extra_metric else [])
    rows = []
    for model, g in cases.groupby("model", sort=False):
        row = {"model": model, "n_cases": len(g), f"n_{unit}s": n_units}
        for m in metrics:
            row[m] = g[m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_baseline(cases: pd.DataFrame, model: str, baseline: str, metric: str = "hit@3",
                        cluster_col: str = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison with a bootstrap CI on the difference (model - baseline).
    Resamples whole baskets/customers (cluster_col) so repeated cases of the same
    basket are not treated as independent. Returns a dict with a plain-English verdict."""
    cluster_col = cluster_col or ("basket" if "basket" in cases else "customer")
    a = cases[cases["model"] == model].groupby(cluster_col)[metric].mean()
    b = cases[cases["model"] == baseline].groupby(cluster_col)[metric].mean()
    diff = (a - b.reindex(a.index)).to_numpy()
    lo, hi = _bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    mean_diff = float(diff.mean())
    base = float(b.mean())
    if lo > 0:
        verdict = "better than baseline (95% CI excludes 0)"
    elif hi < 0:
        verdict = "WORSE than baseline (95% CI excludes 0)"
    else:
        verdict = "not distinguishable from baseline (95% CI includes 0)"
    return {"model": model, "baseline": baseline, "metric": metric,
            "model_score": base + mean_diff, "baseline_score": base,
            "abs_diff": mean_diff, "rel_lift_pct": (mean_diff / base * 100) if base else float("nan"),
            "ci95_low": lo, "ci95_high": hi, "verdict": verdict}


def save_eval_chart(summary: pd.DataFrame, title: str, filename: str, out_dir: str = "reports",
                    ks=(1, 3, 5)) -> str:
    """Grouped bars: hit-rate@k for every model in an evaluation summary. Saved, never shown."""
    import os
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    data = summary.set_index("model")[[f"hit@{k}" for k in ks]]
    ax = data.plot(kind="bar", figsize=(9, 5), rot=15)
    ax.set_ylabel("hit rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.figure.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)
    return path


if __name__ == "__main__":
    from src.data_loader import get_connection

    orders = load_orders(get_connection())
    s1, c1 = evaluate_bought_together(orders)
    print("== Frequently bought together ==")
    print(s1.round(3).to_string(index=False))
    s2, c2 = evaluate_next_order(orders)
    print("\n== Next order (repeat customers) ==")
    print(s2.round(3).to_string(index=False))
