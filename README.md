# Project 9 — Data Intelligence over Orders
### Robusta AI Internship 2026

An intelligence layer built on top of e-commerce order data: order
analytics, a recommendation engine, and a margin-aware promo generator.

---

## Project Description

Robusta's internal order data sits unused as raw transactional history.
This project turns it into three usable outputs:

1. **An analytics layer** over order history — basket composition,
   repeat-purchase behavior, and reorder timing per customer.
2. **A recommendation engine** — "frequently bought together" and
   "next-order" suggestions, evaluated offline against a popularity
   baseline.
3. **A promo-code generator** — targeted offers per customer segment
   with expected margin impact stated explicitly, for human
   approval rather than autonomous issuance.

---

## Dataset

### Comparison: Instacart vs. Olist

| Factor | Instacart | Olist (chosen) |
|---|---|---|
| Scale | ~3.4M orders, 200K+ users | ~100K orders, 2016–2018 |
| Native reorder flag | Yes | No — derived via `customer_unique_id` |
| Price / payment data | None | Yes — order value, freight, payment method |
| Structure | 6 CSVs | 9 relational CSVs |
| Fit for promo/margin deliverable | Weak — no price data | Moderate — real prices enable real margin estimates |

**Decision: Olist.** It's the only dataset that reasonably supports
**all three** deliverables — Instacart is stronger for recommendations
alone, but has zero price data, which would leave the promo/margin
deliverable built on fully simulated numbers. Olist's multi-seller,
marketplace structure is also more representative of the kind of
client platforms a software development company like Robusta
typically builds.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

### Schema — 9 relational tables

| File | Contents |
|---|---|
| `olist_customers_dataset.csv` | Customer IDs, unique customer ID (tracks repeat buyers), location |
| `olist_orders_dataset.csv` | Order status, purchase timestamp, delivery dates |
| `olist_order_items_dataset.csv` | Product + seller per order, price, freight value |
| `olist_order_payments_dataset.csv` | Payment type, installments, payment value |
| `olist_order_reviews_dataset.csv` | Review scores, comments |
| `olist_products_dataset.csv` | Product category, weight, dimensions |
| `olist_sellers_dataset.csv` | Seller location |
| `olist_geolocation_dataset.csv` | Zip code → lat/lng mapping |
| `product_category_name_translation.csv` | Portuguese → English category names |

**Note:** Olist has no native reorder flag — repeat-purchase behavior
is derived by joining on `customer_unique_id` across orders (see
`src/sql_basics.py::get_repeat_customers`).

---

## Tech Stack

| Tool | Role |
|---|---|
| **SQLite** | Data storage, filtering, joins, aggregation |
| **Pandas / NumPy** | Numerical computation, matrix operations |
| **Matplotlib** | EDA visualizations |
| **mlxtend** | Market basket analysis (Apriori, association rules) |
| **scikit-learn** | Recommendation evaluation, clustering |
| **LangChain + Gemini** | Natural-language SQL querying |
| **FastAPI / Streamlit** *(planned)* | Final packaging and UI |

**Architectural principle:** SQL owns the data layer (storage,
filtering, joining, aggregating); Python/Pandas owns the modeling
layer (ML, statistics, business logic). This mirrors how these
systems are architected in production.

---

## Project Structure

```
project9/
├── data/                            # raw CSVs (gitignored)
├── olist.db                         # SQLite database (gitignored)
├── src/
│   ├── data_loader.py               # Task 1 — load CSVs into SQLite
│   ├── sql_basics.py                # Task 2 — core SQL query functions
│   ├── eda.py                       # Task 3 — EDA functions
│   └── chat_agent.py                # Task 4 — LangChain SQL agent
├── notebooks/
│   ├── Data_Loading_Intro_SQL.ipynb # Tasks 1-2
│   ├── Task3_Data_Exploration.ipynb
│   └── Task4_chatting_with_data.ipynb
├── reports/                         # saved charts, findings
├── requirements.txt
├── .gitignore
└── README.md
```

Notebooks are kept intentionally — they're used for interactive
exploration and verification — but contain minimal logic themselves;
all real logic lives in `src/` as reusable, importable functions.

---

## Task Log

### ✅ Task 1 — Data Loading
`src/data_loader.py` reads all 9 Olist CSVs and loads them into a
single SQLite database (`olist.db`). Verified by querying
`sqlite_master` to confirm all 9 tables exist.

### ✅ Task 2 — Core SQL Techniques
`src/sql_basics.py` implements and demonstrates:
- `SELECT` / `WHERE` — filtering to delivered orders only
- `JOIN` — connecting `orders` and `order_items`
- `GROUP BY` + aggregates — orders per customer
- Multi-table `JOIN` + aggregate — revenue by category
- Subquery / `HAVING` — deriving repeat customers (no native reorder flag in Olist)
- Date functions (`julianday`) — customer activity span, feeds into RFM later

### ✅ Task 3 — Exploratory Data Analysis (EDA)
`src/eda.py` covers:
- Basket size distribution
- Top categories by volume
- Repeat-purchase rate
- Reorder timing (gap between repeat orders)
- Customer lifetime value
- Revenue concentration (80/20 check — do top customers drive most revenue?)

Charts are saved to `reports/`.

### ✅ Task 4 — Natural-Language Data Query (Chat With Your Data)

`src/chat_agent.py` implements a LangChain SQL agent that answers
plain-English questions about the order data.

**Why LangChain instead of a single open-source chat-with-data agent:**

We evaluated open-source alternatives for this task but chose
LangChain's SQL agent for two practical reasons:

1. **Multi-file/relational data support** — our data spans 9
   relational CSV files. Several open-source "chat with your data"
   agents are built around single-file or single-document inputs and
   don't handle multi-table relational joins well out of the box.
2. **Database-agent reliability** — open-source agents built
   specifically for database querying were either not free to use at
   the quality needed, or had reliability issues (inconsistent SQL
   generation, poor error handling) in testing.

LangChain's SQL agent connects directly to `olist.db`, understands the
full relational schema, and generates correct multi-table SQL queries
from natural-language questions.

**Example:**
> **Q:** "Which product category has the highest total revenue?"
> **A:** The agent generates and runs the appropriate JOIN + GROUP BY
> SQL query against `olist.db` and returns the answer in plain English.

**Setup:** requires a Gemini API key set as the `GOOGLE_API_KEY`
environment variable. Model used: **`gemini-3.1-flash-lite`**
(`gemini-1.5-flash` was deprecated and returns a 404 as of 2026).

**Scope note:** this is a single-turn natural-language-to-SQL query
tool, not a multi-turn conversational chatbot — each question is
answered independently with no memory of prior questions.

---

## Upcoming Tasks

| Task | Goal |
|---|---|
| Task 5 | Market basket analysis — Apriori / association rules |
| Task 6 | Recommendation engine + evaluation vs. popularity baseline |
| Task 7 | Customer segmentation via RFM |
| Task 8 | Promo generator with margin impact |
| Task 9 | LLM-generated plain-language segment/promo narratives |
| Task 10 | Streamlit UI / FastAPI packaging |
| Task 11 | Evaluation write-up |
| Task 12 | Publish |

---

## Setup

```bash
pip install -r requirements.txt
```

Download the Olist dataset from
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and place
the 9 CSVs in a `data/` folder in the project root.

Set your Gemini API key (required for Task 4):
```bash
# Windows PowerShell
setx GOOGLE_API_KEY "your-key-here"
```

Run `notebooks/Data_Loading_Intro_SQL.ipynb` first to build the
database, then proceed through the other notebooks in order.
