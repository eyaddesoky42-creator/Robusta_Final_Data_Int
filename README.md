# Project 9 — Data Intelligence over Orders

**Robusta AI Internship 2026 — Month 2 Project**

An intelligence layer over Olist's historical e-commerce orders: analytics, a
recommendation engine, and a promo-code generator with margin logic — built
honestly, with every claim backed by a measured number or clearly flagged as
an assumption.

**Live demo:** (https://robustafinaldataint-cgqqfkq6c3wxuapufod6no.streamlit.app/)

---

## 1. What this project does

1. **Analytics layer** — basket composition, repeat-purchase behaviour, reorder
   timing per customer.
2. **Recommendation engine** — "frequently bought together" and "next order"
   suggestions, evaluated against a popularity baseline with statistical
   confidence, honest about where the lift is small.
3. **Promo-code generator** — targeted offers per customer segment with
   expected margin impact, for human approval — nothing is issued automatically.

## 2. Dataset

[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) —
~99k orders, 2016–2018, 9 relational CSVs (customers, orders, order items,
payments, reviews, products, sellers, geolocation, category translations).

## 3. Architecture

```
SQL (SQLite)         →  data storage, filtering, joins, aggregation
Pandas / NumPy        →  modeling layer: recommenders, RFM, promo maths
scikit-learn          →  KMeans cross-check, evaluation metrics
LangChain + Gemini    →  natural-language "chat with your data"
Streamlit             →  the app you're looking at
```

All logic lives in `src/`; notebooks in `Jupyters/` are thin wrappers that
import and call it, so nothing is duplicated between the two.

```
project9/
├── app.py                  # Streamlit app (6 tabs — see Section 5)
├── src/                     # all logic: data_loader, eda, basket_analysis,
│                             #   recommender, segmentation, promo_generator,
│                             #   chat_agent, narratives, data_prep
├── Jupyters/                # one notebook per task, thin wrappers over src/
├── data/                    # 9 Olist CSVs (source of truth)
├── reports/                 # generated charts, offer proposals
├── .env                     # GOOGLE_API_KEY (gitignored, never committed)
├── requirements.txt
└── railway.json / (n/a)     # not needed on Streamlit Community Cloud
```

`olist.db` is **not** committed — the app builds it automatically from `data/`
on first run (see `get_db_connection()` in `app.py`).

## 4. Task-by-task summary

| Task | What it does | Headline finding |
|---|---|---|
| 1–2 | Load CSVs into SQLite, core SQL | Fixed a bug grouping by `customer_id` (unique per order) instead of `customer_unique_id` |
| 3 | EDA | ~10% of orders have 2+ items; raw repeat rate ~3% |
| 4 | Chat with your data (LangChain + Gemini) | Secured API key via `.env` after an earlier hardcoding incident |
| 5 | Market basket analysis | Only 1.02% of orders span 2+ categories; rules mined on those only |
| 6 | Recommendation engine + evaluation | Bought-together beats popularity (real lift); next-order beats popularity but a simple "rebuy the same category" rule matches the learned model |
| 7 | RFM segmentation + KMeans cross-check | One-time high spenders are 38% of customers, 72% of revenue — the second purchase is the main lever |
| 8 | Promo generator with margin logic | Champions shouldn't be discounted; other segments proposed as A/B tests, not blanket sends |
| 9 | LLM narrative layer | Segment/offer descriptions grounded strictly in the numbers from Tasks 7–8 |
| 10 | Streamlit app | Six tabs, one per capability |
| 11 | Evaluation write-up | Full honest accounting of what worked, what didn't, and why |
| 12 | Deployment | Streamlit Community Cloud (after a Railway detour — see Section 7) |

## 5. The app's six tabs

1. **EDA** — basket sizes, category volume, repeat-purchase rate, revenue concentration.
2. **Chat** — ask plain-English questions about the order data.
3. **Segments** — RFM table, six named segments, KMeans cross-check.
4. **Recommendations** — frequently-bought-together and next-order models, live evaluation.
5. **Promo Generator** — proposed offers, margin sensitivity, approve → export codes.
6. **Narratives** — LLM-written plain-language descriptions of segments and offers.

## 6. A data-quality decision worth knowing about

About 27% of the gaps between a customer's consecutive orders were under one
hour — almost certainly split checkouts, not real repeat purchases. `src/data_prep.py`
merges orders within a 1-hour window into a single "shopping trip" before any
task counts repeat behaviour. This is applied consistently across Tasks 5–8.

## 7. Running it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires a `.env` file in the project root:
```
GOOGLE_API_KEY=your-key-here
```

`olist.db` builds itself from `data/` on first run — no manual step needed.

## 8. Deployment note

This app is deployed on **Streamlit Community Cloud**, not Railway. Railway's
build system (Railpack) doesn't auto-detect how to start a Streamlit app the
way Streamlit Cloud does natively, which cost real debugging time before
switching. If you fork this repo and want to redeploy, Streamlit Community
Cloud requires no start-command configuration at all — just point it at
`app.py` and add `GOOGLE_API_KEY` under the app's Secrets.

## 9. Known limitations (stated plainly, not hidden)

- Next-order prediction adds little over "customers rebuy the same category" —
  reported as a negative result, not spun as a win.
- Predicting a *new* category a customer hasn't bought before shows **no**
  measurable lift over popularity, most likely due to sample size (~2,000
  repeat customers).
- Promo margin (20%) and uplift assumptions are not in Olist's data — isolated
  in one `Assumptions` class and shown as low/base/high scenarios, not a
  single confident number.
