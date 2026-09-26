
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_connection
from src.eda import (
    compute_basket_sizes, compute_category_volume, compute_repeat_rate,
    compute_reorder_gaps, compute_customer_value, compute_revenue_concentration,
)
from src.segmentation import compute_rfm, add_segments, segment_profile
from src.promo_generator import Assumptions, build_offers, approve_offers, export_customer_codes

st.set_page_config(page_title="Project 9 — Order Intelligence", layout="wide")


# --------------------------------------------------------------------------
# Cached resources / data — safe to recompute automatically, never mutated
# --------------------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    if not os.path.exists("olist.db"):
        st.error(
            "olist.db not found. Run notebooks/Data_Loading_Intro_SQL.ipynb "
            "first to build the database, then restart this app."
        )
        st.stop()
    return get_connection()


@st.cache_data
def load_eda_data(_conn):
    basket_sizes = compute_basket_sizes(_conn)
    category_volume = compute_category_volume(_conn)
    repeat_rate, _ = compute_repeat_rate(_conn)
    gaps = compute_reorder_gaps(_conn)
    customer_value = compute_customer_value(_conn)
    top20_share = compute_revenue_concentration(customer_value)
    return basket_sizes, category_volume, repeat_rate, gaps, customer_value, top20_share


@st.cache_data
def load_segment_profile(_conn):
    rfm = compute_rfm(_conn, merge_gap_hours=1.0)
    segmented = add_segments(rfm)
    profile = segment_profile(segmented)
    return segmented, profile


@st.cache_data(show_spinner=False)
def load_orders_for_recs(_conn):
    from src.recommender import load_orders, customer_histories
    orders = load_orders(_conn, merge_gap_hours=1.0)
    hist = customer_histories(orders)
    return orders, hist


@st.cache_data(show_spinner=False)
def run_bought_together_eval(_orders):
    from src.recommender import evaluate_bought_together
    return evaluate_bought_together(_orders)


@st.cache_data(show_spinner=False)
def run_next_order_eval(_orders):
    from src.recommender import evaluate_next_order
    return evaluate_next_order(_orders)


@st.cache_data
def load_base_offers(_conn):
    return build_offers(_conn, Assumptions())


@st.cache_resource
def get_chat_agent():
    from src.chat_agent import build_agent
    return build_agent(verbose=False)


conn = get_db_connection()

st.title("Project 9 — Order Intelligence Dashboard")
st.caption("Robusta AI Internship 2026 — Data Intelligence over Orders")

(tab_eda, tab_chat, tab_segments, tab_recs,
 tab_promo, tab_narratives) = st.tabs([
    "📊 EDA", "💬 Chat With Your Data", "👥 Customer Segments",
    "🎯 Recommendations", "🎁 Promo Generator", "📝 LLM Narratives",
])


# --------------------------------------------------------------------------
# TAB 1 — EDA (Task 3)
# --------------------------------------------------------------------------
with tab_eda:
    st.header("Exploratory Data Analysis")

    basket_sizes, category_volume, repeat_rate, gaps, customer_value, top20_share = load_eda_data(conn)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. basket size", f"{basket_sizes['num_items'].mean():.2f} items")
    col2.metric("Repeat purchase rate", f"{repeat_rate:.2%}")
    col3.metric("Top 20% customers drive", f"{top20_share:.1%} of revenue")

    st.subheader("Basket size distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    basket_sizes["num_items"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Number of items")
    ax.set_ylabel("Number of orders")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 10 categories by volume")
    fig, ax = plt.subplots(figsize=(9, 5))
    category_volume.head(10).plot(kind="barh", x="category", y="times_purchased", ax=ax, legend=False)
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Days between repeat orders")
    if len(gaps) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        gaps.hist(bins=30, ax=ax)
        ax.set_xlabel("Days")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough repeat customers in the current data to show a gap distribution.")

    with st.expander("Raw customer value table"):
        st.dataframe(customer_value.head(50), use_container_width=True)


# --------------------------------------------------------------------------
# TAB 2 — Chat With Your Data (Task 4)
# --------------------------------------------------------------------------
with tab_chat:
    st.header("Ask a question about the order data")
    st.caption("Natural-language → SQL, powered by a LangChain agent (Gemini). "
               "Single-turn: each question is answered independently.")

    if "GOOGLE_API_KEY" not in os.environ:
        st.warning(
            "GOOGLE_API_KEY not set. Make sure a .env file with your key exists "
            "in the project root, then restart this app."
        )
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)

        question = st.chat_input("e.g. Which product category has the highest total revenue?")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Querying the database..."):
                    try:
                        agent = get_chat_agent()
                        result = agent.invoke({"input": question})
                        answer = result["output"]
                    except Exception as e:
                        answer = f"Something went wrong: {e}"
                    st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))


# --------------------------------------------------------------------------
# TAB 3 — Customer Segments (Task 7)
# --------------------------------------------------------------------------
with tab_segments:
    st.header("Customer Segments (RFM)")
    st.caption("Recency / Frequency / Monetary — six rule-based segments, "
               "with orders merged into shopping trips at a 1-hour gap.")

    segmented, profile = load_segment_profile(conn)

    st.dataframe(profile, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        profile["pct_customers"].plot(kind="barh", ax=ax, color="#4C72B0")
        ax.invert_yaxis()
        ax.set_title("% of customers")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        profile["pct_revenue"].plot(kind="barh", ax=ax, color="#55A868")
        ax.invert_yaxis()
        ax.set_title("% of revenue")
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------------------------------
# TAB 4 — Recommendations (Task 6)
# --------------------------------------------------------------------------
with tab_recs:
    st.header("Recommendation Engine — Evaluation")
    st.caption("Two recommenders, evaluated with cross-validation and bootstrap "
               "confidence intervals against popularity baselines. This can take "
               "a moment to compute the first time.")

    orders, hist = load_orders_for_recs(conn)
    repeat_customers = sum(len(h) >= 2 for h in hist.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("Orders / trips", len(orders))
    col2.metric("Customers", len(hist))
    col3.metric("Repeat customers", repeat_customers)

    st.subheader("1. Frequently bought together")
    if st.button("Run evaluation", key="run_fbt"):
        with st.spinner("Running 5-fold cross-validation..."):
            from src.recommender import compare_to_baseline
            fbt_summary, fbt_cases = run_bought_together_eval(orders)
            st.session_state["fbt_summary"] = fbt_summary
            st.session_state["fbt_cases"] = fbt_cases

    if "fbt_summary" in st.session_state:
        st.dataframe(st.session_state["fbt_summary"].round(3), use_container_width=True)
        from src.recommender import compare_to_baseline
        rows = [compare_to_baseline(st.session_state["fbt_cases"], "bought_together", b, "hit@3")
                for b in ["popularity_multi_cat", "popularity_all_orders"]]
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)

    st.subheader("2. Next order (repeat customers only)")
    if st.button("Run evaluation", key="run_next"):
        with st.spinner("Running 5-fold cross-validation..."):
            next_summary, next_cases = run_next_order_eval(orders)
            st.session_state["next_summary"] = next_summary
            st.session_state["next_cases"] = next_cases

    if "next_summary" in st.session_state:
        st.dataframe(st.session_state["next_summary"].round(3), use_container_width=True)
        from src.recommender import compare_to_baseline
        rows = [compare_to_baseline(st.session_state["next_cases"], m, "popularity", "hit@3")
                for m in ["repeat_last", "transition", "hybrid"]]
        rows += [compare_to_baseline(st.session_state["next_cases"], m, "repeat_last", "hit@3")
                 for m in ["transition", "hybrid"]]
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)
        st.caption("The second table compares transition/hybrid against `repeat_last` directly — "
                   "this is what reveals whether the learned model adds anything over the simple "
                   "'rebuy the same category' rule.")

    st.subheader("Try it — one example")
    from src.recommender import PopularityRecommender, BoughtTogetherRecommender, NextOrderRecommender
    pop = PopularityRecommender().fit(orders["categories"])
    multi = [b for b in orders["categories"] if len(b) >= 2]
    fbt_model = BoughtTogetherRecommender().fit(multi, pop)
    all_categories = sorted(pop.ranked)
    picked_category = st.selectbox("Pick a category", all_categories)
    st.write(f"Frequently bought with **{picked_category}**:", fbt_model.recommend({picked_category}, k=5))


# --------------------------------------------------------------------------
# TAB 5 — Promo Generator (Task 8)
# --------------------------------------------------------------------------
with tab_promo:
    st.header("Promo-Code Generator")
    st.caption("One proposed offer per segment, with expected margin impact. "
               "Nothing is issued until a named approver signs off below.")

    if "offers" not in st.session_state:
        st.session_state["offers"] = load_base_offers(conn).copy()

    offers = st.session_state["offers"]

    display_cols = ["segment", "customers", "avg_order_value", "baseline_p0", "discount",
                    "break_even_uplift", "net_per_customer_low", "net_per_customer_base",
                    "net_per_customer_high", "recommendation", "status"]
    st.dataframe(offers[display_cols].round(4), use_container_width=True)

    st.subheader("Approve offers")
    pending = offers[offers["status"] != "approved"]["segment"].tolist()
    col1, col2 = st.columns([2, 1])
    with col1:
        to_approve = st.multiselect("Segments to approve", pending)
        approver = st.text_input("Approved by (your name)")
    with col2:
        force = st.checkbox("Force-approve 'not_recommended' segments")
        approve_clicked = st.button("Approve selected", type="primary")

    if approve_clicked:
        if not to_approve or not approver:
            st.warning("Pick at least one segment and enter an approver name.")
        else:
            try:
                st.session_state["offers"] = approve_offers(offers, to_approve, approver, force=force)
                st.success(f"Approved: {', '.join(to_approve)}")
                st.rerun()
            except ValueError as e:
                st.error(f"Blocked: {e}")

    st.subheader("Export approved codes")
    approved_count = (st.session_state["offers"]["status"] == "approved").sum()
    st.write(f"{approved_count} segment(s) currently approved.")
    if st.button("Generate codes for approved offers"):
        with st.spinner("Generating unique customer codes..."):
            codes = export_customer_codes(conn, st.session_state["offers"], merge_gap_hours=1.0)
        if codes.empty:
            st.info("No approved offers yet — nothing to export.")
        else:
            st.dataframe(codes.head(50), use_container_width=True)
            st.download_button(
                "Download all codes as CSV",
                data=codes.to_csv(index=False),
                file_name="promo_codes_approved.csv",
                mime="text/csv",
            )


# --------------------------------------------------------------------------
# TAB 6 — LLM Narratives (Task 9)
# --------------------------------------------------------------------------
with tab_narratives:
    st.header("Plain-Language Narratives")
    st.caption("Grounded in the exact numbers from Segments (Task 7) and "
               "Promo Generator (Task 8) — nothing here is invented beyond "
               "what those tables already show.")

    if "GOOGLE_API_KEY" not in os.environ:
        st.warning("GOOGLE_API_KEY not set — narratives need the same key as the Chat tab.")
    else:
        from src.narratives import generate_segment_description, generate_promo_rationale

        st.subheader("Segment description")
        _, profile = load_segment_profile(conn)
        selected_segment = st.selectbox("Choose a segment to describe", profile.index.tolist())
        if st.button("Generate segment description"):
            with st.spinner("Generating..."):
                stats = profile.loc[selected_segment].to_dict()
                stats["segment"] = selected_segment
                try:
                    st.success(generate_segment_description(stats))
                except Exception as e:
                    st.error(f"Could not generate description: {e}")

        st.divider()

        st.subheader("Promo rationale")
        if "offers" not in st.session_state:
            st.info("Visit the Promo Generator tab first to build the offers table.")
        else:
            offers = st.session_state["offers"]
            selected_offer_segment = st.selectbox(
                "Choose a segment's offer to explain", offers["segment"].tolist(), key="narr_offer_seg"
            )
            if st.button("Generate promo rationale"):
                with st.spinner("Generating..."):
                    row = offers[offers["segment"] == selected_offer_segment].iloc[0].to_dict()
                    try:
                        st.success(generate_promo_rationale(row))
                    except Exception as e:
                        st.error(f"Could not generate rationale: {e}")
