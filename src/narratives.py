"""
src/narratives.py
Task 9 — LLM narrative layer: turns Task 7 (segment_profile) and Task 8
(build_offers) numeric rows into grounded, plain-language descriptions.

Design principle: the LLM is given ONLY the numbers in the row and explicit
instructions not to invent anything beyond them. This is grounded prompt
engineering, not free-form generation — the goal is a faithful translation
of numbers into sentences a marketer/manager can act on, not creative writing.

Depends on: Task 7 (segment_profile output), Task 8 (build_offers output),
            the GOOGLE_API_KEY set up in Task 4 (.env, python-dotenv).
Feeds:      Task 10/11 — these narratives are what get shown in the
            Streamlit app and referenced in the evaluation write-up.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"

SEGMENT_PROMPT_TEMPLATE = """You are describing a customer segment for a marketing team.
Use ONLY the numbers provided below. Do not invent demographic details,
motivations, or any fact not explicitly given. Do not speculate about who
these customers are beyond what the numbers show. Write 2-3 plain sentences
a marketer could act on.

Segment: {segment}
Number of customers: {customers}
% of total customers: {pct_customers}%
% of total revenue: {pct_revenue}%
Average recency (days since last order): {avg_recency_days}
Average number of orders: {avg_orders}
Average total spend: R$ {avg_spend}
Average order value: R$ {avg_order_value}
"""

PROMO_PROMPT_TEMPLATE = """You are explaining a proposed promotional offer to a manager
who needs to approve or reject it. Use ONLY the numbers provided below.
Do not invent customer behavior, motivations, or facts not given. State
plainly whether this looks like a good bet, a risky test, or a bad idea,
and why, using the actual numbers. Write 2-4 plain sentences.

Segment: {segment}
Number of customers targeted: {customers}
Average order value: R$ {avg_order_value}
Baseline chance of reordering with no promo (90 days): {baseline_p0}
Proposed discount: {discount}
Break-even uplift needed (extra reorder probability the promo must create): {break_even_uplift}
Expected net margin impact per customer - worst case: R$ {net_per_customer_low}
Expected net margin impact per customer - base case: R$ {net_per_customer_base}
Expected net margin impact per customer - best case: R$ {net_per_customer_high}
System recommendation: {recommendation}
Approval status: {status}
"""


def _get_llm(model_name: str = MODEL_NAME, temperature: float = 0.2):
    """temperature=0.2 (not 0) — narrative phrasing benefits from slight
    variation, unlike Task 4's SQL agent which needs deterministic queries."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Make sure .env contains GOOGLE_API_KEY=... "
            "in the project root (same setup as Task 4)."
        )
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _extract_text(response) -> str:
    """Handles both response.content shapes seen across langchain-google-genai
    versions: a plain string, or a list of content blocks (dicts or strings).
    Fixes: AttributeError: 'list' object has no attribute 'strip'."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts).strip()
    return str(content).strip()


def generate_segment_description(segment_stats: dict, llm=None) -> str:
    """segment_stats: one row from Task 7's segment_profile(), as a dict.
    Example: profile.reset_index().to_dict('records')[0]"""
    llm = llm or _get_llm()
    prompt = SEGMENT_PROMPT_TEMPLATE.format(**segment_stats)
    response = llm.invoke(prompt)
    return _extract_text(response)


def generate_promo_rationale(promo_row: dict, llm=None) -> str:
    """promo_row: one row from Task 8's build_offers(), as a dict.
    Example: offers.to_dict('records')[0]"""
    llm = llm or _get_llm()
    prompt = PROMO_PROMPT_TEMPLATE.format(**promo_row)
    response = llm.invoke(prompt)
    return _extract_text(response)


def generate_all_segment_descriptions(profile_df, llm=None) -> dict:
    """profile_df: Task 7's segment_profile() output (segment as index).
    Returns {segment_name: description}. Reuses one llm instance across
    all calls instead of creating a new connection per segment."""
    llm = llm or _get_llm()
    out = {}
    for segment, row in profile_df.iterrows():
        stats = row.to_dict()
        stats["segment"] = segment
        out[segment] = generate_segment_description(stats, llm=llm)
    return out


def generate_all_promo_rationales(offers_df, llm=None) -> dict:
    """offers_df: Task 8's build_offers() output.
    Returns {segment_name: rationale}."""
    llm = llm or _get_llm()
    out = {}
    for _, row in offers_df.iterrows():
        out[row["segment"]] = generate_promo_rationale(row.to_dict(), llm=llm)
    return out


if __name__ == "__main__":
    from src.data_loader import get_connection
    from src.segmentation import compute_rfm, add_segments, segment_profile
    from src.promo_generator import build_offers

    conn = get_connection()
    profile = segment_profile(add_segments(compute_rfm(conn, merge_gap_hours=1.0)))
    offers = build_offers(conn)

    print("=== Segment descriptions ===")
    for seg, desc in generate_all_segment_descriptions(profile).items():
        print(f"\n{seg}:\n{desc}")

    print("\n\n=== Promo rationales ===")
    for seg, rationale in generate_all_promo_rationales(offers).items():
        print(f"\n{seg}:\n{rationale}")
