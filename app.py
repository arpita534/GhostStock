"""
GhostStock — AI-Powered Inventory Intelligence & Mismatch Detection Platform
A single-file Streamlit prototype demonstrating a complete
Upload -> Evidence -> Detection -> Investigation -> Resolution workflow.

Run with:
    python -m streamlit run app.py
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

# =====================================================================================
# PAGE CONFIG
# =====================================================================================

st.set_page_config(
    page_title="GhostStock | Inventory Intelligence",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
# GLOBAL CONSTANTS
# =====================================================================================

REQUIRED_COLUMNS = ["Product ID", "Product", "Zone", "System Stock", "Sales", "Physical Evidence"]
NUMERIC_COLUMNS = ["System Stock", "Sales", "Physical Evidence"]

COLOR_HIGH = "#FF5C6C"
COLOR_MEDIUM = "#FFB84C"
COLOR_LOW = "#3DDC97"
COLOR_BRAND_PURPLE = "#8B5CF6"
COLOR_BRAND_CYAN = "#22D3EE"
COLOR_BG_CARD = "rgba(255,255,255,0.04)"

CHECKLIST_ITEMS = [
    "Verify physical count",
    "Check barcode scan history",
    "Check recent warehouse movement",
    "Verify storage location",
    "Check damaged stock records",
    "Reconcile inventory system",
]

INVESTIGATION_STATUSES = ["Not Started", "Investigating", "Verified", "Resolved"]
RESOLUTION_TYPES = ["Confirm Discrepancy", "Physical Count Corrected", "False Alert / Data Error"]
CONFIDENCE_LEVELS = ["High", "Medium", "Low"]

NAV_PAGES = {
    "Overview": "📊",
    "Upload & Data": "📁",
    "Evidence Input": "📸",
    "Detection Center": "🔍",
    "Investigation": "🧭",
    "Risk Alerts": "🚨",
    "Action Plan": "🗺️",
    "Inventory Health": "📈",
    "About GhostStock": "ℹ️",
}

# =====================================================================================
# CUSTOM CSS — PREMIUM SAAS STYLE
# =====================================================================================

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 10% 0%, #150f2e 0%, #0b0c1a 45%, #08080f 100%);
    color: #E7E9F5;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0f0b26 0%, #0a0a17 100%);
    border-right: 1px solid rgba(139,92,246,0.15);
}}

section[data-testid="stSidebar"] * {{
    color: #E7E9F5 !important;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

.gs-brand {{
    padding: 18px 6px 22px 6px;
    border-bottom: 1px solid rgba(139,92,246,0.18);
    margin-bottom: 18px;
}}
.gs-brand-title {{
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, {COLOR_BRAND_PURPLE}, {COLOR_BRAND_CYAN});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}
.gs-brand-sub {{
    font-size: 12.5px;
    color: #9CA3D4;
    margin-top: 2px;
    letter-spacing: 0.02em;
}}
.gs-sidebar-footer {{
    position: fixed;
    bottom: 14px;
    font-size: 11px;
    color: #676C99;
    letter-spacing: 0.03em;
}}

.gs-hero {{
    padding: 38px 40px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(139,92,246,0.20), rgba(34,211,238,0.10));
    border: 1px solid rgba(139,92,246,0.25);
    margin-bottom: 28px;
}}
.gs-hero-title {{
    font-size: 42px;
    font-weight: 800;
    margin: 0;
    color: #F5F3FF;
}}
.gs-hero-tagline {{
    font-size: 19px;
    font-weight: 600;
    color: #C9C3F5;
    margin-top: 6px;
}}
.gs-hero-desc {{
    font-size: 14.5px;
    color: #A9AEDA;
    margin-top: 14px;
    max-width: 780px;
    line-height: 1.55;
}}

.gs-section-title {{
    font-size: 21px;
    font-weight: 700;
    color: #F0EEFB;
    margin: 6px 0 2px 0;
}}
.gs-section-sub {{
    font-size: 13px;
    color: #8D92C4;
    margin-bottom: 16px;
}}

.gs-metric-card {{
    background: {COLOR_BG_CARD};
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 20px;
    height: 100%;
    transition: all 0.2s ease-in-out;
}}
.gs-metric-card:hover {{
    border-color: rgba(139,92,246,0.45);
    transform: translateY(-2px);
}}
.gs-metric-label {{
    font-size: 12.5px;
    color: #9297C4;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.gs-metric-value {{
    font-size: 30px;
    font-weight: 800;
    color: #F5F3FF;
    margin-top: 6px;
}}
.gs-metric-delta {{
    font-size: 12px;
    color: #7C82B5;
    margin-top: 4px;
}}

.gs-alert-card {{
    background: linear-gradient(135deg, rgba(255,92,108,0.16), rgba(255,92,108,0.04));
    border: 1px solid rgba(255,92,108,0.35);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 26px;
}}
.gs-alert-title {{
    font-size: 19px;
    font-weight: 800;
    color: #FF8A96;
    margin-bottom: 6px;
}}
.gs-alert-desc {{
    font-size: 13px;
    color: #C9A9AE;
    line-height: 1.5;
}}

.gs-card {{
    background: {COLOR_BG_CARD};
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 16px;
}}

.gs-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.gs-badge-high {{ background: rgba(255,92,108,0.18); color: {COLOR_HIGH}; border: 1px solid rgba(255,92,108,0.4);}}
.gs-badge-medium {{ background: rgba(255,184,76,0.18); color: {COLOR_MEDIUM}; border: 1px solid rgba(255,184,76,0.4);}}
.gs-badge-low {{ background: rgba(61,220,151,0.18); color: {COLOR_LOW}; border: 1px solid rgba(61,220,151,0.4);}}

.gs-insight {{
    background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(139,92,246,0.08));
    border: 1px solid rgba(34,211,238,0.25);
    border-radius: 16px;
    padding: 18px 22px;
    font-size: 14px;
    color: #D6D9F5;
    line-height: 1.6;
}}

.gs-intel-card {{
    background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(34,211,238,0.08));
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 24px;
}}
.gs-intel-title {{
    font-size: 17px;
    font-weight: 800;
    color: #F0EEFB;
    margin-bottom: 8px;
}}
.gs-intel-text {{
    font-size: 14px;
    color: #D6D9F5;
    line-height: 1.65;
}}

.gs-timeline-item {{
    border-left: 2px solid rgba(139,92,246,0.4);
    padding: 4px 0 14px 18px;
    margin-left: 6px;
    position: relative;
}}
.gs-timeline-item::before {{
    content: '';
    position: absolute;
    left: -6px;
    top: 6px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: {COLOR_BRAND_CYAN};
}}
.gs-timeline-event {{
    font-size: 13.5px;
    font-weight: 700;
    color: #F0EEFB;
}}
.gs-timeline-meta {{
    font-size: 11.5px;
    color: #8D92C4;
    margin-top: 1px;
}}
.gs-timeline-detail {{
    font-size: 12.5px;
    color: #B5B9E3;
    margin-top: 3px;
}}

.gs-prototype-tag {{
    display: inline-block;
    background: rgba(255,184,76,0.15);
    border: 1px solid rgba(255,184,76,0.4);
    color: {COLOR_MEDIUM};
    font-size: 11.5px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 10px;
}}

.gs-health-badge {{
    display: inline-block;
    padding: 8px 20px;
    border-radius: 999px;
    font-size: 18px;
    font-weight: 800;
}}

.stButton>button, .stDownloadButton>button {{
    background: linear-gradient(90deg, {COLOR_BRAND_PURPLE}, {COLOR_BRAND_CYAN});
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 18px;
    transition: all 0.15s ease-in-out;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    filter: brightness(1.12);
    transform: translateY(-1px);
}}

hr {{
    border-color: rgba(255,255,255,0.08);
}}

[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================================================
# SESSION STATE INITIALIZATION
# =====================================================================================

def init_session_state():
    defaults = {
        "data": None,
        "data_source": None,
        "investigation_status": {},
        "investigation_checklist": {},
        "evidence_timeline": {},
        "resolution_history": [],
        "priority_plan": None,
        "current_page": "Overview",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# =====================================================================================
# DEMO DATA GENERATION
# =====================================================================================

def generate_demo_data() -> pd.DataFrame:
    """
    Hand-crafted demo dataset guaranteed to produce a spread of
    High, Medium and Low risk products across four zones, with
    Zone B standing out as the clear top inspection priority.
    """
    rows = [
        # Product ID, Product,           Zone, System Stock, Sales, Physical Evidence
        ("P001", "Wireless Mouse",       "A",   80,   20,   58),   # small mismatch -> low
        ("P002", "Bluetooth Speaker",    "B",   50,   7,    36),   # medium-large mismatch
        ("P003", "27in Monitor",         "B",   20,   3,    3),    # large mismatch -> high
        ("P004", "USB-C Cable",          "A",   200,  60,   139),  # small mismatch -> low
        ("P005", "Mechanical Keyboard",  "C",   40,   10,   18),   # large mismatch -> high
        ("P006", "Laptop Stand",         "C",   35,   5,    27),   # small-medium -> low/med
        ("P007", "Webcam HD",            "B",   45,   15,   20),   # medium-large -> high
        ("P008", "Desk Lamp",            "D",   60,   12,   40),   # medium mismatch
        ("P009", "Noise Cancelling HP",  "D",   30,   8,    19),   # medium mismatch
        ("P010", "Graphics Tablet",      "C",   25,   5,    19),   # small mismatch -> low
        ("P011", "Portable SSD 1TB",     "B",   55,   18,   22),   # large mismatch -> high
        ("P012", "Office Chair",         "D",   15,   2,    12),   # tiny mismatch -> low
    ]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return df


def generate_sample_csv_bytes() -> bytes:
    df = generate_demo_data()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# =====================================================================================
# EVIDENCE TIMELINE / DATA MUTATION HELPERS
# =====================================================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_timeline_event(product_id: str, event: str, details: str = ""):
    timeline = st.session_state["evidence_timeline"].setdefault(product_id, [])
    timeline.append({"timestamp": now_str(), "event": event, "details": details})


def initialize_workflow_state(df: pd.DataFrame):
    """Called whenever a fresh dataset is loaded (demo or upload)."""
    st.session_state["evidence_timeline"] = {}
    st.session_state["investigation_status"] = {}
    st.session_state["investigation_checklist"] = {}
    st.session_state["resolution_history"] = []
    st.session_state["priority_plan"] = None
    for _, row in df.iterrows():
        pid = row["Product ID"]
        add_timeline_event(
            pid,
            "Digital Inventory Loaded",
            f"System Stock: {int(row['System Stock'])}, Sales: {int(row['Sales'])}, "
            f"Physical Evidence: {int(row['Physical Evidence'])}.",
        )
        add_timeline_event(pid, "GhostStock Analysis Generated", "Initial risk analysis computed.")


def update_data_field(product_id: str, field: str, new_value):
    """Directly overwrite a field for a product row in the live dataset. Returns old value."""
    df = st.session_state["data"]
    idx = df.index[df["Product ID"] == product_id]
    if len(idx) == 0:
        return None
    old_value = df.loc[idx[0], field]
    df.loc[idx[0], field] = new_value
    st.session_state["data"] = df
    return old_value


def update_physical_evidence(product_id: str, new_value, source: str, confidence: str = None, note: str = ""):
    """Update Physical Evidence for a product and record a timeline event."""
    old_value = update_data_field(product_id, "Physical Evidence", new_value)
    if old_value is None:
        return False
    details = f"Evidence source: {source}. Previous: {int(old_value)} → New: {int(new_value)}."
    if confidence:
        details += f" Confidence: {confidence}."
    if note:
        details += f" Note: {note}"
    add_timeline_event(product_id, "Physical Evidence Updated", details)
    return True


def product_options(df: pd.DataFrame):
    """Return display labels like 'Product Name (P001)' for unambiguous selection."""
    return [f"{row['Product']} ({row['Product ID']})" for _, row in df.iterrows()]


def parse_product_id(label: str) -> str:
    return label.split("(")[-1].rstrip(")").strip()


# =====================================================================================
# CORE ANALYTICS ENGINE
# =====================================================================================

def validate_dataframe(df: pd.DataFrame):
    """Returns (is_valid: bool, messages: list[str])"""
    messages = []
    is_valid = True

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        is_valid = False
        messages.append(f"Missing required column(s): {', '.join(missing_cols)}")
        return is_valid, messages

    if df.empty:
        is_valid = False
        messages.append("The uploaded file contains no rows.")
        return is_valid, messages

    for col in NUMERIC_COLUMNS:
        non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        if non_numeric.any():
            is_valid = False
            messages.append(f"Column '{col}' contains non-numeric values.")

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            is_valid = False
            messages.append(f"Column '{col}' has missing values.")

    if is_valid:
        for col in NUMERIC_COLUMNS:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if (numeric_series < 0).any():
                is_valid = False
                messages.append(f"Column '{col}' contains negative values, which is not allowed.")

    if df["Product ID"].duplicated().any():
        messages.append("Warning: duplicate Product ID values detected. Rows were kept but review is recommended.")

    return is_valid, messages


def compute_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core GhostStock detection engine — an Inventory Intelligence and
    Explainable Risk Analysis Engine built entirely with transparent,
    rule-based Python logic (no external AI model).

    Adds: Expected Stock, Mismatch, Mismatch Percentage, Risk Score,
          Risk Level, Risk Reason, Recommendation
    """
    data = df.copy()
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    data["Expected Stock"] = data["System Stock"] - data["Sales"]
    data["Expected Stock"] = data["Expected Stock"].clip(lower=0)

    data["Mismatch"] = data["Expected Stock"] - data["Physical Evidence"]

    data["Mismatch Percentage"] = np.where(
        data["Expected Stock"] > 0,
        (data["Mismatch"] / data["Expected Stock"]) * 100,
        0,
    )
    data["Mismatch Percentage"] = data["Mismatch Percentage"].clip(lower=0)

    # ---- Weighted, normalized risk scoring model ----
    # Factor 1: Mismatch Severity (60%) — based on mismatch percentage, 0-100
    severity_score = data["Mismatch Percentage"].clip(0, 100)

    # Factor 2: Absolute Discrepancy (25%) — normalized against the largest
    # mismatch observed in the dataset so it scales relative to context
    max_abs_mismatch = max(data["Mismatch"].clip(lower=0).max(), 1)
    abs_score = (data["Mismatch"].clip(lower=0) / max_abs_mismatch) * 100

    # Factor 3: Relative Inventory Context (15%) — smaller expected-stock
    # baselines make percentage swings less statistically reliable, so a
    # mild extra weight is applied when the expected-stock base is small
    max_expected = max(data["Expected Stock"].max(), 1)
    reliability_score = (1 - (data["Expected Stock"] / max_expected)) * 100

    raw_score = (severity_score * 0.60) + (abs_score * 0.25) + (reliability_score * 0.15)

    # No real discrepancy (mismatch <= 0) => risk is forced to zero
    raw_score = np.where(data["Mismatch"] <= 0, 0, raw_score)

    data["Risk Score"] = np.clip(raw_score, 0, 100).round(1)

    def classify_risk(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        else:
            return "Low"

    data["Risk Level"] = data["Risk Score"].apply(classify_risk)

    def build_reason(row):
        if row["Mismatch"] <= 0:
            return "Physical evidence matches or exceeds expected stock. No discrepancy detected."
        return (
            f"Expected inventory after recorded sales is {int(row['Expected Stock'])} units, "
            f"while available physical evidence suggests {int(row['Physical Evidence'])} units. "
            f"This creates a discrepancy of {int(row['Mismatch'])} unit(s) "
            f"({row['Mismatch Percentage']:.1f}% of expected stock)."
        )

    data["Risk Reason"] = data.apply(build_reason, axis=1)

    def build_recommendation(row):
        if row["Risk Level"] == "High":
            return "Immediate physical verification recommended. Prioritize for inspection."
        elif row["Risk Level"] == "Medium":
            return "Physical verification recommended during next inventory cycle."
        else:
            return "Continue routine monitoring. No urgent action required."

    data["Recommendation"] = data.apply(build_recommendation, axis=1)

    return data


def compute_ml_anomaly_score(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """
    ML-based anomaly detection using Isolation Forest.

    The model looks for unusual combinations of:
    - Expected Stock
    - Physical Evidence
    - Inventory Mismatch
    - Mismatch Percentage

    This is an unsupervised model — it flags statistical outliers relative
    to the other products in the CURRENT dataset. It does not claim to be
    trained on external or historical data, and it does not report an
    accuracy percentage, since there are no ground-truth labels to score
    against in this prototype.
    """

    data = analyzed_df.copy()

    required = ["Expected Stock", "Physical Evidence", "Mismatch", "Mismatch Percentage"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        # Defensive fallback — should not happen if compute_analysis ran first,
        # but avoids a hard crash if this is ever called out of order.
        data["ML Anomaly Score"] = 0.0
        data["ML Anomaly"] = "Normal"
        return data

    features = data[required].copy()

    # Isolation Forest needs enough records to identify unusual patterns.
    # For very small datasets, fall back safely instead of fitting a
    # meaningless model.
    if len(data) < 5:
        data["ML Anomaly Score"] = 0.0
        data["ML Anomaly"] = "Normal"
        return data

    # Prevent extreme negative values from affecting the model
    features["Mismatch"] = features["Mismatch"].clip(lower=0)
    features = features.fillna(0)

    try:
        model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
        )
        model.fit(features)

        # Higher value = more anomalous
        raw_scores = -model.score_samples(features)

        # Normalize scores to 0–100
        min_score = raw_scores.min()
        max_score = raw_scores.max()

        if max_score == min_score:
            normalized_scores = np.zeros(len(raw_scores))
        else:
            normalized_scores = ((raw_scores - min_score) / (max_score - min_score)) * 100

        data["ML Anomaly Score"] = np.round(normalized_scores, 1)

        predictions = model.predict(features)
        data["ML Anomaly"] = np.where(predictions == -1, "Anomalous", "Normal")

    except Exception:
        # Never let an ML failure crash the app — fall back to a safe default
        # and let the rule-based engine continue to work on its own.
        data["ML Anomaly Score"] = 0.0
        data["ML Anomaly"] = "Normal"

    return data


def compute_zone_priority(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate discrepancy data by zone and compute a transparent Priority Score."""
    suspicious = analyzed_df[analyzed_df["Mismatch"] > 0]

    if suspicious.empty:
        return pd.DataFrame(
            columns=["Zone", "Suspicious Products", "Suspected Missing Units",
                     "Average Risk Score", "Priority Score", "Recommended Action"]
        )

    grouped = suspicious.groupby("Zone").agg(
        **{
            "Suspicious Products": ("Product ID", "count"),
            "Suspected Missing Units": ("Mismatch", "sum"),
            "Average Risk Score": ("Risk Score", "mean"),
        }
    ).reset_index()

    grouped["Priority Score"] = (
        (grouped["Suspected Missing Units"] * 2)
        + grouped["Average Risk Score"]
        + (grouped["Suspicious Products"] * 10)
    ).round(1)

    grouped["Average Risk Score"] = grouped["Average Risk Score"].round(1)

    def recommend_action(row):
        if row["Average Risk Score"] >= 70:
            return "Inspect zone immediately"
        elif row["Average Risk Score"] >= 40:
            return "Schedule inspection this week"
        else:
            return "Monitor during routine audit"

    grouped["Recommended Action"] = grouped.apply(recommend_action, axis=1)
    grouped = grouped.sort_values("Priority Score", ascending=False).reset_index(drop=True)
    grouped.insert(0, "Rank", grouped.index + 1)

    return grouped


def compute_accuracy_score(analyzed_df: pd.DataFrame) -> float:
    """Inventory Accuracy Score = 100 - average mismatch percentage (clamped 0-100)."""
    if analyzed_df.empty:
        return 100.0
    avg_mismatch_pct = analyzed_df["Mismatch Percentage"].mean()
    score = 100 - avg_mismatch_pct
    return float(np.clip(score, 0, 100))


def possible_cause_indicators(row: pd.Series) -> list:
    """Generate non-definitive, responsible-language cause indicators."""
    indicators = []
    mismatch = row["Mismatch"]
    pct = row["Mismatch Percentage"]

    if mismatch <= 0:
        return ["No discrepancy indicators — physical evidence aligns with expected stock."]

    if pct >= 60:
        indicators.append("Large discrepancy detected — physical verification strongly recommended")
    elif pct >= 30:
        indicators.append("Moderate discrepancy detected — verification recommended")
    else:
        indicators.append("Minor discrepancy detected — low urgency, monitor")

    if mismatch >= 10:
        indicators.append("Possible missed scan during receiving or sale")
    if mismatch >= 5:
        indicators.append("Possible location mismatch within warehouse")
    indicators.append("Possible inventory count error")

    return indicators


def derive_detection_method(row: pd.Series) -> str:
    """
    Combines the rule-based engine and the ML anomaly model into a single
    plain-language verdict per product. Used by the Detection Center to
    show how each product was caught.
    """
    rule_flag = row["Mismatch"] > 0 and row["Risk Level"] in ("Medium", "High")
    ml_flag = row.get("ML Anomaly", "Normal") == "Anomalous"

    if rule_flag and ml_flag:
        return "Confirmed by Rule Engine + ML"
    elif rule_flag and not ml_flag:
        return "Rule Engine Only"
    elif ml_flag and not rule_flag:
        return "ML Only (Unusual Pattern)"
    else:
        return "No Flag"


def generate_intelligence_summary(df: pd.DataFrame, zone_priority: pd.DataFrame) -> str:
    """
    Programmatically generated natural-language summary built from
    the analyzed dataset using rule-based Python logic. This is NOT
    produced by an external generative AI model.
    """
    total = len(df)
    suspicious = df[df["Mismatch"] > 0]
    high_risk = df[df["Risk Level"] == "High"]
    total_missing = int(suspicious["Mismatch"].sum())

    if suspicious.empty:
        return (
            f"GhostStock analyzed {total} product(s) and did not detect any suspected discrepancies. "
            f"Current inventory records appear consistent with available physical evidence."
        )

    most_critical = df.sort_values("Risk Score", ascending=False).iloc[0]

    summary = (
        f"GhostStock analyzed {total} products and detected {len(suspicious)} suspected discrepancies "
        f"across {df['Zone'].nunique()} zone(s), with {len(high_risk)} classified as high risk. "
        f"An estimated {total_missing} unit(s) are currently unaccounted for. "
    )

    if not zone_priority.empty:
        top_zone = zone_priority.iloc[0]["Zone"]
        summary += (
            f"Zone {top_zone} is currently the highest inspection priority because of its combined "
            f"discrepancy volume and risk score. "
        )

    summary += f"The most critical product is {most_critical['Product']} (risk score {most_critical['Risk Score']:.1f}). "

    if len(high_risk) > 0:
        summary += "Immediate physical verification is recommended."
    else:
        summary += "Physical verification is recommended during the next inventory cycle."

    return summary


def generate_priority_plan(df: pd.DataFrame, zone_priority: pd.DataFrame, suspicious: pd.DataFrame) -> list:
    """Build a dynamic, ranked step-by-step inspection plan from the analyzed dataset."""
    steps = []

    if not zone_priority.empty:
        top_zone = zone_priority.iloc[0]
        steps.append((
            "Inspect highest priority warehouse zone",
            f"Zone {top_zone['Zone']} has the highest combined discrepancy volume and risk score "
            f"(Priority Score {top_zone['Priority Score']:.1f}).",
        ))

    if not suspicious.empty:
        top_product = suspicious.iloc[0]
        steps.append((
            "Verify the highest-risk product",
            f"{top_product['Product']} (Zone {top_product['Zone']}) has a risk score of "
            f"{top_product['Risk Score']:.1f} with {int(top_product['Mismatch'])} unit(s) unaccounted for.",
        ))

        next_products = suspicious.iloc[1:4]
        if not next_products.empty:
            names = ", ".join(next_products["Product"].tolist())
            steps.append((
                "Verify the next highest-risk products",
                f"Review {names} for potential inventory discrepancies.",
            ))

    steps.append((
        "Review relevant inventory records",
        "Cross-check recorded sales and system stock entries for all flagged products.",
    ))
    steps.append((
        "Update investigation status",
        "Mark verified products as Investigating or Resolved in the Investigation workspace.",
    ))

    return steps


def determine_health_status(accuracy_score: float, high_risk_count: int, suspicious_count: int):
    """Classify overall inventory health as Healthy / Needs Attention / Critical."""
    if accuracy_score >= 85 and high_risk_count == 0:
        return "Healthy", COLOR_LOW
    elif accuracy_score >= 60 and high_risk_count <= 2:
        return "Needs Attention", COLOR_MEDIUM
    else:
        return "Critical", COLOR_HIGH


# =====================================================================================
# UI HELPER COMPONENTS
# =====================================================================================

def risk_badge_html(level: str) -> str:
    css_class = {"High": "gs-badge-high", "Medium": "gs-badge-medium", "Low": "gs-badge-low"}.get(level, "gs-badge-low")
    return f'<span class="gs-badge {css_class}">{level.upper()} RISK</span>'


def metric_card(label: str, value: str, delta: str = ""):
    delta_html = f'<div class="gs-metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="gs-metric-card">
            <div class="gs-metric-label">{label}</div>
            <div class="gs-metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="gs-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="gs-section-sub">{subtitle}</div>', unsafe_allow_html=True)


def no_data_placeholder():
    st.markdown(
        """
        <div class="gs-card" style="text-align:center; padding:50px 20px;">
            <div style="font-size:40px;">👻</div>
            <div style="font-size:18px; font-weight:700; margin-top:10px; color:#F0EEFB;">
                No data loaded yet
            </div>
            <div style="font-size:13.5px; color:#8D92C4; margin-top:6px;">
                Head to <b>Upload &amp; Data</b> to load your inventory CSV or try the demo dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeline(product_id: str):
    """Render the evidence/investigation timeline for a single product."""
    events = st.session_state["evidence_timeline"].get(product_id, [])

    if not events:
        st.info("No timeline events recorded yet for this product.")
        return

    html_items = ""

    for ev in events:
        detail_html = (
            f'<div class="gs-timeline-detail">{ev["details"]}</div>'
            if ev.get("details")
            else ""
        )

        item_html = (
            '<div class="gs-timeline-item">'
            f'<div class="gs-timeline-event">{ev["event"]}</div>'
            f'<div class="gs-timeline-meta">{ev["timestamp"]}</div>'
            f'{detail_html}'
            '</div>'
        )

        html_items += item_html

    st.markdown(
        f'<div class="gs-card">{html_items}</div>',
        unsafe_allow_html=True
    )

# =====================================================================================
# SIDEBAR NAVIGATION
# =====================================================================================

def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="gs-brand">
            <div class="gs-brand-title">👻 GhostStock</div>
            <div class="gs-brand-sub">Inventory Intelligence Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = [f"{icon}  {name}" for name, icon in NAV_PAGES.items()]
    names = list(NAV_PAGES.keys())

    current_index = names.index(st.session_state["current_page"]) if st.session_state["current_page"] in names else 0
    selected_label = st.sidebar.radio("Navigation", labels, index=current_index, label_visibility="collapsed")
    selected_name = names[labels.index(selected_label)]
    st.session_state["current_page"] = selected_name

    st.sidebar.markdown("<br>" * 3, unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="gs-sidebar-footer">Prototype • CodeBuild 1.0</div>',
        unsafe_allow_html=True,
    )

    return selected_name


# =====================================================================================
# PAGE 1 — OVERVIEW
# =====================================================================================

def page_overview():
    st.markdown(
        """
        <div class="gs-hero">
            <div class="gs-hero-title">👻 GhostStock</div>
            <div class="gs-hero-tagline">See the inventory your system thinks you have.</div>
            <div class="gs-hero-desc">
                GhostStock compares digital inventory records with available operational evidence
                to identify suspected discrepancies and prioritize physical verification. Follow the
                journey: <b>Upload → Evidence → Detection → Investigation → Resolution.</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["data"] is None:
        no_data_placeholder()
        return

    df = compute_analysis(st.session_state["data"])
    df = compute_ml_anomaly_score(df)
    suspicious = df[df["Mismatch"] > 0]
    high_risk = df[df["Risk Level"] == "High"]
    accuracy = compute_accuracy_score(df)
    zone_priority = compute_zone_priority(df)
    ml_anomalies = df[df["ML Anomaly"] == "Anomalous"]

    # ---- Metric cards ----
    cols = st.columns(6)
    with cols[0]:
        metric_card("Total Products", str(len(df)))
    with cols[1]:
        metric_card("With Discrepancies", str(len(suspicious)))
    with cols[2]:
        metric_card("High Risk Products", str(len(high_risk)))
    with cols[3]:
        metric_card("Suspected Missing Units", f"{int(suspicious['Mismatch'].sum())}")
    with cols[4]:
        metric_card("Inventory Accuracy Score", f"{accuracy:.1f}%")
    with cols[5]:
        metric_card("ML Anomalies", str(len(ml_anomalies)))

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- GhostStock Intelligence Summary ----
    summary_text = generate_intelligence_summary(df, zone_priority)
    st.markdown(
        f"""
        <div class="gs-intel-card">
            <div class="gs-intel-title">🧠 GhostStock Intelligence Summary</div>
            <div class="gs-intel-text">{summary_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Priority alert section ----
    if not zone_priority.empty:
        top_zone = zone_priority.iloc[0]
        st.markdown(
            f"""
            <div class="gs-alert-card">
                <div class="gs-alert-title">🚨 INSPECT ZONE {top_zone['Zone']} FIRST</div>
                <div class="gs-alert-desc">
                    <b>{int(top_zone['Suspicious Products'])}</b> suspicious product(s) &nbsp;•&nbsp;
                    <b>{int(top_zone['Suspected Missing Units'])}</b> suspected missing units &nbsp;•&nbsp;
                    Average risk score <b>{top_zone['Average Risk Score']:.1f}</b> &nbsp;•&nbsp;
                    Priority score <b>{top_zone['Priority Score']:.1f}</b><br><br>
                    This zone is prioritized based on discrepancy volume, number of suspicious products
                    and average risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="gs-insight">✅ No zones currently show suspected inventory discrepancies.</div>
            <br>
            """,
            unsafe_allow_html=True,
        )

    # ---- Charts ----
    section_header("Discrepancy &amp; Risk Overview")
    chart_cols = st.columns([1.3, 1])

    with chart_cols[0]:
        chart_df = df.sort_values("Mismatch", ascending=False)
        colors = [
            COLOR_HIGH if lvl == "High" else COLOR_MEDIUM if lvl == "Medium" else COLOR_LOW
            for lvl in chart_df["Risk Level"]
        ]
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=chart_df["Product"],
                    y=chart_df["Mismatch"],
                    marker_color=colors,
                    text=chart_df["Mismatch"],
                    textposition="outside",
                )
            ]
        )
        fig_bar.update_layout(
            title="Inventory Discrepancy by Product",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=50, b=10, l=10, r=10),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_cols[1]:
        risk_counts = df["Risk Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.55,
                    marker=dict(colors=[COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW]),
                )
            ]
        )
        fig_pie.update_layout(
            title="Risk Distribution",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    if not zone_priority.empty:
        fig_zone = go.Figure(
            data=[
                go.Bar(
                    x=zone_priority["Zone"],
                    y=zone_priority["Priority Score"],
                    marker_color=COLOR_BRAND_PURPLE,
                    text=zone_priority["Priority Score"],
                    textposition="outside",
                )
            ]
        )
        fig_zone.update_layout(
            title="Zone Risk Analysis (Priority Score)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_zone, use_container_width=True)

    # ---- Top priority products table ----
    section_header("Top Priority Products")
    top5 = df.sort_values("Risk Score", ascending=False).head(5).reset_index(drop=True)
    top5_display = top5[[
        "Product", "Zone", "Expected Stock", "Physical Evidence", "Mismatch",
        "Risk Score", "Risk Level", "Recommendation",
    ]].copy()
    top5_display.insert(0, "Rank", top5_display.index + 1)
    st.dataframe(top5_display, use_container_width=True, hide_index=True)

    # ---- Dynamic insight ----
    section_header("Insight Summary")
    if len(suspicious) > 0 and not zone_priority.empty:
        insight_text = (
            f"GhostStock detected {len(suspicious)} product(s) requiring physical verification. "
            f"Zone {zone_priority.iloc[0]['Zone']} should be inspected first due to the highest "
            f"combined discrepancy and risk score."
        )
    else:
        insight_text = "GhostStock did not detect any suspected discrepancies in the current dataset."
    st.markdown(f'<div class="gs-insight">💡 {insight_text}</div>', unsafe_allow_html=True)


# =====================================================================================
# PAGE 2 — UPLOAD & DATA
# =====================================================================================

def page_upload():
    section_header(
        "Upload &amp; Data",
        "Step 1 of the GhostStock workflow: load your inventory data to begin analysis.",
    )

    st.markdown(
        """
        <div class="gs-card">
            Required columns: <b>Product ID, Product, Zone, System Stock, Sales, Physical Evidence</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Sample CSV",
            data=generate_sample_csv_bytes(),
            file_name="ghoststock_sample_inventory.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button("⚡ Use Demo Data", use_container_width=True):
            demo_df = generate_demo_data()
            st.session_state["data"] = demo_df
            st.session_state["data_source"] = "Demo Data"
            initialize_workflow_state(demo_df)
            st.success("Demo data loaded successfully. GhostStock analysis is ready across all pages.")

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your inventory CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the uploaded file. Please make sure it is a valid CSV. ({e})")
            new_df = None

        if new_df is not None:
            is_valid, messages = validate_dataframe(new_df)
            if not is_valid:
                st.error("The uploaded file has issues that need to be fixed before analysis:")
                for m in messages:
                    st.markdown(f"- {m}")
            else:
                if messages:
                    for m in messages:
                        st.warning(m)
                st.session_state["data"] = new_df
                st.session_state["data_source"] = uploaded_file.name
                initialize_workflow_state(new_df)
                st.success(f"'{uploaded_file.name}' uploaded and validated successfully. Next: add physical evidence.")

    # ---- Data preview ----
    if st.session_state["data"] is not None:
        df = st.session_state["data"]
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Data Preview", f"Source: {st.session_state.get('data_source', 'Unknown')}")

        cols = st.columns(3)
        with cols[0]:
            metric_card("Number of Products", str(len(df)))
        with cols[1]:
            metric_card("Number of Zones", str(df["Zone"].nunique()))
        with cols[2]:
            total_stock = pd.to_numeric(df["System Stock"], errors="coerce").fillna(0).sum()
            metric_card("Total System Stock", f"{int(total_stock)}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="gs-insight">➡️ Next step: go to <b>Evidence Input</b> to add or update physical '
            'evidence for your products.</div>',
            unsafe_allow_html=True,
        )
    else:
        no_data_placeholder()


# =====================================================================================
# PAGE 3 — EVIDENCE INPUT
# =====================================================================================

def page_evidence_input():
    section_header(
        "Evidence Input",
        "Step 2 of the GhostStock workflow: add or update the physical evidence used to detect discrepancies.",
    )

    if st.session_state["data"] is None:
        no_data_placeholder()
        return

    df = st.session_state["data"]

    tab1, tab2, tab3 = st.tabs(["📋 Manual Physical Count", "📡 Barcode / Scanner Simulation", "🖼️ Evidence Upload"])

    # ---------------------------------------------------------------------------
    # TAB 1 — MANUAL PHYSICAL COUNT
    # ---------------------------------------------------------------------------
    with tab1:
        st.markdown(
            '<div class="gs-card">Enter a physical count taken directly by a warehouse team member.</div>',
            unsafe_allow_html=True,
        )
        options = product_options(df)
        selection = st.selectbox("Select a product", options, key="manual_product_select")
        pid = parse_product_id(selection)
        row = df[df["Product ID"] == pid].iloc[0]

        expected_stock = max(int(row["System Stock"]) - int(row["Sales"]), 0)

        mcols = st.columns(3)
        with mcols[0]:
            metric_card("System Stock", str(int(row["System Stock"])))
        with mcols[1]:
            metric_card("Recorded Sales", str(int(row["Sales"])))
        with mcols[2]:
            metric_card("Expected Stock", str(expected_stock))

        counted_qty = st.number_input(
            "Physically counted quantity",
            min_value=0,
            value=int(row["Physical Evidence"]),
            step=1,
            key="manual_counted_qty",
        )

        if st.button("✅ Update Physical Evidence", key="manual_update_btn"):
            update_physical_evidence(pid, counted_qty, source="Manual Physical Count")
            st.success(f"Physical Evidence for '{row['Product']}' updated to {counted_qty} units.")
            st.rerun()

    # ---------------------------------------------------------------------------
    # TAB 2 — BARCODE / SCANNER SIMULATION
    # ---------------------------------------------------------------------------
    with tab2:
        st.markdown('<div class="gs-prototype-tag">Prototype scanner simulation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="gs-card">This simulates scanning a product barcode. No real barcode hardware '
            'is connected — the Product ID is entered manually to represent a scan event.</div>',
            unsafe_allow_html=True,
        )

        scanned_id = st.text_input("Scan / enter Product ID", key="scan_product_id_input")

        if scanned_id:
            match = df[df["Product ID"].str.strip().str.upper() == scanned_id.strip().upper()]
            if match.empty:
                st.warning(f"Product ID '{scanned_id}' was not found in the current dataset.")
            else:
                row = match.iloc[0]
                pid = row["Product ID"]
                st.success(f"Product found: {row['Product']} (Zone {row['Zone']})")

                scols = st.columns(3)
                with scols[0]:
                    metric_card("System Stock", str(int(row["System Stock"])))
                with scols[1]:
                    metric_card("Recorded Sales", str(int(row["Sales"])))
                with scols[2]:
                    metric_card("Current Physical Evidence", str(int(row["Physical Evidence"])))

                scanned_qty = st.number_input(
                    "Detected physical quantity",
                    min_value=0,
                    value=int(row["Physical Evidence"]),
                    step=1,
                    key="scan_detected_qty",
                )

                if st.button("📡 Record Scan Evidence", key="scan_record_btn"):
                    update_physical_evidence(pid, scanned_qty, source="Barcode / Scanner Simulation")
                    st.success(f"Scan evidence recorded for '{row['Product']}': {scanned_qty} units.")
                    st.rerun()

    # ---------------------------------------------------------------------------
    # TAB 3 — EVIDENCE UPLOAD
    # ---------------------------------------------------------------------------
    with tab3:
        st.markdown('<div class="gs-prototype-tag">Prototype Evidence Upload — manual verification required</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="gs-card">Upload a photo as supporting evidence (e.g. a shelf photo). '
            'GhostStock does <b>not</b> perform automatic image recognition in this prototype — '
            'the quantity must be entered manually after reviewing the image.</div>',
            unsafe_allow_html=True,
        )

        uploaded_image = st.file_uploader("Upload evidence image", type=["jpg", "jpeg", "png"], key="evidence_image_upload")
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded evidence (manual review only)", width=280)

        options = product_options(df)
        selection = st.selectbox("This evidence belongs to", options, key="upload_product_select")
        pid = parse_product_id(selection)
        row = df[df["Product ID"] == pid].iloc[0]

        verified_qty = st.number_input(
            "Verified quantity from image",
            min_value=0,
            value=int(row["Physical Evidence"]),
            step=1,
            key="upload_verified_qty",
        )
        confidence = st.selectbox("Evidence confidence", CONFIDENCE_LEVELS, key="upload_confidence_select")

        if st.button("🖼️ Confirm Evidence", key="upload_confirm_btn"):
            update_physical_evidence(pid, verified_qty, source="Evidence Upload (Image)", confidence=confidence)
            st.success(f"Evidence confirmed for '{row['Product']}': {verified_qty} units (confidence: {confidence}).")
            st.rerun()

    # ---------------------------------------------------------------------------
    # EVIDENCE TIMELINE VIEWER
    # ---------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Evidence Timeline", "View the full history of evidence and analysis events for a product.")
    options = product_options(df)
    timeline_selection = st.selectbox("Select a product to view its timeline", options, key="timeline_product_select")
    timeline_pid = parse_product_id(timeline_selection)
    render_timeline(timeline_pid)


# =====================================================================================
# PAGE 4 — DETECTION CENTER
# =====================================================================================

def page_detection_center():
    section_header(
        "Detection Center",
        "Step 3 of the GhostStock workflow: review suspected inventory discrepancies.",
    )

    if st.session_state["data"] is None:
        no_data_placeholder()
        return

    df = compute_analysis(st.session_state["data"])
    df = compute_ml_anomaly_score(df)
    df["Detection Method"] = df.apply(derive_detection_method, axis=1)

    # ---- Filters ----
    st.markdown('<div class="gs-card">', unsafe_allow_html=True)
    f1, f2, f3, f4, f5 = st.columns([1.3, 0.9, 0.9, 1.1, 1.1])
    with f1:
        search_term = st.text_input("Search product", "")
    with f2:
        zone_options = ["All"] + sorted(df["Zone"].unique().tolist())
        zone_filter = st.selectbox("Zone", zone_options)
    with f3:
        risk_options = ["All", "High", "Medium", "Low"]
        risk_filter = st.selectbox("Risk level", risk_options)
    with f4:
        min_risk = st.slider("Minimum risk score", 0, 100, 0)
    with f5:
        method_options = ["All", "Rule Engine", "ML", "Both"]
        method_filter = st.selectbox("Detection method", method_options)
    st.markdown('</div>', unsafe_allow_html=True)

    filtered = df.copy()
    if search_term:
        filtered = filtered[filtered["Product"].str.contains(search_term, case=False, na=False)]
    if zone_filter != "All":
        filtered = filtered[filtered["Zone"] == zone_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["Risk Level"] == risk_filter]
    filtered = filtered[filtered["Risk Score"] >= min_risk]

    if method_filter == "Rule Engine":
        filtered = filtered[filtered["Detection Method"].isin(
            ["Rule Engine Only", "Confirmed by Rule Engine + ML"]
        )]
    elif method_filter == "ML":
        filtered = filtered[filtered["Detection Method"].isin(
            ["ML Only (Unusual Pattern)", "Confirmed by Rule Engine + ML"]
        )]
    elif method_filter == "Both":
        filtered = filtered[filtered["Detection Method"] == "Confirmed by Rule Engine + ML"]

    filtered = filtered.sort_values("Risk Score", ascending=False)

    # ---- Detection method summary strip ----
    method_counts = df["Detection Method"].value_counts()
    summary_cols = st.columns(4)
    with summary_cols[0]:
        metric_card("Confirmed (Both Methods)", str(int(method_counts.get("Confirmed by Rule Engine + ML", 0))))
    with summary_cols[1]:
        metric_card("Rule Engine Only", str(int(method_counts.get("Rule Engine Only", 0))))
    with summary_cols[2]:
        metric_card("ML Only", str(int(method_counts.get("ML Only (Unusual Pattern)", 0))))
    with summary_cols[3]:
        metric_card("No Flag", str(int(method_counts.get("No Flag", 0))))

    st.markdown("<br>", unsafe_allow_html=True)

    section_header("Results")
    if filtered.empty:
        st.info("No products match the current filters. Try adjusting your search criteria.")
    else:
        display_cols = [
            "Product ID", "Product", "Zone", "System Stock", "Sales", "Expected Stock",
            "Physical Evidence", "Mismatch", "Mismatch Percentage", "Risk Score",
            "Risk Level", "ML Anomaly Score", "ML Anomaly", "Detection Method",
            "Recommendation",
        ]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    # ---- Product detail explorer ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Product Detail Explorer")

    if df.empty:
        st.info("No products available to explore.")
        return

    product_choice = st.selectbox("Select a product", df["Product"].tolist(), key="detection_product_select")
    row = df[df["Product"] == product_choice].iloc[0]

    cols = st.columns(4)
    with cols[0]:
        metric_card("System Stock", str(int(row["System Stock"])))
        metric_card("Sales", str(int(row["Sales"])))
    with cols[1]:
        metric_card("Expected Stock", str(int(row["Expected Stock"])))
        metric_card("Physical Evidence", str(int(row["Physical Evidence"])))
    with cols[2]:
        metric_card("Suspected Missing", str(int(max(row["Mismatch"], 0))))
        metric_card("Risk Score", f"{row['Risk Score']:.1f}")
    with cols[3]:
        st.markdown(
            f"""
            <div class="gs-metric-card">
                <div class="gs-metric-label">Risk Level</div>
                <div style="margin-top:10px;">{risk_badge_html(row['Risk Level'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Why was this product flagged?", "Two independent detection methods, shown side by side.")

    detail_cols = st.columns(2)

    with detail_cols[0]:
        st.markdown(
            f"""
            <div class="gs-card">
                <div style="font-size:12.5px; font-weight:800; color:{COLOR_BRAND_CYAN}; letter-spacing:0.05em;">
                    RULE ENGINE
                </div>
                <p style="color:#D6D9F5; margin-top:8px;">{row['Risk Reason']}</p>
                <p style="margin-top:10px;"><b>Risk Level:</b> {risk_badge_html(row['Risk Level'])}</p>
                <p><b>Risk Score:</b> {row['Risk Score']:.1f} / 100</p>
                <p><b>Recommended Action:</b> {row['Recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_cols[1]:
        ml_verdict = row["ML Anomaly"]
        ml_score = row["ML Anomaly Score"]
        if ml_verdict == "Anomalous":
            ml_text = (
                "This product's combination of expected stock, physical evidence and mismatch size "
                "is statistically unusual compared with the other products in the current dataset. "
                "This is a signal worth a second look, not a confirmed finding on its own."
            )
            ml_badge_class = "gs-badge-high"
        else:
            ml_text = (
                "This product's stock pattern falls within the normal range compared with the other "
                "products in the current dataset. No unusual pattern was detected."
            )
            ml_badge_class = "gs-badge-low"

        st.markdown(
            f"""
            <div class="gs-card">
                <div style="font-size:12.5px; font-weight:800; color:{COLOR_BRAND_PURPLE}; letter-spacing:0.05em;">
                    ML ANOMALY DETECTION
                </div>
                <p style="color:#D6D9F5; margin-top:8px;">{ml_text}</p>
                <p style="margin-top:10px;"><b>Verdict:</b>
                    <span class="gs-badge {ml_badge_class}">{ml_verdict.upper()}</span>
                </p>
                <p><b>Anomaly Score:</b> {ml_score:.1f} / 100</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Combined verdict ----
    method = row["Detection Method"]
    if method == "Confirmed by Rule Engine + ML":
        combined_text = (
            "Both the rule-based engine and the ML anomaly model flag this product. This agreement "
            "between two independent methods is a stronger signal — prioritize this product for inspection."
        )
        combined_color = COLOR_HIGH
    elif method == "Rule Engine Only":
        combined_text = (
            "The rule-based engine flags this product based on mismatch size and percentage. The ML "
            "model did not find it statistically unusual relative to the rest of the dataset."
        )
        combined_color = COLOR_MEDIUM
    elif method == "ML Only (Unusual Pattern)":
        combined_text = (
            "The ML model finds this product's pattern unusual, even though the rule engine does not "
            "classify it as high or medium risk. Worth a second look, but not a confirmed issue on its own."
        )
        combined_color = COLOR_MEDIUM
    else:
        combined_text = "Neither method currently flags this product as suspicious."
        combined_color = COLOR_LOW

    st.markdown(
        f"""
        <div class="gs-card" style="border-left: 4px solid {combined_color};">
            <div style="font-size:12.5px; font-weight:800; color:{combined_color}; letter-spacing:0.05em;">
                COMBINED VERDICT — {method.upper()}
            </div>
            <p style="color:#D6D9F5; margin-top:8px;">{combined_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Evidence Breakdown")
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Expected Stock", "Physical Evidence", "Mismatch"],
                y=[row["Expected Stock"], row["Physical Evidence"], max(row["Mismatch"], 0)],
                marker_color=[COLOR_BRAND_CYAN, COLOR_BRAND_PURPLE, COLOR_HIGH],
                text=[int(row["Expected Stock"]), int(row["Physical Evidence"]), int(max(row["Mismatch"], 0))],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="gs-insight">➡️ Next step: go to <b>Investigation</b> to track and resolve this issue.</div>',
        unsafe_allow_html=True,
    )


# =====================================================================================
# PAGE 5 — INVESTIGATION
# =====================================================================================

def page_investigation():
    section_header(
        "Investigation Workspace",
        "Step 4 of the GhostStock workflow: investigate a suspicious product and resolve the discrepancy.",
    )

    if st.session_state["data"] is None:
        no_data_placeholder()
        return

    df = compute_analysis(st.session_state["data"])
    suspicious = df[df["Mismatch"] > 0].sort_values("Risk Score", ascending=False)

    if suspicious.empty:
        st.markdown(
            '<div class="gs-insight">✅ No suspicious products currently require investigation.</div>',
            unsafe_allow_html=True,
        )
    else:
        options = product_options(suspicious)
        selection = st.selectbox("Select a suspicious product to investigate", options, key="investigation_product_select")
        product_id = parse_product_id(selection)
        row = suspicious[suspicious["Product ID"] == product_id].iloc[0]

        # ---- Investigation summary ----
        section_header("Investigation Summary")
        st.markdown(
            f"""
            <div class="gs-card">
                <table style="width:100%; color:#D6D9F5; font-size:14px;">
                    <tr><td style="padding:6px 0;"><b>Product:</b></td><td>{row['Product']}</td></tr>
                    <tr><td style="padding:6px 0;"><b>Zone:</b></td><td>{row['Zone']}</td></tr>
                    <tr><td style="padding:6px 0;"><b>Expected Stock:</b></td><td>{int(row['Expected Stock'])}</td></tr>
                    <tr><td style="padding:6px 0;"><b>Physical Evidence:</b></td><td>{int(row['Physical Evidence'])}</td></tr>
                    <tr><td style="padding:6px 0;"><b>Discrepancy:</b></td><td>{int(row['Mismatch'])}</td></tr>
                    <tr><td style="padding:6px 0;"><b>Risk:</b></td><td>{risk_badge_html(row['Risk Level'])}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Possible cause indicators ----
        section_header("Possible Cause Indicators", "These are potential indicators only — not confirmed findings.")
        indicators = possible_cause_indicators(row)
        indicator_html = "".join([f"<li style='margin-bottom:6px;'>{ind}</li>" for ind in indicators])
        st.markdown(f'<div class="gs-card"><ul>{indicator_html}</ul></div>', unsafe_allow_html=True)

        # ---- Investigation checklist ----
        section_header("Investigation Checklist")
        if product_id not in st.session_state["investigation_checklist"]:
            st.session_state["investigation_checklist"][product_id] = {item: False for item in CHECKLIST_ITEMS}

        checklist_state = st.session_state["investigation_checklist"][product_id]
        for item in CHECKLIST_ITEMS:
            checklist_state[item] = st.checkbox(item, value=checklist_state[item], key=f"chk_{product_id}_{item}")

        completed = sum(checklist_state.values())
        total = len(CHECKLIST_ITEMS)
        st.progress(completed / total)
        st.caption(f"{completed} of {total} checklist items completed.")

        # ---- Investigation status ----
        section_header("Investigation Status")
        if product_id not in st.session_state["investigation_status"]:
            st.session_state["investigation_status"][product_id] = "Not Started"

        status = st.selectbox(
            "Current status",
            INVESTIGATION_STATUSES,
            index=INVESTIGATION_STATUSES.index(st.session_state["investigation_status"][product_id]),
            key=f"status_{product_id}",
        )
        if status != st.session_state["investigation_status"][product_id]:
            st.session_state["investigation_status"][product_id] = status
            if status == "Investigating":
                add_timeline_event(product_id, "Investigation Started", "Status changed to Investigating.")
        st.markdown(f'<div class="gs-insight">Status for <b>{row["Product"]}</b>: <b>{status}</b></div>', unsafe_allow_html=True)

        # ---- Resolution Decision ----
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Resolution Decision", "Choose how this suspected discrepancy should be resolved.")

        resolution_type = st.radio("Resolution type", RESOLUTION_TYPES, key=f"res_type_{product_id}")

        note = ""
        verified_qty = None
        correction_target = None
        corrected_value = None

        if resolution_type == "Confirm Discrepancy":
            st.caption("The discrepancy will remain confirmed. You may add an optional note.")
            note = st.text_area("Optional resolution note", key=f"note_confirm_{product_id}")

        elif resolution_type == "Physical Count Corrected":
            st.caption("Enter the corrected, verified physical quantity for this product.")
            verified_qty = st.number_input(
                "Verified Physical Quantity",
                min_value=0,
                value=int(row["Physical Evidence"]),
                step=1,
                key=f"verified_qty_{product_id}",
            )
            note = st.text_area("Optional resolution note", key=f"note_corrected_{product_id}")

        elif resolution_type == "False Alert / Data Error":
            st.caption("Select which field was incorrect and provide the corrected value.")
            correction_target = st.radio(
                "Field to correct",
                ["Recorded Sales", "Physical Evidence"],
                key=f"correction_target_{product_id}",
            )
            field_map = {"Recorded Sales": "Sales", "Physical Evidence": "Physical Evidence"}
            current_val = int(row[field_map[correction_target]])
            corrected_value = st.number_input(
                f"Corrected {correction_target}",
                min_value=0,
                value=current_val,
                step=1,
                key=f"corrected_value_{product_id}",
            )
            note = st.text_area("Optional resolution note", key=f"note_false_alert_{product_id}")

        if st.button("✅ Resolve Investigation", key=f"resolve_btn_{product_id}"):
            previous_mismatch = row["Mismatch"]
            previous_physical_evidence = row["Physical Evidence"]
            updated_physical_evidence = previous_physical_evidence

            if resolution_type == "Confirm Discrepancy":
                add_timeline_event(
                    product_id, "Resolution Completed",
                    f"Discrepancy confirmed. Note: {note or 'None'}",
                )
            elif resolution_type == "Physical Count Corrected":
                update_physical_evidence(product_id, verified_qty, source="Investigation Resolution", note=note)
                updated_physical_evidence = verified_qty
                add_timeline_event(
                    product_id, "Resolution Completed",
                    f"Physical count corrected to {verified_qty}. Note: {note or 'None'}",
                )
            elif resolution_type == "False Alert / Data Error":
                field_map = {"Recorded Sales": "Sales", "Physical Evidence": "Physical Evidence"}
                target_field = field_map[correction_target]
                old_val = update_data_field(product_id, target_field, corrected_value)
                if target_field == "Physical Evidence":
                    updated_physical_evidence = corrected_value
                add_timeline_event(
                    product_id, "Data Corrected",
                    f"{correction_target} corrected from {int(old_val)} to {int(corrected_value)}.",
                )
                add_timeline_event(
                    product_id, "Resolution Completed",
                    f"Marked as false alert / data error. Note: {note or 'None'}",
                )

            st.session_state["investigation_status"][product_id] = "Resolved"

            st.session_state["resolution_history"].append({
                "Timestamp": now_str(),
                "Product ID": product_id,
                "Product": row["Product"],
                "Zone": row["Zone"],
                "Previous Discrepancy": int(previous_mismatch),
                "Resolution Type": resolution_type,
                "Previous Physical Evidence": int(previous_physical_evidence),
                "Updated Physical Evidence": int(updated_physical_evidence),
                "Resolution Note": note or "",
                "Final Status": "Resolved",
            })

            st.success(f"Investigation for '{row['Product']}' marked as Resolved. Dashboard and alerts will refresh.")
            st.rerun()

        # ---- Timeline for this product ----
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Evidence &amp; Investigation Timeline")
        render_timeline(product_id)

    # ---- Resolution history table (all products) ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Resolution History", "Full record of resolved investigations across all products.")
    if st.session_state["resolution_history"]:
        history_df = pd.DataFrame(st.session_state["resolution_history"])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No investigations have been resolved yet.")


# =====================================================================================
# PAGE 6 — RISK ALERTS
# =====================================================================================

def page_risk_alerts():
    section_header("Risk Alerts", "Live alert center — automatically refreshes after evidence and resolution updates.")

    if st.session_state["data"] is None:
        no_data_placeholder()
        return

    df = compute_analysis(st.session_state["data"])
    high_risk = df[df["Risk Level"] == "High"].sort_values("Risk Score", ascending=False)
    medium_risk = df[df["Risk Level"] == "Medium"].sort_values("Risk Score", ascending=False)

    if high_risk.empty:
        st.markdown(
            """
            <div class="gs-insight">
                ✅ No high-risk discrepancies currently detected.<br>
                Continue monitoring medium-risk products.
            </div>
            <br>
            """,
            unsafe_allow_html=True,
        )
    else:
        section_header("🚨 High Risk Alerts")
        for _, row in high_risk.iterrows():
            status = st.session_state["investigation_status"].get(row["Product ID"], "Not Started")
            st.markdown(
                f"""
                <div class="gs-card" style="border-left: 4px solid {COLOR_HIGH};">
                    <div style="font-size:15px; font-weight:700; color:{COLOR_HIGH};">🚨 HIGH RISK — {row['Product']}</div>
                    <div style="font-size:13px; color:#A9AEDA; margin-top:4px;">Zone {row['Zone']} • Investigation status: {status}</div>
                    <table style="width:100%; margin-top:10px; font-size:13.5px; color:#D6D9F5;">
                        <tr><td><b>Expected Stock:</b></td><td>{int(row['Expected Stock'])}</td>
                            <td><b>Physical Evidence:</b></td><td>{int(row['Physical Evidence'])}</td></tr>
                        <tr><td><b>Suspected Discrepancy:</b></td><td>{int(row['Mismatch'])}</td>
                            <td><b>Risk Score:</b></td><td>{row['Risk Score']:.1f}</td></tr>
                    </table>
                    <p style="margin-top:10px; font-size:13px; color:#C9CDF0;"><b>Why Flagged:</b> {row['Risk Reason']}</p>
                    <p style="font-size:13px; color:#C9CDF0;"><b>Recommended Action:</b> {row['Recommendation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not medium_risk.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("⚠️ Medium Risk Products")
        for _, row in medium_risk.iterrows():
            status = st.session_state["investigation_status"].get(row["Product ID"], "Not Started")
            st.markdown(
                f"""
                <div class="gs-card" style="border-left: 4px solid {COLOR_MEDIUM}; opacity:0.92;">
                    <div style="font-size:14px; font-weight:700; color:{COLOR_MEDIUM};">⚠️ {row['Product']} — Zone {row['Zone']}</div>
                    <div style="font-size:12px; color:#8D92C4;">Investigation status: {status}</div>
                    <p style="margin-top:8px; font-size:13px; color:#C9CDF0;">{row['Risk Reason']}</p>
                    <p style="font-size:13px; color:#8D92C4;"><b>Recommended Action:</b> {row['Recommendation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================================================
# PAGE 7 — ACTION PLAN
# =====================================================================================

def page_action_plan():
    section_header("Action Plan", "Turn GhostStock's analysis into a concrete, ranked inspection plan.")

    if st.session_state["data"] is None:
        no_data_placeholder()
        return

    df = compute_analysis(st.session_state["data"])
    zone_priority = compute_zone_priority(df)
    suspicious = df[df["Mismatch"] > 0].sort_values("Risk Score", ascending=False)

    if st.button("⚡ Generate Today's Priority Plan", use_container_width=False):
        st.session_state["priority_plan"] = generate_priority_plan(df, zone_priority, suspicious)

    section_header("Today's Recommended Inspection Plan")
    plan = st.session_state.get("priority_plan")

    if plan is None:
        st.info("Click 'Generate Today's Priority Plan' to build a ranked plan from the current analysis.")
    elif not plan:
        st.markdown(
            '<div class="gs-insight">✅ No inspection actions required today. Inventory looks healthy.</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, (title, reason) in enumerate(plan, start=1):
            st.markdown(
                f"""
                <div class="gs-card">
                    <div style="font-size:13px; font-weight:800; color:{COLOR_BRAND_CYAN}; letter-spacing:0.05em;">STEP {i}</div>
                    <div style="font-size:15px; font-weight:700; color:#F0EEFB; margin-top:4px;">{title}</div>
                    <div style="font-size:13px; color:#A9AEDA; margin-top:4px;">{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Zone priority table ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Zone Priority Table")
    if zone_priority.empty:
        st.info("No zones currently require prioritization.")
    else:
        st.dataframe(zone_priority, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="gs-insight">
            <b>Priority Score formula:</b> (Suspected Missing Units × 2) + Average Risk Score +
            (Suspicious Products × 10). This transparent formula rewards zones with more missing
            units, higher average risk, and a greater number of flagged products.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================================
# PAGE 8 — INVENTORY HEALTH
# =====================================================================================

def page_inventory_health():
    section_header("Inventory Health", "A single view of overall inventory health and reliability.")

    if st.session_state["data"] is None:
        no_data_placeholder()
        return

    df = compute_analysis(st.session_state["data"])
    suspicious = df[df["Mismatch"] > 0]
    high_risk = df[df["Risk Level"] == "High"]
    accuracy = compute_accuracy_score(df)
    zone_priority = compute_zone_priority(df)

    status_label, status_color = determine_health_status(accuracy, len(high_risk), len(suspicious))

    st.markdown(
        f"""
        <div class="gs-card" style="text-align:center;">
            <div style="font-size:13px; color:#8D92C4; text-transform:uppercase; letter-spacing:0.05em;">
                Overall Inventory Health
            </div>
            <div style="margin-top:10px;">
                <span class="gs-health-badge" style="background: rgba(0,0,0,0); border: 2px solid {status_color}; color:{status_color};">
                    {status_label}
                </span>
            </div>
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )

    # ---- Accuracy + discrepancy summary ----
    cols = st.columns(4)
    with cols[0]:
        metric_card("Inventory Accuracy Score", f"{accuracy:.1f}%")
    with cols[1]:
        metric_card("Suspicious Products", str(len(suspicious)))
    with cols[2]:
        metric_card("High Risk Products", str(len(high_risk)))
    with cols[3]:
        metric_card("Suspected Missing Units", f"{int(suspicious['Mismatch'].sum())}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Risk distribution + zone comparison ----
    chart_cols = st.columns(2)
    with chart_cols[0]:
        risk_counts = df["Risk Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.55,
                    marker=dict(colors=[COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW]),
                )
            ]
        )
        fig_pie.update_layout(
            title="High / Medium / Low Risk Distribution",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_cols[1]:
        if zone_priority.empty:
            st.info("No zone comparison available — no suspicious products detected.")
        else:
            fig_zone = go.Figure(
                data=[
                    go.Bar(
                        x=zone_priority["Zone"],
                        y=zone_priority["Average Risk Score"],
                        marker_color=COLOR_BRAND_CYAN,
                        text=zone_priority["Average Risk Score"],
                        textposition="outside",
                    )
                ]
            )
            fig_zone.update_layout(
                title="Zone Comparison — Average Risk Score",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_zone, use_container_width=True)

    # ---- Demo audit trend simulation ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Audit Trend")
    st.markdown('<div class="gs-prototype-tag">Demo audit trend simulation — for illustrative purposes only, not real historical data</div>', unsafe_allow_html=True)

    rng = np.random.default_rng(42)
    cycles = ["Cycle -5", "Cycle -4", "Cycle -3", "Cycle -2", "Cycle -1", "Current"]
    noise = rng.normal(loc=0, scale=4, size=len(cycles) - 1)
    simulated_scores = [float(np.clip(accuracy + n, 40, 100)) for n in noise] + [accuracy]

    fig_trend = go.Figure(
        data=[
            go.Scatter(
                x=cycles,
                y=simulated_scores,
                mode="lines+markers",
                line=dict(color=COLOR_BRAND_PURPLE, width=3),
                marker=dict(size=8, color=COLOR_BRAND_CYAN),
            )
        ]
    )
    fig_trend.update_layout(
        title="Simulated Inventory Accuracy Trend (Demo Only)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        yaxis_range=[0, 100],
        margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# =====================================================================================
# PAGE 9 — ABOUT GHOSTSTOCK
# =====================================================================================

def page_about():
    section_header("About GhostStock")

    st.markdown(
        """
        <div class="gs-card">
            <div style="font-size:16px; font-weight:700; color:#F0EEFB;">The Problem</div>
            <p style="color:#C9CDF0; margin-top:6px;">
                Digital inventory records do not always reflect actual available stock. Items may be
                missing, misplaced, damaged, unscanned, moved without being recorded, or affected by
                counting errors — creating a gap between what the system says and what is physically true.
            </p>
        </div>

        <div class="gs-card">
            <div style="font-size:16px; font-weight:700; color:#F0EEFB;">How GhostStock Works</div>
            <ol style="color:#C9CDF0; margin-top:6px; line-height:1.8;">
                <li>Reads digital inventory data</li>
                <li>Collects physical evidence (manual count, scan simulation, or evidence upload)</li>
                <li>Calculates expected stock after sales</li>
                <li>Compares expected stock with physical evidence</li>
                <li>Detects discrepancies</li>
                <li>Calculates risk using a transparent, rule-based scoring engine</li>
                <li>Cross-checks each product with an unsupervised ML anomaly model</li>
                <li>Explains why an item was flagged, by which method</li>
                <li>Prioritizes inspection by warehouse zone</li>
                <li>Guides the user through investigation and resolution</li>
            </ol>
        </div>

        <div class="gs-card">
            <div style="font-size:16px; font-weight:700; color:#F0EEFB;">What GhostStock Is</div>
            <p style="color:#C9CDF0; margin-top:6px;">
                GhostStock is an <b>Inventory Intelligence and Explainable Risk Analysis Engine</b>.
                Risk scoring and summaries are generated using transparent, rule-based Python logic,
                cross-checked against an unsupervised Isolation Forest model that flags statistical
                outliers within the current dataset. Neither component is an external generative AI
                model, and the ML model is not claimed to be trained on external or historical data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Key Differentiators")
    diffs = [
        ("🔍", "Evidence-Based Detection", "Flags discrepancies using system stock, sales and physical evidence — not guesswork."),
        ("🧠", "Explainable Risk Scoring", "Every risk score comes with a clear, human-readable reason."),
        ("🤖", "ML Cross-Check", "An Isolation Forest model flags statistically unusual products, shown alongside the rule engine's verdict."),
        ("📍", "Zone-Level Prioritization", "Ranks warehouse zones so teams know where to look first."),
        ("🎯", "Investigation Workflow", "Built-in checklists, statuses, and a full resolution workflow."),
    ]
    cols = st.columns(len(diffs))
    for col, (icon, title, desc) in zip(cols, diffs):
        with col:
            st.markdown(
                f"""
                <div class="gs-card" style="height:100%;">
                    <div style="font-size:26px;">{icon}</div>
                    <div style="font-size:14px; font-weight:700; color:#F0EEFB; margin-top:8px;">{title}</div>
                    <div style="font-size:12.5px; color:#8D92C4; margin-top:6px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Prototype Disclosures")
    st.markdown(
        """
        <div class="gs-insight">
            GhostStock identifies suspected discrepancies based on uploaded data and available evidence.
            Physical verification is required before making final conclusions. The Barcode / Scanner
            Simulation and Evidence Upload features are prototype simulations — no real barcode hardware
            or automatic image-recognition AI is connected. The ML anomaly model is an unsupervised
            Isolation Forest that compares products only within the currently loaded dataset; it requires
            at least 5 rows to run, and it is not trained on any external or historical data. The audit
            trend chart on the Inventory Health page uses simulated demo data and does not represent
            real historical warehouse records.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================================
# MAIN ROUTER
# =====================================================================================

def main():
    selected_page = render_sidebar()

    if selected_page == "Overview":
        page_overview()
    elif selected_page == "Upload & Data":
        page_upload()
    elif selected_page == "Evidence Input":
        page_evidence_input()
    elif selected_page == "Detection Center":
        page_detection_center()
    elif selected_page == "Investigation":
        page_investigation()
    elif selected_page == "Risk Alerts":
        page_risk_alerts()
    elif selected_page == "Action Plan":
        page_action_plan()
    elif selected_page == "Inventory Health":
        page_inventory_health()
    elif selected_page == "About GhostStock":
        page_about()


if __name__ == "__main__":
    main()