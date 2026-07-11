"""
MedCore — Patient Experience Dashboard + Live Feedback Triage
Run with:   streamlit run medcore_dashboard.py
Requires:   pip install streamlit pandas numpy matplotlib scikit-learn plotly "jinja2>=3.1"
            (jinja2 powers the styled comments table — older pinned versions
            raise "the .style accessor requires jinja2"; upgrade if you see that.
            plotly powers the interactive revenue-impact chart.)
Data:       patient_experience_master.csv in the same folder

The risk and no-show models below are trained automatically, in-memory, the
moment this script runs (cached for the session so it only happens once).
The Live Feedback Triage tab is rule-based (dictionary lookup), not a
trained model — see the THEME_MAP/ACTIONS tables below. Just run the
command above.

NOTE on "$" in markdown text: Streamlit's markdown renderer treats a single
"$" as the start/end of inline LaTeX math. Any text passed to st.markdown /
st.caption (including the raw HTML built by kpi_card_html, since that still
goes through the markdown renderer) that contains two or more literal "$"
characters will have everything between them swallowed into "math mode" and
rendered without spaces. Every dollar amount below is written as "\\$" for
this reason -- don't remove the backslash when editing these strings.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "patient_experience_master.csv"
VALID_RATINGS = {1, 2, 3, 4, 5}
MIN_DEPT_RESPONSES = 50
N_HOTSPOTS = 10  # top-N worst departments count as hotspots

# ---------------------------------------------------------------
# Design tokens — light "dashboard" palette (KPI cards + charts)
# ---------------------------------------------------------------
COLOR = {
    "ink":     "#1E2430",
    "slate":   "#64748B",
    "mist":    "#F6F7F9",
    "line":    "#E4E7EC",
    "teal":    "#0E7C7B",
    "indigo":  "#4C5FD5",
    "coral":   "#E2725B",
    "amber":   "#DB9A3C",
    "crimson": "#C1483D",
}
FONT_SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
FONT_SERIF = "Georgia, 'Iowan Old Style', 'Palatino Linotype', serif"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": COLOR["line"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": COLOR["line"],
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": COLOR["ink"],
    "axes.labelcolor": COLOR["slate"],
    "xtick.color": COLOR["slate"],
    "ytick.color": COLOR["slate"],
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": COLOR["ink"],
    "figure.dpi": 110,
})
pd.set_option("display.max_colwidth", 120)

# ---------------------------------------------------------------
# Triage classification tables (from the live triage panel)
# ---------------------------------------------------------------
THEME_MAP = {
    "Staff were helpful.": "Positive - Staff",
    "Good experience.": "Positive - General",
    "Everything was fine.": "Positive - General",
    "I waited much longer than expected.": "Wait Time",
    "The wait was too long.": "Wait Time",
    "The waiting area was crowded.": "Wait Time",
    "Scheduling needs improvement.": "Scheduling",
    "Instructions were unclear.": "Communication",
    "I did not receive enough updates.": "Communication",
    "Communication could be better.": "Communication",
    "Insurance information was not clear.": "Billing",
    "The bill was confusing.": "Billing",
    "I received multiple billing notices.": "Billing",
    "The provider seemed rushed.": "Provider",
    "The visit felt incomplete.": "Provider",
    "I wanted more time with the clinician.": "Provider",
    "The facility was hard to navigate.": "Facilities",
    "Parking was difficult.": "Facilities",
}

ACTIONS = {
    "Wait Time": "Apologize for delay, offer priority scheduling on next visit; flag department for scheduling capacity review.",
    "Scheduling": "Route to scheduling team; check provider availability and booking-to-visit lag for that department.",
    "Communication": "Trigger a callback from the provider or nurse to clarify instructions within 48 hours.",
    "Billing": "Route to billing team to review and explain charges; consider fee waiver for repeat billing complaints.",
    "Provider": "Manager review of the visit; consider provider coaching or reassignment for next appointment.",
    "Facilities": "Log to operations/facilities team — building-level fix (parking, navigation, cleanliness), not per-patient.",
    "Positive - Staff": "No action needed — candidate for positive review request / testimonial.",
    "Positive - General": "No action needed — track for satisfaction benchmarking.",
}

SEVERITY = {
    1: {"label": "CRITICAL", "window": "same-day response, manager notified directly", "color": "#F0917C"},
    2: {"label": "HIGH", "window": "respond within 24 hours", "color": "#F0917C"},
    3: {"label": "STANDARD", "window": "respond within 48–72 hours, routine queue", "color": "#E3B95F"},
}

# ---------------------------------------------------------------
# Predictive risk model — feature configuration
# ---------------------------------------------------------------
# Only features that are known once an appointment is booked (before the
# visit happens and long before any feedback/rating exists). Rating,
# comment, complaint_category, and visit "status" are deliberately excluded
# to avoid leaking the outcome we're trying to predict.
RISK_NUM_FEATS = [
    "wait_days", "overbooking_index", "avg_visit_duration_min",
    "manual_workload_multiplier", "years_experience", "provider_panel_size",
    "annual_budget", "base_wait_days", "claim_denial_risk", "lab_delay_risk",
]
RISK_BOOL_FEATS = ["chronic_diabetes", "chronic_hypertension"]
RISK_CAT_FEATS = ["department_name", "specialty", "visit_type", "insurance_type", "region", "gender"]

# Plain-English labels for the base (non-category-expanded) features.
RISK_LABELS = {
    "wait_days": "Longer wait time",
    "overbooking_index": "Higher overbooking at the clinic",
    "avg_visit_duration_min": "Longer average visit duration",
    "manual_workload_multiplier": "Higher manual staff workload",
    "years_experience": "More provider experience",
    "provider_panel_size": "Larger provider patient panel",
    "annual_budget": "Higher department budget",
    "base_wait_days": "Higher department baseline wait",
    "claim_denial_risk": "Higher claim-denial risk (dept.)",
    "lab_delay_risk": "Higher lab-delay risk (dept.)",
    "chronic_diabetes": "Patient has diabetes",
    "chronic_hypertension": "Patient has hypertension",
}
# Plain-English group labels for one-hot-encoded categorical features
# (e.g. "cat__insurance_type_Self-Pay" -> "Insurance: Self-Pay").
RISK_CAT_LABELS = {
    "department_name": "Department",
    "specialty": "Specialty",
    "visit_type": "Visit type",
    "insurance_type": "Insurance",
    "region": "Region",
    "gender": "Gender",
}


def friendly_feature_name(raw_name):
    """Turn an sklearn ColumnTransformer feature name (e.g.
    'num__wait_days' or 'cat__insurance_type_Self-Pay') into a short,
    plain-English label for charts."""
    if raw_name.startswith("num__"):
        base = raw_name[len("num__"):]
        return RISK_LABELS.get(base, base.replace("_", " ").capitalize())
    if raw_name.startswith("cat__"):
        rest = raw_name[len("cat__"):]
        for feat, label in RISK_CAT_LABELS.items():
            prefix = feat + "_"
            if rest.startswith(prefix):
                value = rest[len(prefix):]
                return f"{label}: {value}"
        return rest.replace("_", " ")
    return raw_name


# ---------------------------------------------------------------
# PAGE + STYLES
# ---------------------------------------------------------------
st.set_page_config(page_title="MedCore Patient Experience", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
.stApp{background:#F5F6F4;}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;font-weight:600;}
/* -- live triage monitor panel (dark) -- */
.monitor{background:#16233A;border-radius:6px;padding:26px;color:#fff;}
.readout{background:#1F2E42;border-radius:4px;padding:18px 20px;margin-top:6px;}
.readout .status{font-family:'Fraunces',serif;font-size:22px;font-weight:600;}
.status-risk{color:#F0917C;} .status-ok{color:#7FD4C6;} .status-review{color:#E3B95F;}
.readout .action{font-size:13.5px;color:#D6DEDC;line-height:1.5;}
.readout .action b{color:#fff;}
.escalate{margin-top:14px;padding-top:14px;border-top:1px solid #33465C;
  font-size:13.5px;color:#D6DEDC;line-height:1.5;}
.mono{font-family:'IBM Plex Mono',monospace;}
/* -- department dashboard header -- */
.dept-title{font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-size:30px;
  font-weight:700;color:#1E2430;margin-bottom:2px;}
.dept-sub{font-family:Helvetica Neue,Helvetica,Arial,sans-serif;font-size:13px;color:#64748B;
  margin-bottom:14px;padding-bottom:14px;border-bottom:2px solid #E4E7EC;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="font-family:Fraunces,serif;font-weight:600;margin-bottom:0;">MedCore — Patient Experience</h1>',
            unsafe_allow_html=True)
st.caption(
    "Four views: the department-level dashboard, a rule-based live feedback-triage panel, the revenue "
    "impact of missed appointments, and a pre-visit predictive risk model."
)

# ---------------------------------------------------------------
# Live refresh — this app re-reads patient_experience_master.csv on
# every refresh. It's not a push-based real-time feed (nothing pings
# this app when a new patient checks in); it's pull-based: as new rows
# get appended to that CSV (e.g. by whatever system exports it), hitting
# "Refresh now" clears the cached data/models and reloads from the file,
# so the numbers catch up to whatever's in it at that moment.
# ---------------------------------------------------------------
if "last_refreshed" not in st.session_state:
    st.session_state["last_refreshed"] = datetime.now()

with st.sidebar:
    st.markdown("#### 🔄 Live data")
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["last_refreshed"] = datetime.now()
        st.rerun()
    st.caption(f"Last refreshed: {st.session_state['last_refreshed'].strftime('%b %d, %Y · %I:%M %p')}")
    st.caption(
        "Pulls straight from patient_experience_master.csv. As new patient visits are appended to that file, "
        "refreshing will bring them into every tab — dashboard, triage, revenue impact, and the predictive model."
    )

# ---------------------------------------------------------------
# DATA LOADING (shared by all views)
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Loading feedback data…")
def load_data(source):
    df = pd.read_csv(
        source,
        parse_dates=["feedback_date", "appointment_date", "scheduled_date", "dob", "registration_date"],
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["wait_days"] = pd.to_numeric(df["wait_days"], errors="coerce")
    df["feedback_month"] = df["feedback_date"].dt.to_period("M").dt.to_timestamp()
    return df


local_path = Path(__file__).parent / DATA_FILE
if local_path.exists():
    df = load_data(str(local_path))
else:
    uploaded = st.sidebar.file_uploader(f"'{DATA_FILE}' not found — upload it here", type="csv")
    if uploaded is None:
        st.info(f"Place **{DATA_FILE}** next to this script, or upload it in the sidebar.")
        st.stop()
    df = load_data(uploaded)

departments = sorted(df["department_name"].dropna().unique().tolist())

with st.sidebar:
    latest_visit = df["appointment_date"].max()
    if pd.notna(latest_visit):
        st.caption(f"Most recent appointment in file: {latest_visit.strftime('%b %d, %Y')}")
    st.caption(f"Rows loaded: {len(df):,}")

# ---------------------------------------------------------------
# Revenue benchmark — computed directly from real billed-claim data,
# not an assumed/placeholder dollar figure. Averaged over visits that
# actually generated a claim (has_claim == True) rather than all
# "Completed"-status rows — about 35% of Completed visits have no claim
# on file, and including those as $0 would understate the per-visit
# figure. This matches the methodology in the MedCore Patient Experience
# Analysis report (avg. completed visit ≈ $617; ~$20.0M lost to all
# no-shows over the 3-year dataset).
# ---------------------------------------------------------------
AVG_COMPLETED_VISIT_REVENUE = df.loc[df["has_claim"] == True, "total_claim_amount"].mean()
historical_noshow_count = int((df["status"] == "No Show").sum())
historical_revenue_lost = historical_noshow_count * AVG_COMPLETED_VISIT_REVENUE


@st.cache_data(show_spinner=False)
def compute_triage_stats(df):
    """Theme / department aggregates used by the live triage classifier."""
    d = df[df["rating"].isin(VALID_RATINGS)].copy()
    d["theme"] = d["comment"].map(THEME_MAP)
    d = d.dropna(subset=["theme"])
    d["at_risk"] = d["rating"] <= 3

    org_rate = d["at_risk"].mean()

    themes = (
        d.groupby("theme")
        .agg(count=("theme", "size"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    dept_stats = (
        d.groupby(["department_id", "department_name"])
        .agg(count=("rating", "size"), at_risk_rate=("at_risk", "mean"))
        .reset_index()
    )
    dept_stats = dept_stats[dept_stats["count"] >= MIN_DEPT_RESPONSES].sort_values("at_risk_rate", ascending=False)
    hotspots = set(dept_stats.head(N_HOTSPOTS)["department_id"])
    return org_rate, themes, dept_stats, hotspots


org_rate, themes, dept_stats, hotspots = compute_triage_stats(df)
dept_lookup = {d.department_id: d for d in dept_stats.itertuples()}


# ---------------------------------------------------------------
# AI MODEL #1 — pre-visit predictive risk model
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Training pre-visit risk model…")
def train_risk_model(df):
    """Predicts P(patient will be at-risk / dissatisfied) using only
    information available once an appointment is booked — no rating,
    comment, or complaint_category is used as an input feature.

    Note on accuracy: we also tested a random forest, engineered features
    (wait relative to department baseline, wait x overbooking interaction,
    each patient's prior-visit average rating), and non-linear transforms
    of wait_days. None improved test AUC beyond ~0.665-0.667 — the dataset
    appears to encode dissatisfaction almost entirely as a (noisy) function
    of wait time, so that's the practical ceiling with these features. We
    kept logistic regression since it matches the ceiling and is fully
    interpretable for the feature-importance chart below.
    """
    d = df.dropna(subset=RISK_NUM_FEATS + RISK_CAT_FEATS).copy()
    for c in RISK_BOOL_FEATS:
        d[c] = d[c].astype(int)

    X = d[RISK_NUM_FEATS + RISK_BOOL_FEATS + RISK_CAT_FEATS]
    y = d["at_risk"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("num", StandardScaler(), RISK_NUM_FEATS + RISK_BOOL_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), RISK_CAT_FEATS),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    full_auc = roc_auc_score(y_test, proba)

    # Baseline: wait_days alone — the single strongest univariate driver —
    # so the presentation can show how much the other 15 features add.
    base = LogisticRegression(max_iter=1000)
    base.fit(X_train[["wait_days"]], y_train)
    base_auc = roc_auc_score(y_test, base.predict_proba(X_test[["wait_days"]])[:, 1])

    # Feature importance from standardized/one-hot coefficients, relabeled
    # into plain English for display.
    feat_names = pre.get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    importance = (
        pd.DataFrame({"feature": [friendly_feature_name(f) for f in feat_names], "coef": coefs})
        .assign(abs_coef=lambda x: x["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(8)
    )

    # Department-level averages, used to prefill the what-if calculator.
    dept_profile = d.groupby("department_name")[RISK_NUM_FEATS].mean()
    dept_mode = d.groupby("department_name")[RISK_CAT_FEATS[1:] + RISK_BOOL_FEATS].agg(lambda s: s.mode().iat[0])

    return pipe, full_auc, base_auc, importance, dept_profile, dept_mode


risk_pipe, risk_full_auc, risk_base_auc, risk_importance, risk_dept_profile, risk_dept_mode = train_risk_model(df)


# ---------------------------------------------------------------
# AI MODEL #2 — no-show predictor (money-lost tool)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Training no-show predictor…")
def train_noshow_model(df):
    """Predicts the chance a single, specific appointment gets missed,
    using only details known when it's booked (department, visit type,
    wait time, insurance, etc. — the same inputs as the risk model above).

    Why this model and not just a historical average: a department's
    overall no-show rate can already be read straight off the data with
    no model at all. What that number *can't* do is score one particular
    upcoming appointment — this patient, this wait time, this insurance —
    before we know what happens. That's the one thing this model adds.
    """
    d = df.dropna(subset=RISK_NUM_FEATS + RISK_CAT_FEATS + ["status"]).copy()
    for c in RISK_BOOL_FEATS:
        d[c] = d[c].astype(int)
    d["no_show"] = (d["status"] == "No Show").astype(int)

    feat_cols = RISK_NUM_FEATS + RISK_BOOL_FEATS + RISK_CAT_FEATS
    X = d[feat_cols]
    y = d["no_show"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("num", StandardScaler(), RISK_NUM_FEATS + RISK_BOOL_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), RISK_CAT_FEATS),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)

    return pipe


noshow_pipe = train_noshow_model(df)


# ---------------------------------------------------------------
# AI MODEL #3 — feedback-based next-appointment no-show risk
# (used only in the Live Feedback Triage tab)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Training feedback-based no-show model…")
def train_feedback_noshow_model(df):
    """Given a rating a patient just left, plus their wait time and visit
    type on the visit they're rating, predicts the chance they no-show
    their *next* appointment.

    This is a different question from the no-show model above, and
    deliberately uses `rating` as an input — the whole point here is that
    the rating is already known (feedback just came in), so using it isn't
    leakage in this context the way it would be for scoring an appointment
    that hasn't happened yet.

    Caveat worth being upfront about: in this dataset, `future_no_show_flag`
    lines up with the *same* visit's status rather than a genuinely separate
    later appointment, so treat this as an illustrative "what does this
    rating pattern predict" estimate, not a guarantee about a specific
    future booking.
    """
    d = df[df["rating"].isin(VALID_RATINGS)].copy()
    d = d.dropna(subset=["future_no_show_flag"])
    d["future_no_show_flag"] = d["future_no_show_flag"].astype(int)
    d["wait_days"] = d["wait_days"].fillna(d["wait_days"].median())

    visit_dummies = pd.get_dummies(d["visit_type"], prefix="visit_type")
    d = pd.concat([d, visit_dummies], axis=1)

    feat_cols = ["rating", "wait_days"] + list(visit_dummies.columns)
    X = d[feat_cols]
    y = d["future_no_show_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Department-level typical wait time / visit type, used to fill in the
    # two inputs the triage tab doesn't otherwise collect.
    dept_wait_typical = d.groupby("department_name")["wait_days"].median()
    dept_visit_typical = d.groupby("department_name")["visit_type"].agg(lambda s: s.mode().iat[0])

    return model, feat_cols, list(visit_dummies.columns), dept_wait_typical, dept_visit_typical


(
    feedback_noshow_model,
    feedback_noshow_feats,
    feedback_visit_cols,
    feedback_dept_wait,
    feedback_dept_visit,
) = train_feedback_noshow_model(df)

# =================================================================
# TAB LAYOUT
# =================================================================
tab_dashboard, tab_triage, tab_revenue, tab_risk = st.tabs(
    ["📊 Department Dashboard", "🚨 Live Feedback Triage", "💰 Revenue Impact", "🔮 Predictive Risk Model"]
)

# -----------------------------------------------------------------
# TAB 1 — Department dashboard (converted from the notebook)
# -----------------------------------------------------------------
with tab_dashboard:

    def get_department_data(dept):
        return df[df["department_name"] == dept].copy()

    def compute_kpis(d):
        """NaN/column-safe so a missing/empty slice never breaks the KPI row."""
        n = len(d)

        def safe_mean(col, pct=False, cond=None):
            if not n or col not in d.columns:
                return np.nan
            series = (d[col] == cond) if cond is not None else d[col]
            val = series.mean()
            return val * 100 if pct else val

        return {
            "n_feedback": n,
            "avg_rating": safe_mean("rating"),
            "satisfied_pct": safe_mean("patient_satisfaction_after_visit", pct=True, cond="Satisfied"),
            "at_risk_pct": safe_mean("at_risk", pct=True) if "at_risk" in d.columns
                           else ((d["rating"] <= 3).mean() * 100 if n else np.nan),
            "avg_wait_days": safe_mean("wait_days"),
            "no_show_pct": safe_mean("status", pct=True, cond="No Show"),
            "avg_budget": safe_mean("annual_budget"),
            "location": d["location"].mode().iat[0] if n and "location" in d.columns and not d["location"].mode().empty else "N/A",
        }

    def fmt(value, spec="", suffix=""):
        """Render a number, or 'N/A' if it's missing — never a raw 'nan'."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:{spec}}{suffix}"

    def kpi_card_html(label, value, accent, sub=""):
        # Built as ONE single-line string (no embedded newlines/indentation) so it
        # always renders as an inline HTML fragment instead of being mistaken for
        # an indented markdown code block.
        return (
            f'<div style="flex:1; min-width:150px; background:{COLOR["mist"]}; '
            f'border-radius:10px; padding:16px 18px; margin:6px; border-top:4px solid {accent}; '
            f'box-shadow:0 1px 2px rgba(16,24,40,0.04); font-family:{FONT_SANS};">'
            f'<div style="font-size:11px; font-weight:600; color:{COLOR["slate"]}; '
            f'text-transform:uppercase; letter-spacing:0.06em;">{label}</div>'
            f'<div style="font-size:28px; font-weight:700; color:{COLOR["ink"]}; margin-top:6px; '
            f'line-height:1.1;">{value}</div>'
            f'<div style="font-size:11px; color:{COLOR["slate"]}; margin-top:4px; min-height:14px;">{sub}</div>'
            f'</div>'
        )

    def render_kpi_row(kpis):
        cards = [
            kpi_card_html("Feedback Count", fmt(kpis["n_feedback"], ","), COLOR["indigo"], "total responses"),
            kpi_card_html("Avg. Rating", fmt(kpis["avg_rating"], ".2f", " / 5"), COLOR["teal"], "across all feedback"),
            kpi_card_html("Satisfied %", fmt(kpis["satisfied_pct"], ".1f", "%"), COLOR["teal"], "self-reported satisfied"),
            kpi_card_html("At-Risk %", fmt(kpis["at_risk_pct"], ".1f", "%"), COLOR["amber"], "flagged as at-risk"),
            kpi_card_html("Avg. Wait (days)", fmt(kpis["avg_wait_days"], ".1f"), COLOR["slate"], "scheduled to seen"),
            kpi_card_html("No-Show %", fmt(kpis["no_show_pct"], ".1f", "%"), COLOR["crimson"], "missed appointments"),
        ]
        st.markdown(f'<div style="display:flex; flex-wrap:wrap;">{"".join(cards)}</div>', unsafe_allow_html=True)

    def _style_axis(ax, title):
        ax.set_title(title, pad=10)
        ax.tick_params(length=0)
        ax.set_axisbelow(True)

    def plot_charts(d, dept):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

        rating_counts = d["rating"].value_counts().sort_index()
        rating_counts = rating_counts.reindex([1, 2, 3, 4, 5], fill_value=0)
        total = rating_counts.sum()
        bars = axes[0].bar(rating_counts.index.astype(str), rating_counts.values,
                            color=COLOR["teal"], width=0.6, zorder=3)
        for b, v in zip(bars, rating_counts.values):
            if total:
                axes[0].annotate(f"{v/total*100:.0f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                                  textcoords="offset points", xytext=(0, 4), ha="center",
                                  fontsize=9, color=COLOR["slate"])
        _style_axis(axes[0], "Rating Distribution")
        axes[0].set_xlabel("Rating")
        axes[0].set_ylabel("Count")
        axes[0].margins(y=0.15)

        complaint_counts = d["complaint_category"].value_counts(dropna=True).head(6).sort_values()
        if len(complaint_counts):
            bars2 = axes[1].barh(complaint_counts.index, complaint_counts.values,
                                  color=COLOR["coral"], height=0.6, zorder=3)
            cmax = complaint_counts.max()
            for b, v in zip(bars2, complaint_counts.values):
                axes[1].annotate(f"{v:,}", (b.get_width(), b.get_y() + b.get_height()/2),
                                  textcoords="offset points", xytext=(6, 0), va="center",
                                  fontsize=9, color=COLOR["slate"])
            axes[1].set_xlim(0, cmax * 1.18)
        else:
            axes[1].text(0.5, 0.5, "No complaint data", ha="center", va="center",
                          color=COLOR["slate"], transform=axes[1].transAxes)
        _style_axis(axes[1], "Top Complaint Categories")
        axes[1].set_xlabel("Count")
        axes[1].grid(axis="x")
        axes[1].grid(axis="y", visible=False)

        monthly = d.groupby("feedback_month")["rating"].mean().sort_index()
        if len(monthly):
            axes[2].plot(monthly.index, monthly.values, marker="o", markersize=4,
                          color=COLOR["indigo"], linewidth=2, zorder=3)
            axes[2].fill_between(monthly.index, monthly.values, monthly.values.min() - 0.3,
                                  color=COLOR["indigo"], alpha=0.08)
        _style_axis(axes[2], "Avg. Rating Trend (Monthly)")
        axes[2].set_ylabel("Avg. Rating")
        axes[2].set_ylim(1, 5)
        axes[2].tick_params(axis="x", rotation=45)

        fig.suptitle(dept, fontsize=15, fontweight="bold", color=COLOR["ink"], x=0.01, ha="left", y=1.03)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    def render_feedback_table(d, n=8):
        cols = ["feedback_date", "rating", "complaint_category", "status", "comment"]
        table = (
            d.dropna(subset=["comment"])
             .sort_values("feedback_date", ascending=False)
             .loc[:, cols]
             .head(n)
             .rename(columns={
                 "feedback_date": "Date",
                 "rating": "Rating",
                 "complaint_category": "Category",
                 "status": "Visit Status",
                 "comment": "Comment",
             })
        )
        table["Date"] = table["Date"].dt.strftime("%b %d, %Y")
        table["Category"] = table["Category"].fillna("General")

        st.markdown(
            f"<h4 style='font-family:{FONT_SANS}; color:{COLOR['ink']}; "
            f"margin:22px 0 8px 0; font-size:15px;'>Recent Patient Comments</h4>",
            unsafe_allow_html=True,
        )

        styled = (
            table.style
            .hide(axis="index")
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", COLOR["ink"]), ("color", "white"),
                    ("font-family", FONT_SANS), ("font-size", "12px"),
                    ("text-align", "left"), ("padding", "8px 12px"),
                    ("text-transform", "uppercase"), ("letter-spacing", "0.04em"),
                ]},
                {"selector": "td", "props": [
                    ("font-family", FONT_SANS), ("font-size", "13px"),
                    ("padding", "8px 12px"), ("border-bottom", f"1px solid {COLOR['line']}"),
                    ("color", COLOR["ink"]),
                ]},
                {"selector": "tr:nth-child(even) td", "props": [
                    ("background-color", COLOR["mist"]),
                ]},
            ])
        )
        st.markdown(styled.to_html(), unsafe_allow_html=True)

    dept_choice = st.selectbox("Department:", departments, index=0, key="dept_dashboard_select")
    d = get_department_data(dept_choice)
    location = d["location"].mode().iat[0] if not d.empty and not d["location"].mode().empty else "N/A"

    st.markdown(
        f'<div class="dept-title">{dept_choice}</div>'
        f'<div class="dept-sub">Location: {location}</div>',
        unsafe_allow_html=True,
    )
    kpis = compute_kpis(d)
    render_kpi_row(kpis)
    plot_charts(d, dept_choice)
    render_feedback_table(d)

# -----------------------------------------------------------------
# TAB 2 — Live feedback triage (from the standalone script)
# -----------------------------------------------------------------
with tab_triage:
    st.markdown('<h2 style="font-family:Fraunces,serif;font-weight:600;">Live prototype — instant feedback triage</h2>',
                unsafe_allow_html=True)
    st.markdown('<div class="monitor" style="margin-bottom:8px;"><div style="display:flex;justify-content:space-between;">'
                '<span class="eyebrow" style="color:#7FD4C6;">● LIVE CLASSIFIER</span>'
                '<span class="eyebrow" style="color:#9FB0AE;">INTERNAL — NOT PATIENT-FACING</span>'
                '</div></div>', unsafe_allow_html=True)

    theme_names = list(themes["theme"])
    default_theme = theme_names.index("Billing") if "Billing" in theme_names else 0

    c1, c2, c3 = st.columns(3)
    rating = c1.slider("Patient rating", 1, 5, 2, key="triage_rating")
    theme = c2.selectbox("Comment theme", theme_names, index=default_theme, key="triage_theme")
    dept_id = c3.selectbox(
        "Department",
        list(dept_stats["department_id"]),
        format_func=lambda i: (f"{i} · {dept_lookup[i].department_name} "
                               f"(avg {dept_lookup[i].at_risk_rate*100:.0f}% at-risk)"),
        key="triage_dept",
    )

    at_risk = rating <= 3
    theme_avg = float(themes.loc[themes["theme"] == theme, "avg_rating"].iloc[0])

    if at_risk:
        status_html = '<div class="status status-risk">At Risk</div>'
        sev = SEVERITY[rating]
        sev_html = (f'<div class="mono" style="font-size:13px;margin-top:6px;">'
                    f'<span style="color:{sev["color"]};font-weight:600;">{sev["label"]}</span> '
                    f'<span style="color:#7FD4C6;">· {sev["window"]}</span></div>')
        action_html = f'<b>{theme}</b> — {ACTIONS[theme]}'
    else:
        status_html = '<div class="status status-ok">Satisfied</div>'
        sev_html = ""
        action_html = (f'<b>No action needed</b> — rating is 4–5, so this feedback is classified Satisfied '
                       f'regardless of theme selected. (In the data, "{theme}" has a historical average '
                       f'rating of {theme_avg:.1f}.)')

    escalate_html = ""
    if at_risk and dept_id in hotspots:
        dd = dept_lookup[dept_id]
        escalate_html = (f'<div class="escalate"><span class="eyebrow" style="color:#F0917C;">⚠ Escalation — '
                         f'pattern detected</span><br><b>{dept_id} · {dd.department_name}</b> already runs a '
                         f'{dd.at_risk_rate*100:.0f}% At-Risk rate across {dd.count:,} responses — well above the '
                         f'{org_rate*100:.1f}% org average. This complaint isn\'t an isolated case; it\'s '
                         f'confirming an existing department-level problem. In addition to resolving this '
                         f'patient\'s issue, this is auto-flagged for the department manager as a recurring '
                         f'pattern, not a one-off.</div>')

    # -- Feedback-based next-appointment no-show risk (the one trained-model
    # number on this otherwise rule-based tab) --
    dept_name_for_ml = dept_lookup[dept_id].department_name
    next_wait = float(feedback_dept_wait.get(dept_name_for_ml, feedback_dept_wait.median()))
    next_visit_type = feedback_dept_visit.get(dept_name_for_ml, feedback_dept_visit.mode().iat[0])

    ml_row = {c: 0 for c in feedback_visit_cols}
    ml_row["rating"] = rating
    ml_row["wait_days"] = next_wait
    vt_col = f"visit_type_{next_visit_type}"
    if vt_col in ml_row:
        ml_row[vt_col] = 1
    ml_df = pd.DataFrame([ml_row])[feedback_noshow_feats]
    next_noshow_prob = float(feedback_noshow_model.predict_proba(ml_df)[0, 1])

    next_noshow_html = (
        f'<div style="margin-top:14px;padding-top:14px;border-top:1px solid #33465C;'
        f'font-size:13.5px;color:#D6DEDC;line-height:1.5;">'
        f'<span class="eyebrow" style="color:#7FD4C6;">🤖 ML estimate</span><br>'
        f'📅 <b style="color:#fff;">{next_noshow_prob*100:.0f}% chance this patient no-shows their next '
        f'appointment</b> — based on this rating ({rating}/5), plus typical wait time and visit type for '
        f'{dept_name_for_ml}.</div>'
    )

    st.markdown(f"""
    <div class="monitor" style="margin-top:4px;">
      <div class="readout" style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:16px;">
        <div>
          <div class="eyebrow" style="color:#9FB0AE;margin-bottom:4px;">Classification</div>
          {status_html}{sev_html}
        </div>
        <div class="action" style="max-width:460px;">
          <div class="eyebrow" style="color:#9FB0AE;margin-bottom:4px;">Recommended action</div>
          {action_html}
        </div>
      </div>
      {escalate_html}
      {next_noshow_html}
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------
# TAB 3 — Revenue impact (historical, live-refreshed, no model).
# Placed before the Predictive tab: this is the simpler, immediately-
# actionable "here's what missed appointments already cost us" story,
# read straight from the billing data. The Predictive tab builds on
# top of it with a model that scores risk on a specific upcoming
# appointment.
# -----------------------------------------------------------------
with tab_revenue:
    st.markdown(
        f"<h2 style='font-family:Fraunces,serif;font-weight:600;display:inline;'>Revenue impact — money lost to "
        f"missed appointments</h2> "
        f"<span style='font-size:11px; font-weight:600; color:{COLOR['teal']}; text-transform:uppercase; "
        f"letter-spacing:0.06em; vertical-align:middle;'>● live</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"How this number is calculated: when a visit actually happens, it gets billed — about "
        f"\\${AVG_COMPLETED_VISIT_REVENUE:,.0f} on average in this data. When a patient no-shows, the visit "
        f"never happens, so nothing gets billed — \\$0. That gap is real, already-realized revenue lost every "
        f"time an appointment is missed, not a projection or estimate. Everything on this tab is computed "
        f"directly from the billing data — no model is involved. (For a model-based estimate of the dollar "
        f"risk on one specific *upcoming* appointment, see the what-if calculator on the Predictive Risk "
        f"Model tab.)"
    )
    st.caption(
        f"Pulled from patient_experience_master.csv as of your last refresh "
        f"({st.session_state['last_refreshed'].strftime('%b %d, %Y · %I:%M %p')}) — use \"Refresh now\" in the "
        f"sidebar to pull in newly added patients."
    )

    money_cards = [
        kpi_card_html("Appointments Missed", f"{historical_noshow_count:,}", COLOR["crimson"], "No Show, all-time"),
        kpi_card_html("Revenue Lost", f"\\${historical_revenue_lost:,.0f}", COLOR["crimson"],
                      f"{historical_noshow_count:,} × ~\\${AVG_COMPLETED_VISIT_REVENUE:,.0f} per visit"),
        kpi_card_html("Avg. Revenue / Visit", f"\\${AVG_COMPLETED_VISIT_REVENUE:,.0f}", COLOR["teal"], "completed visits with a billed claim"),
    ]
    st.markdown(f'<div style="display:flex; flex-wrap:wrap;">{"".join(money_cards)}</div>', unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:13px; color:{COLOR['slate']}; margin:22px 0 8px 0; font-family:{FONT_SANS};'>"
        f"Which departments miss the most appointments? "
        f"<span style='color:{COLOR['slate']}; font-weight:400;'>(hover the bars for exact numbers)</span></div>",
        unsafe_allow_html=True,
    )

    dm1, dm2 = st.columns([2, 1])
    with dm1:
        metric_choice = st.radio(
            "View by",
            ["Share of appointments missed (%)", "Revenue lost historically ($)"],
            horizontal=True,
            key="noshow_metric_choice",
            label_visibility="collapsed",
        )
    with dm2:
        top_n = st.slider("Departments to show", 5, 20, 10, key="noshow_topn")

    g = df[df["status"].notna()].groupby("department_name")["status"]
    dept_ns = pd.DataFrame({
        "pct_missed": g.apply(lambda s: (s == "No Show").mean() * 100),
        "no_show_count": g.apply(lambda s: (s == "No Show").sum()),
    })
    dept_ns["revenue_lost"] = dept_ns["no_show_count"] * AVG_COMPLETED_VISIT_REVENUE

    if metric_choice.startswith("Share"):
        col, x_title, hover_fmt = "pct_missed", "Share of appointments missed (%)", "%{x:.1f}%"
    else:
        col, x_title, hover_fmt = "revenue_lost", "Revenue lost ($)", "$%{x:,.0f}"

    dept_plot = dept_ns.sort_values(col, ascending=False).head(top_n).sort_values(col)

    fig3 = px.bar(
        dept_plot,
        x=col,
        y=dept_plot.index,
        orientation="h",
        color_discrete_sequence=[COLOR["crimson"]],
    )
    fig3.update_traces(hovertemplate="<b>%{y}</b><br>" + hover_fmt + "<extra></extra>")
    fig3.update_layout(
        xaxis_title=x_title,
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT_SANS, color=COLOR["ink"], size=12),
        margin=dict(l=10, r=20, t=10, b=40),
        height=110 + 28 * len(dept_plot),
        bargap=0.3,
    )
    fig3.update_xaxes(gridcolor=COLOR["line"], zeroline=False)
    fig3.update_yaxes(showgrid=False)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "The cards and chart above update every time you refresh the data (sidebar) — the model doesn't "
        "touch them, they're just live counts off the file. For a model-based estimate of the dollar risk "
        "on one specific upcoming appointment, see the what-if calculator on the Predictive Risk Model tab."
    )

# -----------------------------------------------------------------
# TAB 4 — Predictive pre-visit risk model
# -----------------------------------------------------------------
with tab_risk:
    st.markdown(
        '<h2 style="font-family:Fraunces,serif;font-weight:600;">Predictive risk model — catch it before the visit</h2>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The triage tab reacts *after* a patient reports a bad experience. This model scores the risk of a "
        "poor experience as soon as an appointment is booked — using only information known before the visit "
        "(department, visit type, scheduled wait, provider/department profile, patient demographics). "
        "Rating, comment, and complaint category are never used as inputs, so there's no leakage from the "
        "outcome itself."
    )

    st.markdown(
        f"<h4 style='font-family:{FONT_SANS}; color:{COLOR['ink']}; margin:22px 0 4px 0; font-size:15px;'>"
        f"Top driver of predicted risk: wait time</h4>",
        unsafe_allow_html=True,
    )
    wait_coef = float(risk_importance.loc[risk_importance["feature"] == "Longer wait time", "coef"].iloc[0])
    left, mid, right = st.columns([1, 3, 1])
    with mid:
        fig, ax = plt.subplots(figsize=(6.5, 2.2))
        ax.barh(["Longer wait time"], [wait_coef], color=COLOR["crimson"], height=0.5, zorder=3)
        ax.axvline(0, color=COLOR["ink"], linewidth=0.8)
        ax.set_xlabel("← lowers risk   |   raises risk →", fontsize=9)
        ax.set_xlim(0, wait_coef * 1.25)
        ax.tick_params(length=0, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with st.expander("The model also weighs 7 other factors, at smaller effect sizes"):
        st.caption(
            "Wait time is the dominant signal, but the model isn't wait-time-only — department, insurance "
            "type, visit type, gender, and a few others still shift the prediction, just by less. Full "
            "ranked list, most to least influential:"
        )
        other = risk_importance[risk_importance["feature"] != "Longer wait time"].sort_values("abs_coef", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6.5, 2.6))
        imp2 = other.sort_values("coef")
        colors2 = [COLOR["crimson"] if c > 0 else COLOR["teal"] for c in imp2["coef"]]
        ax2.barh(imp2["feature"], imp2["coef"], color=colors2, zorder=3)
        ax2.axvline(0, color=COLOR["ink"], linewidth=0.8)
        ax2.set_xlabel("← lowers risk   |   raises risk →", fontsize=9)
        ax2.tick_params(length=0, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

    st.markdown(
        f"<h4 style='font-family:{FONT_SANS}; color:{COLOR['ink']}; margin:26px 0 8px 0; font-size:15px;'>"
        f"What-if calculator — score a hypothetical appointment</h4>",
        unsafe_allow_html=True,
    )
    st.caption("Pick a department and a scenario; other operational fields default to that department's average profile.")

    c1, c2, c3, c4 = st.columns(4)
    calc_dept = c1.selectbox("Department", sorted(risk_dept_profile.index.tolist()), key="risk_dept")
    calc_wait = c2.slider("Scheduled wait (days)", 0, 90, int(risk_dept_profile.loc[calc_dept, "wait_days"]), key="risk_wait")
    calc_visit = c3.selectbox("Visit type", sorted(df["visit_type"].dropna().unique().tolist()), key="risk_visit")
    calc_insurance = c4.selectbox("Insurance type", sorted(df["insurance_type"].dropna().unique().tolist()), key="risk_insurance")

    row = {f: risk_dept_profile.loc[calc_dept, f] for f in RISK_NUM_FEATS}
    row["wait_days"] = calc_wait
    row["chronic_diabetes"] = int(risk_dept_mode.loc[calc_dept, "chronic_diabetes"])
    row["chronic_hypertension"] = int(risk_dept_mode.loc[calc_dept, "chronic_hypertension"])
    row["department_name"] = calc_dept
    row["specialty"] = risk_dept_mode.loc[calc_dept, "specialty"]
    row["visit_type"] = calc_visit
    row["insurance_type"] = calc_insurance
    row["region"] = risk_dept_mode.loc[calc_dept, "region"]
    row["gender"] = risk_dept_mode.loc[calc_dept, "gender"]

    calc_df = pd.DataFrame([row])[RISK_NUM_FEATS + RISK_BOOL_FEATS + RISK_CAT_FEATS]
    risk_prob = float(risk_pipe.predict_proba(calc_df)[0, 1])
    calc_noshow_prob = float(noshow_pipe.predict_proba(calc_df)[0, 1])
    calc_dollars = calc_noshow_prob * AVG_COMPLETED_VISIT_REVENUE

    if risk_prob >= 0.55:
        bucket, bucket_color, rec = "HIGH RISK", COLOR["crimson"], "Proactively call the patient before the visit, or move up the appointment if capacity allows."
    elif risk_prob >= 0.40:
        bucket, bucket_color, rec = "MEDIUM RISK", COLOR["amber"], "Add to the department's watch list; a routine reminder call before the visit is worthwhile."
    else:
        bucket, bucket_color, rec = "LOW RISK", COLOR["teal"], "No special intervention needed — standard workflow."

    st.markdown(
        f'<div style="background:{COLOR["mist"]}; border-radius:10px; padding:20px 24px; '
        f'border-left:6px solid {bucket_color}; margin-top:10px;">'
        f'<div style="font-size:12px; font-weight:600; color:{COLOR["slate"]}; text-transform:uppercase; letter-spacing:0.06em;">Predicted risk</div>'
        f'<div style="font-size:32px; font-weight:700; color:{COLOR["ink"]}; margin-top:4px;">{risk_prob*100:.0f}% '
        f'<span style="font-size:16px; color:{bucket_color}; font-weight:700;">· {bucket}</span></div>'
        f'<div style="font-size:13px; color:{COLOR["slate"]}; margin-top:8px;">{rec}</div>'
        f'<div style="font-size:13px; color:{COLOR["slate"]}; margin-top:10px; padding-top:10px; border-top:1px solid {COLOR["line"]};">'
        f'💰 <b style="color:{COLOR["ink"]};">{calc_noshow_prob*100:.0f}% chance this specific appointment gets missed</b> '
        f'— worth about <b style="color:{COLOR["crimson"]};">\\${calc_dollars:,.0f}</b> if it does '
        f'(a completed visit like this bills ~\\${AVG_COMPLETED_VISIT_REVENUE:,.0f} on average).</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "This dollar figure is the one model-based revenue number in the app — it scores *one specific "
        "upcoming appointment* before it happens. For the full historical, live-refreshed picture (total "
        "revenue lost to date, and which departments lose the most), see the Revenue Impact tab."
    )
