"""Phase 8 dashboard: UK PM rhetoric analysis (CLAUDE.md §4, §8).

Single-file Streamlit app, four tabs over the outputs of Phases 3-7:
overview/stylometric profile, sentiment & certainty over time (with crisis
windows), LDA topics over time, and the PM-attribution classifier's results.
Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from gensim import corpora
from gensim.models import LdaModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Import the two pandas/gensim-only modules the dashboard needs directly from
# src/, rather than pip-installing the full hansard_pm_nlp package - that
# package's pyproject.toml pulls in torch/transformers/bertopic/spacy, none
# of which this dashboard uses (all heavy computation already happened
# offline; see requirements-app.txt).
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hansard_pm_nlp.event_study import CRISIS_WINDOWS  # noqa: E402
from hansard_pm_nlp.lda import get_top_words  # noqa: E402

PM_COLORS = {
    "Boris Johnson": "#1f77b4",
    "Liz Truss": "#ff7f0e",
    "Rishi Sunak": "#2ca02c",
    "Keir Starmer": "#d62728",
}

st.set_page_config(page_title="UK PM Rhetoric Analysis", layout="wide")


def _dark(fig: go.Figure) -> go.Figure:
    """Transparent background so Plotly charts blend into Streamlit's dark
    theme instead of showing a plotly-default white/light panel.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar={"bgcolor": "rgba(0,0,0,0)"} if fig.layout.polar is not None else None,
    )
    return fig


@st.cache_data
def load_affect() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "pm_contributions_affect.parquet")


@st.cache_data
def load_eda_summary() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "eda_summary.csv")


@st.cache_data
def load_affect_summary() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "affect_summary.csv")


@st.cache_data
def load_event_study() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "event_study_dataset.parquet")


@st.cache_data
def load_lda_topics() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "lda_topics.parquet")


@st.cache_resource
def load_lda_model() -> tuple[LdaModel, dict[int, list[str]]]:
    model = LdaModel.load(str(PROCESSED_DIR / "lda_model.gensim"))
    dictionary = corpora.Dictionary.load(str(PROCESSED_DIR / "lda_dictionary.gensim"))
    del dictionary
    top_words = get_top_words(model, topn=5)
    return model, top_words


@st.cache_data
def load_classifier_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PROCESSED_DIR / "phase6_test_predictions.csv")
    metrics = pd.read_csv(PROCESSED_DIR / "phase6_metrics.csv")
    fi_logreg = pd.read_csv(PROCESSED_DIR / "phase6_feature_importance_logreg.csv")
    fi_hgb = pd.read_csv(PROCESSED_DIR / "phase6_feature_importance_hgb.csv")
    return predictions, metrics, fi_logreg, fi_hgb


st.title("UK Prime Ministers' Rhetoric, 2019-2026")
st.caption(
    "Hansard speech data, Johnson through Starmer. See the GitHub repos "
    "(hansard-pm-extraction, hansard-pm-nlp) for methodology and full reports."
)

tab_overview, tab_affect, tab_topics, tab_classifier = st.tabs(
    ["Overview", "Sentiment & certainty", "Topics", "PM classifier"]
)

# --- Overview: stylometric radar chart ------------------------------------
with tab_overview:
    st.subheader("Stylometric profile by PM")
    st.caption(
        "Whole-corpus averages (Phases 3-4), not affected by the filters on "
        "other tabs - a radar built from a handful of filtered sittings "
        "would be too noisy to compare fairly."
    )

    eda = load_eda_summary()
    affect_sum = load_affect_summary()
    profile = eda.merge(affect_sum, on=["pm_name", "n_contributions"])

    all_pms = profile["pm_name"].tolist()
    selected_pms = st.multiselect("PMs to compare", all_pms, default=all_pms)

    metrics_cfg = [
        ("mtld", "Lexical diversity (MTLD)"),
        ("mean_flesch_kincaid_grade", "Readability (Flesch-Kincaid)"),
        ("mean_words_per_sentence", "Mean words/sentence"),
        ("mean_hedge_rate", "Hedging rate"),
        ("mean_vader_compound", "Mean sentiment (VADER)"),
    ]
    cols = [c for c, _ in metrics_cfg]
    labels = [label for _, label in metrics_cfg]

    normalized = profile.set_index("pm_name")[cols]
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())

    fig = go.Figure()
    for pm in selected_pms:
        values = normalized.loc[pm, cols].tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=pm,
                line_color=PM_COLORS.get(pm),
            )
        )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
        height=550,
    )
    st.plotly_chart(_dark(fig), width="stretch")
    st.caption(
        "Each axis min-max normalized across PMs (0=lowest, 1=highest) so "
        "shapes are comparable. Type-token ratio (TTR) is deliberately left "
        "off this chart - it is length-biased (lexical.py) and Liz Truss's "
        "tiny corpus (8,842 words vs 220k-520k for the others) makes her TTR "
        "an artifact of sample size, not style; MTLD is the length-corrected "
        "alternative shown here. TTR is still in the raw values table below."
    )

    with st.expander("Raw values"):
        st.dataframe(profile[["pm_name", *cols]].set_index("pm_name"))

# --- Sentiment & certainty over time ---------------------------------------
with tab_affect:
    st.subheader("Sentiment and certainty over time")
    st.caption(
        "Sitting-level means (Phase 7 event-study dataset). Shaded bands mark "
        "the four named crisis windows. Phase 7's regressions found **no "
        "effect surviving multiple-testing correction** for any crisis - see "
        "phase7_event_study_report.md for the full null result and why."
    )

    event_df = load_event_study().sort_values("sitting_date")
    pms_present = event_df["pm_name"].unique().tolist()
    selected = st.multiselect("PMs", pms_present, default=pms_present, key="affect_pms")
    date_min, date_max = event_df["sitting_date"].min(), event_df["sitting_date"].max()
    date_range = st.slider(
        "Date range",
        min_value=date_min.to_pydatetime(),
        max_value=date_max.to_pydatetime(),
        value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    )

    filtered = event_df[
        event_df["pm_name"].isin(selected)
        & (event_df["sitting_date"] >= date_range[0])
        & (event_df["sitting_date"] <= date_range[1])
    ]

    dv_choice = st.radio(
        "Metric",
        ["vader_compound", "transformer_score", "net_certainty"],
        format_func=lambda x: {
            "vader_compound": "Sentiment (VADER)",
            "transformer_score": "Sentiment (transformer)",
            "net_certainty": "Net certainty (booster - hedge rate)",
        }[x],
        horizontal=True,
    )

    fig = go.Figure()
    for pm in selected:
        sub = filtered[filtered["pm_name"] == pm]
        fig.add_trace(
            go.Scatter(
                x=sub["sitting_date"],
                y=sub[dv_choice],
                mode="markers",
                name=pm,
                marker={"color": PM_COLORS.get(pm), "size": 6},
            )
        )
    for name, (start, end) in CRISIS_WINDOWS.items():
        in_view = pd.Timestamp(end) >= pd.Timestamp(date_range[0]) and pd.Timestamp(
            start
        ) <= pd.Timestamp(date_range[1])
        if in_view:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="grey",
                opacity=0.15,
                line_width=0,
                annotation_text=name.replace("_", " "),
                annotation_position="top left",
            )
    fig.update_layout(height=500, xaxis_title="Sitting date", yaxis_title=dv_choice)
    st.plotly_chart(_dark(fig), width="stretch")

# --- Topics over time -------------------------------------------------------
with tab_topics:
    st.subheader("LDA topics over time (K=14, merged to 13 for display)")
    st.caption(
        "One document per (PM, sitting date). Chosen over BERTopic after "
        "comparison (phase5_topic_comparison_report.md) - BERTopic's default "
        "pipeline collapsed most of this 296-document corpus into 1-2 broad "
        "clusters. Topics 'Ukraine/Russia' 0 and 1 are the one pair documented "
        "as overlapping by design (phase5_lda_report.md) and are summed into a "
        "single series below - the other 12 stay separate, including the three "
        "Covid-related topics, which look similar but track distinct sub-phases "
        "(restrictions/testing, vaccines/schools, NHS pay/inquiry) rather than "
        "one duplicated topic. Grey bands mark crisis windows (Phase 7); dotted "
        "lines mark PM transitions."
    )

    _, top_words = load_lda_model()
    topics_df = load_lda_topics()
    topic_cols = [c for c in topics_df.columns if c.startswith("topic_")]

    MERGED_LABEL = "T0+1: " + ", ".join(
        dict.fromkeys(top_words[0][:3] + top_words[1][:3])
    )
    topic_labels = {
        c: f"T{c.removeprefix('topic_')}: "
        + ", ".join(top_words[int(c.removeprefix("topic_"))][:3])
        for c in topic_cols
        if c not in ("topic_0", "topic_1")
    }

    pms_present = topics_df["pm_name"].unique().tolist()
    selected = st.multiselect("PMs", pms_present, default=pms_present, key="topic_pms")
    filtered = topics_df[topics_df["pm_name"].isin(selected)].sort_values("sitting_date")

    merged = filtered[topic_cols].copy()
    merged[MERGED_LABEL] = merged["topic_0"] + merged["topic_1"]
    merged = merged.drop(columns=["topic_0", "topic_1"]).rename(columns=topic_labels)
    merged["sitting_date"] = filtered["sitting_date"].values

    monthly = merged.set_index("sitting_date").resample("MS").mean().dropna(how="all")

    long_df = monthly.reset_index().melt(
        id_vars="sitting_date", var_name="Topic", value_name="Mean topic weight"
    )
    fig = px.area(
        long_df,
        x="sitting_date",
        y="Mean topic weight",
        color="Topic",
        labels={"sitting_date": "Month"},
    )

    chart_start, chart_end = monthly.index.min(), monthly.index.max()
    for name, (start, end) in CRISIS_WINDOWS.items():
        if pd.Timestamp(end) >= chart_start and pd.Timestamp(start) <= chart_end:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="grey",
                opacity=0.15,
                line_width=0,
                annotation_text=name.replace("_", " "),
                annotation_position="top left",
            )

    pm_transitions = filtered.groupby("pm_name")["sitting_date"].min().sort_values()
    for pm_name, start_date in pm_transitions.iloc[1:].items():
        fig.add_vline(
            x=start_date,
            line_dash="dot",
            line_color="rgba(255,255,255,0.4)",
            annotation_text=pm_name,
            annotation_position="top",
            annotation_textangle=-90,
        )

    fig.update_layout(height=600)
    st.plotly_chart(_dark(fig), width="stretch")

    with st.expander("Topic word lists"):
        for i, words in top_words.items():
            st.markdown(f"**Topic {i}**: {', '.join(words)}")

# --- Classifier --------------------------------------------------------------
with tab_classifier:
    st.subheader("PM-attribution classifier (H1)")
    st.caption(
        "3 classes (Boris Johnson, Rishi Sunak, Keir Starmer). Liz Truss "
        "excluded - 5 documents total, too few for reliable per-class "
        "metrics (see split.py). Train/test split by date within each PM's "
        "tenure, not random shuffle, so the test score reflects "
        "generalization to later, unseen sittings rather than a leaked "
        "news cycle."
    )

    predictions, metrics, fi_logreg, fi_hgb = load_classifier_outputs()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Logistic regression accuracy",
            f"{metrics.loc[metrics['model'] == 'logreg', 'accuracy'].iloc[0]:.1%}",
            help="Chance baselines: 33.3% uniform, 49.2% always-majority-class",
        )
    with col2:
        st.metric(
            "HistGradientBoosting accuracy",
            f"{metrics.loc[metrics['model'] == 'hgb', 'accuracy'].iloc[0]:.1%}",
        )

    model_choice = st.radio(
        "Model",
        ["pred_logreg", "pred_hgb"],
        format_func=lambda x: x.removeprefix("pred_"),
        horizontal=True,
    )
    labels = sorted(predictions["pm_name"].unique())
    cm = pd.crosstab(predictions["pm_name"], predictions[model_choice]).reindex(
        index=labels, columns=labels, fill_value=0
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=450)
    st.plotly_chart(_dark(fig), width="stretch")

    fi = fi_logreg if model_choice == "pred_logreg" else fi_hgb
    fig2 = px.bar(
        fi.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 features (permutation importance)",
    )
    fig2.update_layout(height=500)
    st.plotly_chart(_dark(fig2), width="stretch")
