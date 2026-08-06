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
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Import the two pandas/gensim-only modules the dashboard needs directly from
# src/, rather than pip-installing the full hansard_pm_nlp package - that
# package's pyproject.toml pulls in torch/transformers/bertopic/spacy, none
# of which this dashboard uses (all heavy computation already happened
# offline; see requirements-app.txt).
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hansard_pm_nlp.dashboard_helpers import (  # noqa: E402
    confusion_cell_detail,
    crisis_baseline_split,
    crisis_party_split,
    merge_overlapping_topics,
    normalize_radar,
    parse_tfidf_terms,
    pmqs_split_by_pm,
)
from hansard_pm_nlp.event_study import CRISIS_WINDOWS  # noqa: E402
from hansard_pm_nlp.lda import get_top_words  # noqa: E402

PM_COLORS = {
    "Boris Johnson": "#1f77b4",
    "Liz Truss": "#ff7f0e",
    "Rishi Sunak": "#2ca02c",
    "Keir Starmer": "#d62728",
}
TOPIC_LINE_COLOR = "#4fc3f7"

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
def load_mtld_over_time() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "mtld_over_time.parquet")


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
    n_by_pm = profile.set_index("pm_name")["n_contributions"]

    if not selected_pms:
        st.info("Select at least one PM to display the radar chart.")
    else:
        normalized = normalize_radar(profile, cols, selected_pms)

        fig = go.Figure()
        for pm in selected_pms:
            values = normalized.loc[pm, cols].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    name=f"{pm} (n={n_by_pm[pm]})",
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
            "Each axis min-max normalized across the *selected* PMs only "
            "(0=lowest, 1=highest among them) - deselect a PM below to "
            "rescale the others rather than leaving the axes compressed by "
            "a PM that's no longer even drawn. Legend shows each PM's "
            "contribution count: Liz Truss's is 123 vs. 2,195-5,459 for "
            "the other three (a seventeenth to a forty-fourth), over her "
            "49-day tenure, so her line is more sample-size noise than "
            "settled style - read it with that in mind. Type-token ratio "
            "(TTR) is left off entirely - it is length-biased (lexical.py) "
            "and Truss's tiny corpus (8,842 words vs 220k-520k) makes her "
            "TTR purely an artifact of sample size; MTLD is the length-"
            "corrected alternative shown here. TTR is in the raw values "
            "table below."
        )

    with st.expander("Raw values"):
        st.dataframe(profile[["pm_name", *cols, "type_token_ratio"]].set_index("pm_name"))

    st.subheader("Distinctive terms per PM (TF-IDF)")
    st.caption(
        "One document per PM, so a term's score reflects how distinctive it "
        "is to that PM *relative to the other three* (Phase 3, eda.py), not "
        "relative to general English. Parliamentary-address vocabulary (hon, "
        "right, friend, gentleman, house...) is filtered out via the same "
        "domain stopword list the LDA pipeline uses (preprocessing.py) - "
        "otherwise it would dominate every PM's chart identically and say "
        "nothing PM-specific. Read this as 'what stands out about this PM's "
        "usage compared to the other three', not as a topic summary."
    )
    term_cols = st.columns(2)
    for i, pm in enumerate(all_pms):
        terms = parse_tfidf_terms(profile.loc[profile["pm_name"] == pm, "top_tfidf_terms"].iloc[0])
        terms_df = pd.DataFrame(terms[:10], columns=["term", "score"]).sort_values("score")
        fig = px.bar(
            terms_df,
            x="score",
            y="term",
            orientation="h",
            title=pm,
            color_discrete_sequence=[PM_COLORS.get(pm)],
        )
        fig.update_layout(height=350, showlegend=False, margin={"t": 40})
        with term_cols[i % 2]:
            st.plotly_chart(_dark(fig), width="stretch")

    st.subheader("Lexical diversity (MTLD) over time")
    st.caption(
        "MTLD recomputed per PM per month (eda.py, build_mtld_over_time()), "
        "unlike the whole-corpus MTLD in the radar above - shows drift "
        "within a tenure rather than one settled number. Months under "
        "1,500 words are dropped rather than plotted, since MTLD gets "
        "noisy on too little text - the same length-sensitivity issue "
        "flagged for TTR, just smaller-scale. Liz Truss's 49-day tenure "
        "survives that floor for only 2 of her months; read her line as "
        "two data points, not a trend."
    )
    mtld_over_time = load_mtld_over_time()
    mtld_filtered = mtld_over_time[mtld_over_time["pm_name"].isin(selected_pms)]
    if mtld_filtered.empty:
        st.info("No PM selected, or no month has enough text to show.")
    else:
        fig_mtld = px.line(
            mtld_filtered.sort_values("period"),
            x="period",
            y="mtld",
            color="pm_name",
            markers=True,
            color_discrete_map=PM_COLORS,
            labels={"period": "Month", "mtld": "MTLD", "pm_name": "PM"},
        )
        fig_mtld.update_layout(height=450)
        st.plotly_chart(_dark(fig_mtld), width="stretch")

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

    st.subheader("Crisis vs. baseline")
    st.caption(
        "Same crisis dummies as Phase 7's OLS regression (event_study.py) - "
        "this is exactly what was tested, not a redrawn window. Ignores the "
        "date-range slider above (a period comparison needs both sides "
        "populated) but respects the PM selection."
    )
    box_df = event_df[event_df["pm_name"].isin(selected)]

    col_h2, col_h3 = st.columns(2)
    with col_h2:
        crisis_choice = st.selectbox(
            "Crisis (H2)", list(CRISIS_WINDOWS), format_func=lambda x: x.replace("_", " ")
        )
        if crisis_choice == "mini_budget":
            st.caption(
                "Mini-budget is a single sitting in this corpus (Liz Truss, "
                "2022-10-12) - the 'Crisis' box below is one point, not a "
                "distribution. phase7_event_study_report.md treats this "
                "window as uninterpretable given available data, not as "
                "evidence against H2 specifically."
            )
        split_h2 = crisis_baseline_split(box_df, f"crisis_{crisis_choice}", dv_choice)
        fig_h2 = px.box(split_h2, x="period", y="value", color="period", points="all")
        fig_h2.update_layout(height=420, showlegend=False, yaxis_title=dv_choice)
        st.plotly_chart(_dark(fig_h2), width="stretch")
    with col_h3:
        st.markdown("**Any crisis x party (H3)**")
        split_h3 = crisis_party_split(box_df, dv_choice)
        fig_h3 = px.box(split_h3, x="party", y="value", color="period", points="all")
        fig_h3.update_layout(height=420, yaxis_title=dv_choice)
        st.plotly_chart(_dark(fig_h3), width="stretch")

    st.caption(
        "Phase 7: no effect survives Benjamini-Hochberg correction for "
        "either H2 or H3. For Covid-19, Ukraine invasion, and the Labour "
        "leadership crisis, these boxes show why - heavy overlap between "
        "crisis and baseline. Mini-budget is a different problem, not "
        "overlap (see the note above when it's selected). H3 is "
        "additionally underpowered by construction: Labour's crisis side "
        "here is a single crisis window under a single PM (9 sittings vs. "
        "79 for the Conservative side)."
    )

    st.subheader("PMQs vs. other debates")
    st.caption(
        "Contribution-level, whole corpus - ignores the date-range slider "
        "above, respects the PM selection. PROJECT_SUMMARY.md flags PMQs as "
        "genre-built for evasiveness; on this hedging lexicon it's the "
        "opposite (affect.py, affect_report.md) - PMQs hedges *less* than "
        "other debates corpus-wide, which may mean the lexicon misses "
        "PMQs-specific evasion (redirecting to the other party) rather than "
        "PMs being less evasive there. Below: does that hold for every PM, "
        "or is it driven by one of them? Hover a bar for its N - Liz "
        "Truss's PMQs/other split is far thinner than the other three PMs' "
        "(dozens vs. thousands of contributions per group), so treat any "
        "reversal for her as sample-size noise before reading it as a "
        "genre effect."
    )
    pmqs_metric = st.radio(
        "Metric",
        ["mean_hedge_rate", "mean_net_certainty"],
        format_func=lambda x: {
            "mean_hedge_rate": "Hedging rate",
            "mean_net_certainty": "Net certainty",
        }[x],
        horizontal=True,
        key="pmqs_metric",
    )
    affect_contributions = load_affect()
    affect_selected = affect_contributions[affect_contributions["pm_name"].isin(selected)]
    pmqs_by_pm = pmqs_split_by_pm(affect_selected)
    pmqs_by_pm["Debate type"] = pmqs_by_pm["is_pmqs"].map({True: "PMQs", False: "Other"})
    fig_pmqs = px.bar(
        pmqs_by_pm.sort_values("pm_name"),
        x="pm_name",
        y=pmqs_metric,
        color="Debate type",
        barmode="group",
        hover_data={"n_contributions": True},
    )
    fig_pmqs.update_layout(height=420, xaxis_title="", yaxis_title=pmqs_metric)
    st.plotly_chart(_dark(fig_pmqs), width="stretch")

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
        "one duplicated topic. One small panel per topic rather than 13 stacked "
        "colors - past 8-10 categories a shared legend stops being readable, and "
        "a stacked area hides whether a topic is rising or falling behind a "
        "moving baseline. All panels share the same y-axis, so panel heights "
        "are directly comparable. Grey bands mark crisis windows (Phase 7); "
        "dotted lines mark PM transitions - repeated in every panel. Full topic "
        "word lists are in the expander below the heatmap."
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

    if filtered.empty:
        st.info("No PM selected.")
    else:
        merged = merge_overlapping_topics(filtered[topic_cols], topic_labels, MERGED_LABEL)
        merged["sitting_date"] = filtered["sitting_date"].values

        monthly = merged.set_index("sitting_date").resample("MS").mean().dropna(how="all")

        topics = list(monthly.columns)
        n_cols = 4
        n_rows = -(-len(topics) // n_cols)  # ceil division
        panel_titles = [t.split(":")[0].strip() for t in topics]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=panel_titles,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.5 / n_rows,
            horizontal_spacing=0.03,
        )

        chart_start, chart_end = monthly.index.min(), monthly.index.max()
        pm_transitions = filtered.groupby("pm_name")["sitting_date"].min().sort_values()
        y_max = monthly.max().max() * 1.1

        for i, topic in enumerate(topics):
            row, col = divmod(i, n_cols)
            row, col = row + 1, col + 1
            fig.add_trace(
                go.Scatter(
                    x=monthly.index,
                    y=monthly[topic],
                    mode="lines",
                    line={"color": TOPIC_LINE_COLOR, "width": 1.6},
                    fill="tozeroy",
                    fillcolor="rgba(79,195,247,0.15)",
                    showlegend=False,
                    hovertemplate="%{x|%b %Y}: %{y:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(range=[0, y_max], row=row, col=col, showticklabels=(col == 1))

        for _, (start, end) in CRISIS_WINDOWS.items():
            if pd.Timestamp(end) >= chart_start and pd.Timestamp(start) <= chart_end:
                fig.add_vrect(
                    x0=start, x1=end, fillcolor="grey", opacity=0.15, line_width=0,
                    row="all", col="all",
                )
        for pm_name, start_date in pm_transitions.iloc[1:].items():
            fig.add_vline(
                x=start_date, line_dash="dot", line_color="rgba(255,255,255,0.35)",
                row="all", col="all",
            )

        fig.update_layout(height=185 * n_rows, margin={"t": 30, "b": 10})
        for annotation in fig.layout.annotations:
            annotation.font.size = 11
        st.plotly_chart(_dark(fig), width="stretch")

    st.subheader("Mean topic weight by PM")
    st.caption(
        "Same 13 merged topics, averaged across each PM's sittings instead "
        "of over time - a cross-sectional complement to the area chart "
        "above: which topics a PM leans into overall, not when. Row labels "
        "show each PM's sitting count - Liz Truss's row averages far fewer "
        "sittings than the other three, so read her cells as noisier."
    )
    cross_sectional = merge_overlapping_topics(filtered[topic_cols], topic_labels, MERGED_LABEL)
    cross_sectional["pm_name"] = filtered["pm_name"].values
    pm_topic_means = cross_sectional.groupby("pm_name").mean()
    n_sittings = filtered.groupby("pm_name").size()
    pm_topic_means.index = [f"{pm} (n={n_sittings[pm]})" for pm in pm_topic_means.index]

    fig_heat = px.imshow(
        pm_topic_means,
        labels={"x": "Topic", "y": "PM", "color": "Mean weight"},
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig_heat.update_layout(height=350)
    st.plotly_chart(_dark(fig_heat), width="stretch")

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
    cm_event = st.plotly_chart(
        _dark(fig),
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="cm_chart",
    )

    cm_points = cm_event.selection.points
    if cm_points:
        actual_pm, predicted_pm = cm_points[0]["y"], cm_points[0]["x"]
        detail = confusion_cell_detail(predictions, model_choice, actual_pm, predicted_pm)
        st.caption(
            f"{len(detail)} sitting(s) where the actual speaker was **{actual_pm}** "
            f"and the model predicted **{predicted_pm}**:"
        )
        st.dataframe(detail, hide_index=True, width="stretch")
    else:
        st.caption("Click a cell above to see which sittings it contains.")

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
