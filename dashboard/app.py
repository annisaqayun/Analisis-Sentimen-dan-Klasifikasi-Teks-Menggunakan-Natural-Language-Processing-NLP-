# Wine Quality AI - Dashboard v4.1 (Premium Polish)

import streamlit as st
import pandas as pd
import time
import os

from utils.preprocessing import preprocess_text, get_preprocessing_steps
from utils.prediction import (
    load_model, load_vectorizer, load_model_spec,
    predict_quality, get_feature_importance,
)
from utils.visualization import (
    create_probability_bar_chart, create_probability_pie_chart,
    create_confidence_gauge, create_feature_importance_chart,
    create_metrics_bar_chart, create_confusion_matrix_illustration,
    create_class_distribution_chart,
)
from utils.helpers import (
    EXAMPLE_DESCRIPTIONS, get_quality_meta,
    add_to_history, history_to_dataframe,
    export_csv, generate_pdf,
)

# ---- Page config ----
st.set_page_config(
    page_title="Wine Quality AI",
    page_icon="https://em-content.zobj.net/source/twitter/376/wine-glass_1f377.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS ----
_css = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
if os.path.exists(_css):
    with open(_css) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- Session state ----
for _k, _v in [("ta_input", ""), ("prediction_history", []),
                ("last_result", None), ("last_steps", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---- Load model ----
try:
    model = load_model()
    vectorizer = load_vectorizer()
    spec = load_model_spec()
    import json
    import os
    try:
        with open('evaluation_results.json', 'r') as f:
            _eval_data = json.load(f)
            _perf = _eval_data['performance']
            if 'best_performance_metrics' not in spec:
                spec['best_performance_metrics'] = {}
            # Convert percentages back to decimals for the UI formatter
            spec['best_performance_metrics']['Accuracy'] = _perf.get('Accuracy', 0) / 100.0
            spec['best_performance_metrics']['Precision'] = _perf.get('Precision', 0) / 100.0
            spec['best_performance_metrics']['Recall'] = _perf.get('Recall', 0) / 100.0
            spec['best_performance_metrics']['F1-Score'] = _perf.get('F1-Score', 0) / 100.0
    except Exception as e:
        print('Failed to load evaluation results:', e)

    _ok = True
except Exception as e:
    _ok = False
    _err = str(e)

# ---- Callbacks ----
def _set_high():
    st.session_state.ta_input = EXAMPLE_DESCRIPTIONS["high"]

def _set_medium():
    st.session_state.ta_input = EXAMPLE_DESCRIPTIONS["medium"]

def _set_low():
    st.session_state.ta_input = EXAMPLE_DESCRIPTIONS["low"]

def _clear():
    st.session_state.ta_input = ""

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Wine Quality AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">NLP-powered quality prediction</div>', unsafe_allow_html=True)

    if _ok:
        st.markdown('<div class="sidebar-status">Model loaded</div>', unsafe_allow_html=True)

        st.markdown("## Model")
        st.markdown(f"**{spec.get('model_type', 'DecisionTreeClassifier')}**")

        st.markdown("## Performance")
        metrics = spec.get("best_performance_metrics", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{metrics.get("Accuracy",0):.1%}</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{metrics.get("Precision",0):.1%}</div><div class="stat-label">Precision</div></div>', unsafe_allow_html=True)
        st.markdown("")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{metrics.get("Recall",0):.1%}</div><div class="stat-label">Recall</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{metrics.get("F1-Score",0):.1%}</div><div class="stat-label">F1-Score</div></div>', unsafe_allow_html=True)

        st.markdown("## Pipeline")
        for i, s in enumerate(spec.get("preprocessing_steps", []), 1):
            st.markdown(f"{i}. {s}")

        st.markdown("## Vectorizer")
        vec = spec.get("vectorizer", {})
        st.markdown(f"Type: **{vec.get('type', 'TfidfVectorizer')}**")
        st.markdown(f"Features: **{vec.get('max_features', 5000):,}**")

        st.markdown("## Parameters")
        p = spec.get("model_parameters", {})

        try:
            with open('evaluation_results.json', 'r') as f:
                import json
                eval_data = json.load(f)
                perf_metrics = eval_data['performance']
        except Exception:
            eval_data = None
            perf_metrics = spec.get("best_performance_metrics", {})

        st.markdown(f"Alpha: `{p.get('alpha', 0.5)}`")
        st.markdown(f"Fit Prior: `{p.get('fit_prior', True)}`")
        st.markdown(f"Class Prior: `{p.get('class_prior', 'None')}`")
        

        st.markdown("## Classes")
        for cls in spec.get("classes", []):
            color = get_quality_meta(cls)["color"]
            st.markdown(
                f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
                f'background:{color};margin-right:6px;vertical-align:middle;'
                f'box-shadow:0 0 5px {color};"></span>'
                f'<span style="font-size:0.82rem;">{cls}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.error(f"Model error: {_err}")

    st.markdown("---")
    st.markdown('<div class="app-footer">v4.1</div>', unsafe_allow_html=True)

if not _ok:
    st.error("Model files not found. Ensure .pkl files exist in the project root.")
    st.stop()

# ---- Header ----
n_preds = len(st.session_state.prediction_history)
st.markdown(f"""<div class="app-header">
    <div class="header-strip"></div>
    <h1>Wine Quality Prediction</h1>
    <div class="subtitle">Analyse wine descriptions with NLP to predict quality class</div>
    <div class="badge-row">
        <span class="badge">Naive Bayes + TF-IDF</span>
        <span class="badge-ghost">{n_preds} prediction{'s' if n_preds != 1 else ''} this session</span>
    </div>
</div>""", unsafe_allow_html=True)

# ---- Input section ----
st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)

cols = st.columns([1, 1, 1, 0.55])
with cols[0]:
    st.button("High quality sample", key="s_high", on_click=_set_high, use_container_width=True)
with cols[1]:
    st.button("Medium quality sample", key="s_med", on_click=_set_medium, use_container_width=True)
with cols[2]:
    st.button("Low quality sample", key="s_low", on_click=_set_low, use_container_width=True)
with cols[3]:
    st.button("Clear", key="s_clear", on_click=_clear, use_container_width=True)

user_input = st.text_area(
    "Wine description", height=140, key="ta_input",
    placeholder="Describe a wine, e.g. This elegant Pinot Noir offers layers of cherry, raspberry, and subtle oak with a silky finish...",
    label_visibility="collapsed",
)

chars = len(user_input)
words = len(user_input.split()) if user_input.strip() else 0
st.caption(f"{chars:,} characters  |  {words} words")

predict_clicked = st.button("Run prediction", type="primary", use_container_width=True, key="btn_predict")

# ---- Prediction logic ----
if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter a wine description.")
    elif words < 3:
        st.warning("Description too short - use at least 3 words.")
    else:
        progress = st.empty()
        status = st.empty()
        stages = [
            ("Cleaning text...", 0.12),
            ("Removing stopwords...", 0.28),
            ("Applying stemming...", 0.44),
            ("Vectorising with TF-IDF...", 0.64),
            ("Running Decision Tree inference...", 0.84),
            ("Generating results...", 1.0),
        ]
        bar = progress.progress(0)
        for label, pct in stages:
            status.caption(label)
            bar.progress(pct)
            time.sleep(0.16)
        try:
            processed = preprocess_text(user_input)
            steps = get_preprocessing_steps(user_input)
            result = predict_quality(processed, model, vectorizer)
            st.session_state.last_result = result
            st.session_state.last_steps = steps
            add_to_history(
                st.session_state.prediction_history,
                user_input, result["prediction"], result["confidence"],
            )
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Prediction failed: {exc}")
            st.stop()
        progress.empty()
        status.empty()
        st.rerun()

# ---- Results display ----
if st.session_state.last_result is not None:
    res = st.session_state.last_result
    steps = st.session_state.last_steps
    meta = get_quality_meta(res["prediction"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="result-card {meta['css_class']}">
        <div class="result-row">
            <div class="result-dot"></div>
            <div class="result-label">{meta['label']}</div>
            <div class="result-conf">{res['confidence']:.1%} confidence</div>
        </div>
        <div class="result-insight">{meta['insight']}</div>
    </div>""", unsafe_allow_html=True)

    # -- Probability charts --
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Probability Analysis</div>', unsafe_allow_html=True)

    classes = res["classes"]
    probs = res["probabilities_array"]

    t1, t2, t3 = st.tabs(["Bar Chart", "Distribution", "Confidence"])
    with t1:
        st.plotly_chart(create_probability_bar_chart(classes, probs), use_container_width=True, config={"displayModeBar": False})
    with t2:
        st.plotly_chart(create_probability_pie_chart(classes, probs), use_container_width=True, config={"displayModeBar": False})
    with t3:
        st.plotly_chart(create_confidence_gauge(res["confidence"], res["prediction"]), use_container_width=True, config={"displayModeBar": False})

    st.dataframe(
        pd.DataFrame({"Class": classes, "Probability": [f"{p:.2%}" for p in probs]}),
        use_container_width=True, hide_index=True,
    )

    # -- NLP Pipeline --
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">NLP Pipeline</div>', unsafe_allow_html=True)

    with st.expander("View preprocessing steps", expanded=False):
        if steps:
            pipe = [
                ("01 -- Raw input", steps["raw_text"]),
                ("02 -- Lowercase", steps["after_lowercase"]),
                ("03 -- Punctuation removed", steps["after_punctuation_removal"]),
                ("04 -- Numbers removed", steps["after_numeric_removal"]),
                ("05 -- Stopwords removed", " ".join(steps["tokens_after_stopwords"]) + f'  ({len(steps["stopwords_removed"])} removed)'),
                ("06 -- Stemmed output", " ".join(steps["tokens_after_stemming"]) + f'  ({steps["token_count"]} tokens, {steps["unique_token_count"]} unique)'),
            ]
            for lbl, content in pipe:
                st.markdown(
                    f'<div class="pipe-step"><div class="pipe-label">{lbl}</div>'
                    f'<div class="pipe-text">{content}</div></div>',
                    unsafe_allow_html=True,
                )

            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric("Total tokens", steps["token_count"])
            with tc2:
                st.metric("Unique tokens", steps["unique_token_count"])
            with tc3:
                st.metric("Stopwords removed", len(steps["stopwords_removed"]))

    # -- Export --
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        pdf_data = generate_pdf(res, steps)
        if pdf_data:
            st.download_button(
                "Download PDF report", data=pdf_data,
                file_name="wine_prediction_report.pdf",
                mime="application/pdf", use_container_width=True,
            )
    with e2:
        if st.session_state.prediction_history:
            st.download_button(
                "Download history CSV",
                data=export_csv(st.session_state.prediction_history),
                file_name="prediction_history.csv",
                mime="text/csv", use_container_width=True,
            )

# ---- Feature importance ----
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

with st.expander("Top influential words in the model", expanded=False):
    fi = get_feature_importance(model, vectorizer, top_n=20)
    if fi["features"]:
        st.plotly_chart(
            create_feature_importance_chart(fi["features"], fi["importances"]),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.caption("Words ranked by Decision Tree feature importance. Higher values indicate stronger influence on classification.")
    else:
        st.info("No significant features detected.")

# ---- History ----
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Prediction History</div>', unsafe_allow_html=True)

if st.session_state.prediction_history:
    st.dataframe(history_to_dataframe(st.session_state.prediction_history), use_container_width=True)
    if st.button("Clear history", key="btn_clear_hist"):
        st.session_state.prediction_history = []
        st.session_state.last_result = None
        st.session_state.last_steps = None
        st.rerun()
else:
    st.caption("No predictions yet. Enter a description and run a prediction.")

# ---- Analytics ----
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Model Analytics</div>', unsafe_allow_html=True)

at1, at2, at3 = st.tabs(["Performance", "Confusion Matrix", "Class Distribution"])

with at1:
    m = spec.get("best_performance_metrics", {})
    cm = {
        "Accuracy": m.get("Accuracy", 0),
        "Precision": m.get("Precision", 0),
        "Recall": m.get("Recall", 0),
        "F1-Score": m.get("F1-Score", 0),
    }
    mc = st.columns(4)
    for i, (name, val) in enumerate(cm.items()):
        with mc[i]:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{val:.1%}</div>'
                f'<div class="stat-label">{name}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.plotly_chart(create_metrics_bar_chart(cm), use_container_width=True, config={"displayModeBar": False})

with at2:
    st.plotly_chart(
        create_confusion_matrix_illustration(spec.get("classes", ["Low", "Medium", "High"])),
        use_container_width=True, config={"displayModeBar": False},
    )
    st.caption("Illustrative confusion matrix based on ~68.5% model accuracy (Naive Bayes).")

with at3:
    st.plotly_chart(
        create_class_distribution_chart(spec.get("classes", ["Low", "Medium", "High"])),
        use_container_width=True, config={"displayModeBar": False},
    )
    st.caption("Approximate class distribution in the training dataset.")

# ---- Footer ----
st.markdown(
    '<div class="app-footer">Wine Quality AI  |  NLP  |  TF-IDF  |  Decision Tree  |  Streamlit</div>',
    unsafe_allow_html=True,
)
