import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _render_confusion_matrix(cm, title="Confusion Matrix", labels=None):
    cm_arr = np.array(cm)
    if labels is None:
        labels = [str(i) for i in range(cm_arr.shape[0])]
    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale="Blues",
        text=cm_arr,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        height=260,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def render_analytics_dashboard():
    st.markdown("## Analytics Dashboard")
    st.markdown("Tracks BLEU, ROUGE, and METEOR scores for generation tasks, plus Model A & B classifier metrics.")

    eval_path = os.path.join(PROJECT_ROOT, "data", "processed", "eval_metrics.json")
    eval_metrics = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_metrics = json.load(f)
        st.success(f"Pre-computed evaluation metrics loaded from {eval_path}")
    else:
        st.info("No pre-computed metrics found. Run `python src/evaluate.py` to generate them.")

    if eval_metrics:
        st.markdown("### Question generation quality (BLEU / ROUGE / METEOR)")
        st.caption("Metrics measured on the RACE test set.")

        metric_data = {
            "Metric": ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"],
            "Score": [
                eval_metrics.get("bleu1", 0),
                eval_metrics.get("bleu2", 0),
                eval_metrics.get("rouge1", 0),
                eval_metrics.get("rouge2", 0),
                eval_metrics.get("rougeL", 0),
                eval_metrics.get("meteor", 0),
            ],
        }

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                metric_data,
                x="Metric",
                y="Score",
                title=f"Generation metrics — {eval_metrics.get('label', 'Model A')}",
                color="Metric",
                color_discrete_sequence=px.colors.qualitative.Set2,
                range_y=[0, 1],
                text_auto=".3f",
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Score summary**")
            for name, score in zip(metric_data["Metric"], metric_data["Score"]):
                st.metric(label=name, value=f"{score:.4f}")

    st.markdown("---")

    comp_path = os.path.join(PROJECT_ROOT, "data", "processed", "model_a_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comp_data = json.load(f)

        st.markdown("### Model A — Classifier comparison (answer verification)")
        st.caption("Accuracy, Precision, Recall, and Macro F1 on the validation set.")

        supervised_keys = ["Logistic Regression", "SVM (calibrated)", "Ensemble (LR+SVM+NB)"]
        rows = []
        for name in supervised_keys:
            if name in comp_data:
                m = comp_data[name]
                rows.append({
                    "Model": name,
                    "Accuracy": round(m.get("accuracy", 0), 4),
                    "F1 (macro)": round(m.get("f1_macro", 0), 4),
                    "Precision": round(m.get("precision", 0), 4),
                    "Recall": round(m.get("recall", 0), 4),
                })

        if rows:
            comp_df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(comp_df, use_container_width=True)

            fig2 = go.Figure()
            metrics_to_plot = ["Accuracy", "F1 (macro)", "Precision", "Recall"]
            colors = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373"]
            for metric, color in zip(metrics_to_plot, colors):
                fig2.add_trace(go.Bar(
                    name=metric,
                    x=comp_df.index.tolist(),
                    y=comp_df[metric].tolist(),
                    text=[f"{v:.4f}" for v in comp_df[metric]],
                    textposition="outside",
                    marker_color=color,
                ))
            fig2.update_layout(
                barmode="group",
                title="Model A — Supervised Classifier Comparison",
                yaxis=dict(range=[0, 1.1], title="Score"),
                height=380,
                margin=dict(t=60, b=20),
                legend_title="Metric",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Unsupervised & Semi-Supervised results")
        unsup_rows = []
        km = comp_data.get("K-Means (unsupervised)", {})
        lp = comp_data.get("Label Propagation (semi-sup)", {})
        if km:
            unsup_rows.append({
                "Model": "K-Means (unsupervised)",
                "Silhouette": round(km.get("silhouette", 0), 4),
                "Purity": round(km.get("purity", 0), 4),
                "Inertia": round(km.get("inertia", 0), 2),
                "Semi-sup F1": "—",
                "Semi-sup Acc": "—",
            })
        if lp:
            unsup_rows.append({
                "Model": "Label Propagation (semi-sup)",
                "Silhouette": "—",
                "Purity": "—",
                "Inertia": "—",
                "Semi-sup F1": round(lp.get("f1_macro", 0), 4),
                "Semi-sup Acc": round(lp.get("accuracy", 0), 4),
            })
        if unsup_rows:
            st.dataframe(pd.DataFrame(unsup_rows).set_index("Model"), use_container_width=True)

        st.markdown("#### Confusion matrices (binary: correct / incorrect option)")
        cm_cols = st.columns(len(supervised_keys))
        for col, name in zip(cm_cols, supervised_keys):
            if name in comp_data:
                cm = comp_data[name].get("confusion_matrix")
                if cm:
                    with col:
                        fig_cm = _render_confusion_matrix(
                            cm, title=name.split(" (")[0], labels=["Wrong", "Correct"]
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("No Model A comparison data found. Run `python src/model_a_train.py` to generate it.")

    st.markdown("---")

    mb_path = os.path.join(PROJECT_ROOT, "data", "processed", "model_b_metrics.json")
    if os.path.exists(mb_path):
        with open(mb_path) as f:
            mb_data = json.load(f)

        st.markdown("### Model B — Distractor & Hint metrics")

        rf_m = mb_data.get("rf_distractor_ranker", {})
        lr_h_m = mb_data.get("lr_hint_scorer", {})

        mb_rows = []
        if rf_m:
            mb_rows.append({
                "Model": "RF Distractor Ranker",
                "Accuracy": round(rf_m.get("accuracy", 0), 4),
                "F1 (macro)": round(rf_m.get("f1_macro", 0), 4),
                "Precision": round(rf_m.get("precision", 0), 4),
                "Recall": round(rf_m.get("recall", 0), 4),
                "R² Score": "—",
            })
        if lr_h_m:
            mb_rows.append({
                "Model": "LR Hint Scorer",
                "Accuracy": round(lr_h_m.get("accuracy", 0), 4),
                "F1 (macro)": round(lr_h_m.get("f1_macro", 0), 4),
                "Precision": round(lr_h_m.get("precision", 0), 4),
                "Recall": round(lr_h_m.get("recall", 0), 4),
                "R² Score": round(lr_h_m.get("r2_score", 0), 4),
            })

        if mb_rows:
            mb_df = pd.DataFrame(mb_rows).set_index("Model")
            st.dataframe(mb_df, use_container_width=True)
            if lr_h_m.get("r2_score") is not None:
                st.caption(
                    f"R² = {lr_h_m['r2_score']:.4f} — measures how well the hint scorer's "
                    "predicted relevance probabilities correlate with true relevance labels."
                )

        st.markdown("#### Confusion matrices")
        cm_b_cols = st.columns(2)
        if rf_m.get("confusion_matrix"):
            with cm_b_cols[0]:
                fig_rf_cm = _render_confusion_matrix(
                    rf_m["confusion_matrix"], title="RF Distractor Ranker",
                    labels=["Bad", "Good"]
                )
                st.plotly_chart(fig_rf_cm, use_container_width=True)
        if lr_h_m.get("confusion_matrix"):
            with cm_b_cols[1]:
                fig_lh_cm = _render_confusion_matrix(
                    lr_h_m["confusion_matrix"], title="LR Hint Scorer",
                    labels=["Irrelevant", "Relevant"]
                )
                st.plotly_chart(fig_lh_cm, use_container_width=True)
    else:
        st.info("No Model B metrics found. Run `python src/model_b_train.py` to generate them.")

    st.markdown("---")

    st.markdown("### Session log")
    if "session_log" not in st.session_state or not st.session_state.session_log:
        st.info("No sessions recorded yet. Complete a quiz to log results here.")
    else:
        log_df = pd.DataFrame(st.session_state.session_log)
        st.dataframe(log_df, use_container_width=True)

        if len(log_df) > 1:
            metrics_in_log = [c for c in ["bleu1", "rougeL", "meteor"] if c in log_df.columns]
            if metrics_in_log:
                fig3 = px.line(
                    log_df,
                    x=log_df.index,
                    y=metrics_in_log,
                    title="Generation Quality Over Sessions",
                    labels={"index": "Session", "value": "Score"},
                    markers=True,
                )
                fig3.update_layout(height=300, margin=dict(t=50, b=20))
                st.plotly_chart(fig3, use_container_width=True)

        csv_data = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export session log (CSV)",
            data=csv_data,
            file_name="session_log.csv",
            mime="text/csv",
        )

    st.markdown("---")

    st.markdown("### Inference latency")
    if "session_log" in st.session_state and st.session_state.session_log:
        log_df = pd.DataFrame(st.session_state.session_log)
        if "latency_ms" in log_df.columns:
            avg_lat = log_df["latency_ms"].mean()
            max_lat = log_df["latency_ms"].max()
            col_l1, col_l2 = st.columns(2)
            col_l1.metric("Average latency", f"{avg_lat:.1f} ms")
            col_l2.metric("Max latency", f"{max_lat:.1f} ms")
            fig4 = px.histogram(
                log_df,
                x="latency_ms",
                nbins=20,
                title="Latency Distribution (ms)",
                color_discrete_sequence=["#4fc3f7"],
            )
            fig4.update_layout(height=260, margin=dict(t=50, b=20))
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Latency data will appear here after running quiz sessions.")

    exp_path = os.path.join(PROJECT_ROOT, "data", "processed", "experiment_results.csv")
    if os.path.exists(exp_path):
        st.markdown("---")
        st.markdown("### Experiment comparison (experiments.ipynb)")
        exp_df = pd.read_csv(exp_path, index_col=0)
        st.dataframe(exp_df, use_container_width=True)