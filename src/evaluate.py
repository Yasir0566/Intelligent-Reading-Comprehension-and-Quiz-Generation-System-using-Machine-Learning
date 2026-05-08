import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rs

from preprocessing import BASE, load_race_data, generate_questions_for_row
import joblib


def compute_bleu(references, hypotheses, n=1):
    smoother = SmoothingFunction().method1
    refs_tokenized = [[ref.lower().split()] for ref in references]
    hyps_tokenized = [hyp.lower().split() for hyp in hypotheses]
    weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
    score = corpus_bleu(
        refs_tokenized, hyps_tokenized,
        weights=weights,
        smoothing_function=smoother,
    )
    return round(float(score), 4)


def compute_rouge(references, hypotheses):
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1": round(float(np.mean(r1)), 4),
        "rouge2": round(float(np.mean(r2)), 4),
        "rougeL": round(float(np.mean(rl)), 4),
    }


def compute_meteor(references, hypotheses):
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_s = str(ref).strip().lower()
        hyp_s = str(hyp).strip().lower()
        if not ref_s and not hyp_s:
            scores.append(1.0)
            continue
        # If identical strings, treat as perfect match to avoid
        # tokenizer/weighting edge-cases in METEOR implementation.
        if ref_s == hyp_s:
            scores.append(1.0)
            continue
        scores.append(meteor_score([ref_s.split()], hyp_s.split()))

    return round(float(np.mean(scores)), 4)


def evaluate_generation(references, hypotheses, label="Model", save_path=None):
    assert len(references) == len(hypotheses), (
        f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps"
    )

    bleu1 = compute_bleu(references, hypotheses, n=1)
    bleu2 = compute_bleu(references, hypotheses, n=2)
    rouge = compute_rouge(references, hypotheses)
    meteor = compute_meteor(references, hypotheses)

    metrics = {
        "label": label,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "meteor": meteor,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved → {save_path}")

    return metrics


def plot_generation_metrics(metrics_dict, title="Generation Metrics"):
    import plotly.express as px

    labels = ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
    values = [
        metrics_dict["bleu1"],
        metrics_dict["bleu2"],
        metrics_dict["rouge1"],
        metrics_dict["rouge2"],
        metrics_dict["rougeL"],
        metrics_dict["meteor"],
    ]
    fig = px.bar(
        x=labels, y=values,
        title=title,
        labels={"x": "Metric", "y": "Score"},
        range_y=[0, 1],
        color=labels,
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False)
    return fig


def run_evaluation(n_samples=500):
    print("=" * 55)
    print("EVALUATION PIPELINE")
    print("=" * 55)

    tfidf_path = os.path.join(BASE, "models", "model_a", "traditional", "tfidf_vectorizer.pkl")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(
            f"Missing: {tfidf_path}\n"
            "Run model_a_train.py first."
        )
    vectorizer = joblib.load(tfidf_path)

    print(f"\nLoading test data ({n_samples} samples)...")
    test_df = load_race_data("test").head(n_samples)

    references, hypotheses = [], []

    print("Generating questions...")
    for _, row in test_df.iterrows():
        reference = str(row["question"])
        generated = generate_questions_for_row(row, vectorizer, top_k=3)
        references.append(reference)
        hypotheses.append(generated[0])

    save_path = os.path.join(BASE, "data", "processed", "eval_metrics.json")
    metrics = evaluate_generation(
        references, hypotheses,
        label="Model A",
        save_path=save_path,
    )

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    for k, v in metrics.items():
        if k != "label":
            print(f"  {k:<10}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    run_evaluation(n_samples=500)