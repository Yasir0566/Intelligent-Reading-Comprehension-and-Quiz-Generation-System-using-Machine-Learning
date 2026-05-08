import os
import sys
import time
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from preprocessing import (
    BASE,
    clean_text,
    generate_questions_for_row,
)

from model_b_train import generate_distractors, generate_hints

DIR_A = os.path.join(BASE, "models", "model_a", "traditional")
DIR_B = os.path.join(BASE, "models", "model_b", "traditional")


def load_model_a_artifacts():
    required = [
        os.path.join(DIR_A, "tfidf_vectorizer.pkl"),
        os.path.join(DIR_A, "ensemble_verifier.pkl"),
        os.path.join(DIR_A, "lr_verifier.pkl"),
        os.path.join(DIR_A, "svm_verifier.pkl"),
        os.path.join(DIR_A, "nb_qtype.pkl"),
        os.path.join(DIR_A, "tfidf_qtype.pkl"),
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run model_a_train.py first."
            )

    return {
        "tfidf": joblib.load(os.path.join(DIR_A, "tfidf_vectorizer.pkl")),
        "ensemble": joblib.load(os.path.join(DIR_A, "ensemble_verifier.pkl")),
        "lr": joblib.load(os.path.join(DIR_A, "lr_verifier.pkl")),
        "svm": joblib.load(os.path.join(DIR_A, "svm_verifier.pkl")),
        "nb": joblib.load(os.path.join(DIR_A, "nb_qtype.pkl")),
        "tfidf_qtype": joblib.load(os.path.join(DIR_A, "tfidf_qtype.pkl")),
    }


def load_model_b_artifacts():
    required = [
        os.path.join(DIR_B, "tfidf_modelb.pkl"),
        os.path.join(DIR_B, "rf_distractor_ranker.pkl"),
        os.path.join(DIR_B, "lr_hint_scorer.pkl"),
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run model_b_train.py first."
            )

    return {
        "tfidf": joblib.load(os.path.join(DIR_B, "tfidf_modelb.pkl")),
        "rf": joblib.load(os.path.join(DIR_B, "rf_distractor_ranker.pkl")),
        "lr_hint": joblib.load(os.path.join(DIR_B, "lr_hint_scorer.pkl")),
    }


def verify_answer(article, question, options, artifacts_a):
    t0 = time.time()
    tfidf = artifacts_a["tfidf"]
    ensemble = artifacts_a["ensemble"]
    nb = artifacts_a["nb"]
    tfidf_q = artifacts_a["tfidf_qtype"]

    art_c = clean_text(article)
    q_c = clean_text(question)
    labels = ["A", "B", "C", "D"]

    texts = [
        f"{art_c} {art_c} {q_c} {clean_text(str(options[o]))}"
        for o in labels
    ]

    X_tfidf = tfidf.transform(texts)
    art_vec = tfidf.transform([art_c])
    q_vec = tfidf.transform([q_c])
    cos_feats = np.array([
        [
            cosine_similarity(art_vec, tfidf.transform([clean_text(str(options[o]))]))[0, 0],
            cosine_similarity(q_vec, tfidf.transform([clean_text(str(options[o]))]))[0, 0],
            cosine_similarity(art_vec, q_vec)[0, 0],
        ]
        for o in labels
    ], dtype=np.float32)

    X = hstack([X_tfidf, csr_matrix(cos_feats)])
    model_scores = ensemble.predict_proba(X)[:, 1]
    lexical_scores = 0.65 * cos_feats[:, 1] + 0.35 * cos_feats[:, 0]

    def _minmax(values):
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax - vmin < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return (values - vmin) / (vmax - vmin)

    model_norm = _minmax(model_scores)
    lexical_norm = _minmax(lexical_scores)

    prob_range = float(model_scores.max() - model_scores.min())
    if prob_range < 0.08:
        blended_scores = 0.45 * model_norm + 0.55 * lexical_norm
    else:
        blended_scores = 0.75 * model_norm + 0.25 * lexical_norm

    logits = blended_scores - float(np.max(blended_scores))
    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    q_type = nb.predict(tfidf_q.transform([q_c]))[0]

    return {
        "predicted_answer": labels[int(np.argmax(probs))],
        "probabilities": {labels[i]: float(probs[i]) for i in range(4)},
        "question_type": q_type,
        "latency_ms": (time.time() - t0) * 1000,
    }


def generate_question(row, artifacts_a, top_k=3):
    vectorizer = artifacts_a["tfidf"]
    return generate_questions_for_row(row, vectorizer, top_k=top_k)


__all__ = [
    "load_model_a_artifacts",
    "load_model_b_artifacts",
    "verify_answer",
    "generate_question",
    "generate_distractors",
    "generate_hints",
]