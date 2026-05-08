import os
import sys
import joblib
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    BASE,
    load_race_data,
    clean_text,
    generate_questions_for_row,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, silhouette_score,
)

SAVE_DIR = os.path.join(BASE, "models", "model_a", "traditional")
os.makedirs(SAVE_DIR, exist_ok=True)


def detect_question_type(question):
    q = str(question).strip().lower()
    for wh in ["what", "who", "where", "when", "why", "how", "which"]:
        if q.startswith(wh):
            return wh
    return "other"


def load_saved_features():
    processed_dir = os.path.join(BASE, "data", "processed")
    paths = {
        "X_train": os.path.join(processed_dir, "X_train.npz"),
        "X_val": os.path.join(processed_dir, "X_val.npz"),
        "y_train": os.path.join(processed_dir, "y_train.npy"),
        "y_val": os.path.join(processed_dir, "y_val.npy"),
    }
    for path in paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run preprocessing.py first:  python src/preprocessing.py"
            )

    return (
        sp.load_npz(paths["X_train"]),
        sp.load_npz(paths["X_val"]),
        np.load(paths["y_train"]),
        np.load(paths["y_val"]),
    )


def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(
        max_iter=2000,
        C=0.5,
        class_weight={0: 1, 1: 3},
        solver="lbfgs",
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)

    preds = lr.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec = recall_score(y_val, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_val, preds).tolist()
    print(f"    Val Accuracy : {acc:.4f}")
    print(f"    Val Macro F1 : {f1:.4f}")
    print(f"    Val Precision: {prec:.4f}")
    print(f"    Val Recall   : {rec:.4f}")
    print(f"    Confusion Matrix:\n{confusion_matrix(y_val, preds)}")

    save_path = os.path.join(SAVE_DIR, "lr_verifier.pkl")
    joblib.dump(lr, save_path)
    print(f"    Saved → {save_path}")
    return lr, {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec, "confusion_matrix": cm}


def train_svm(X_train, y_train, X_val, y_val):
    print("\n--- Linear SVM (calibrated) ---")
    svm = CalibratedClassifierCV(
        LinearSVC(
            max_iter=3000,
            C=0.1,
            class_weight={0: 1, 1: 3},
        ),
        cv=3,
    )
    svm.fit(X_train, y_train)

    preds = svm.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec = recall_score(y_val, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_val, preds).tolist()
    print(f"    Val Accuracy : {acc:.4f}")
    print(f"    Val Macro F1 : {f1:.4f}")
    print(f"    Val Precision: {prec:.4f}")
    print(f"    Val Recall   : {rec:.4f}")
    print(f"    Confusion Matrix:\n{confusion_matrix(y_val, preds)}")

    save_path = os.path.join(SAVE_DIR, "svm_verifier.pkl")
    joblib.dump(svm, save_path)
    print(f"    Saved → {save_path}")
    return svm, {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec, "confusion_matrix": cm}


def train_naive_bayes(train_df, val_df, vectorizer, X_train=None, y_train=None, X_val=None, y_val=None):
    print("\n--- Naive Bayes (question-type classifier) ---")

    qt_train = [detect_question_type(q) for q in train_df["question"]]
    qt_val = [detect_question_type(q) for q in val_df["question"]]

    X_q_train = vectorizer.transform([clean_text(str(q)) for q in train_df["question"]])
    X_q_val = vectorizer.transform([clean_text(str(q)) for q in val_df["question"]])

    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_q_train, qt_train)

    acc = accuracy_score(qt_val, nb.predict(X_q_val))
    print(f"    Val Accuracy (q-type): {acc:.4f}")

    joblib.dump(nb, os.path.join(SAVE_DIR, "nb_qtype.pkl"))
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "tfidf_qtype.pkl"))
    print(f"    Saved → {SAVE_DIR}/nb_qtype.pkl")

    nb_ensemble = None
    if X_train is not None and y_train is not None:
        print("\n--- Naive Bayes (full feature matrix, for ensemble) ---")
        X_train_nb = X_train.copy()
        X_val_nb = X_val.copy()
        X_train_nb.data = np.maximum(X_train_nb.data, 0)
        X_val_nb.data = np.maximum(X_val_nb.data, 0)

        nb_ensemble = MultinomialNB(alpha=1.0)
        nb_ensemble.fit(X_train_nb, y_train)

        ens_preds = nb_ensemble.predict(X_val_nb)
        ens_acc = accuracy_score(y_val, ens_preds)
        ens_f1 = f1_score(y_val, ens_preds, average="macro")
        print(f"    Val Accuracy (binary verif): {ens_acc:.4f}")
        print(f"    Val Macro F1 (binary verif): {ens_f1:.4f}")
        joblib.dump(nb_ensemble, os.path.join(SAVE_DIR, "nb_verifier.pkl"))
        print(f"    Saved → {SAVE_DIR}/nb_verifier.pkl")

    return nb, nb_ensemble, {"accuracy": acc}


def compute_clustering_purity(labels_true, labels_pred):
    from collections import Counter

    total = len(labels_true)
    correct = 0
    for cluster_id in np.unique(labels_pred):
        mask = labels_pred == cluster_id
        counts = Counter(labels_true[mask])
        correct += counts.most_common(1)[0][1]
    return correct / total


def train_kmeans(X_train, y_train, sample_size=5000):
    print("\n--- K-Means Clustering (unsupervised) ---")

    X_sample = X_train[:sample_size]
    y_sample = y_train[:sample_size]

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_sample)
    cluster_labels = km.labels_

    inertia = km.inertia_
    purity = compute_clustering_purity(y_sample, cluster_labels)

    sil_size = min(1000, sample_size)
    X_dense = X_sample[:sil_size].toarray()
    sil_labels = cluster_labels[:sil_size]
    silhouette = silhouette_score(X_dense, sil_labels, metric="euclidean")

    print(f"    Inertia        : {inertia:.2f}")
    print(f"    Silhouette     : {silhouette:.4f}")
    print(f"    Purity         : {purity:.4f}")
    print(f"    Cluster sizes  : {np.bincount(cluster_labels)}")

    save_path = os.path.join(SAVE_DIR, "kmeans.pkl")
    joblib.dump(km, save_path)
    print(f"    Saved → {save_path}")
    return km, {"inertia": inertia, "silhouette": silhouette, "purity": purity}


def train_label_propagation(X_train, y_train, sample_size=2000):
    print("\n--- Label Propagation (semi-supervised) ---")

    X_sample = X_train[:sample_size].toarray()
    y_sample = y_train[:sample_size].copy()

    rng = np.random.default_rng(seed=42)
    mask_idx = rng.choice(sample_size, size=int(sample_size * 0.3), replace=False)
    y_sample[mask_idx] = -1

    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=100)
    lp.fit(X_sample, y_sample)

    labeled_mask = y_sample != -1
    if labeled_mask.sum() > 0:
        preds = lp.predict(X_sample[labeled_mask])
        true = y_train[:sample_size][labeled_mask]
        f1 = f1_score(true, preds, average="macro")
        acc = accuracy_score(true, preds)
        print(f"    Semi-supervised F1  : {f1:.4f}")
        print(f"    Semi-supervised Acc : {acc:.4f}")
    else:
        f1, acc = 0.0, 0.0

    print(f"    Sample size: {sample_size}  |  Unlabeled: 30%")

    save_path = os.path.join(SAVE_DIR, "label_propagation.pkl")
    joblib.dump(lp, save_path)
    print(f"    Saved → {save_path}")
    return lp, {"f1_macro": f1, "accuracy": acc}


def train_ensemble(lr, svm, nb, X_train, y_train, X_val, y_val):
    print("\n--- Soft-Vote Ensemble (LR + SVM + NB) ---")
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("svm", svm), ("nb", nb)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec = recall_score(y_val, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_val, preds).tolist()
    print(f"    Val Accuracy : {acc:.4f}")
    print(f"    Val Macro F1 : {f1:.4f}")
    print(f"    Val Precision: {prec:.4f}")
    print(f"    Val Recall   : {rec:.4f}")
    print(f"    Confusion Matrix:\n{confusion_matrix(y_val, preds)}")

    save_path = os.path.join(SAVE_DIR, "ensemble_verifier.pkl")
    joblib.dump(ensemble, save_path)
    print(f"    Saved → {save_path}")
    return ensemble, {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec, "confusion_matrix": cm}


def run_question_generation_sample(vectorizer, n_samples=5):
    print("\n--- Question Generation Sample Output ---")
    print("    (Full BLEU / ROUGE / METEOR evaluation: run evaluate.py)")

    try:
        test_df = load_race_data("test").head(n_samples)
    except FileNotFoundError:
        print("    test.csv not found – skipping sample output.")
        return

    for i, (_, row) in enumerate(test_df.iterrows()):
        reference = str(row["question"])
        generated = generate_questions_for_row(row, vectorizer, top_k=3)
        print(f"\n    [{i+1}] Reference : {reference[:90]}")
        print(f"         Generated : {generated[0][:90]}")


def run_full_training(max_train_rows=10000, max_val_rows=2000):
    print("=" * 55)
    print("MODEL A — TRAINING PIPELINE")
    print("=" * 55)

    print("\n[STEP 1] Loading feature matrices...")
    X_train, X_val, y_train, y_val = load_saved_features()
    print(f"    X_train : {X_train.shape}")
    print(f"    X_val   : {X_val.shape}")
    print(f"    Class balance: {y_train.mean():.3f}  (expected ~0.25)")

    print("\n[STEP 2] Loading dataframes and vectorizer...")
    train_df = load_race_data("train").head(max_train_rows)
    val_df = load_race_data("val").head(max_val_rows)
    tfidf_path = os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl")
    vectorizer = joblib.load(tfidf_path)
    print(f"    Train : {len(train_df)} rows | Val : {len(val_df)} rows")

    print("\n[STEP 3] Training supervised classifiers...")
    lr, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    svm, svm_metrics = train_svm(X_train, y_train, X_val, y_val)
    nb, nb_ensemble, nb_metrics = train_naive_bayes(
        train_df, val_df, vectorizer,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    )

    print("\n[STEP 4] Unsupervised and Semi-Supervised models...")
    km, km_metrics = train_kmeans(X_train, y_train)
    lp, lp_metrics = train_label_propagation(X_train, y_train)

    print("\n[STEP 5] Building ensemble...")
    nb_for_ensemble = nb_ensemble if nb_ensemble is not None else nb
    ensemble, ens_metrics = train_ensemble(lr, svm, nb_for_ensemble, X_train, y_train, X_val, y_val)

    print("\n[STEP 6] Question generation sample...")
    run_question_generation_sample(vectorizer, n_samples=5)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Logistic Regression → Acc: {lr_metrics['accuracy']:.4f}  F1: {lr_metrics['f1_macro']:.4f}  Prec: {lr_metrics['precision']:.4f}  Rec: {lr_metrics['recall']:.4f}")
    print(f"  SVM (calibrated)    → Acc: {svm_metrics['accuracy']:.4f}  F1: {svm_metrics['f1_macro']:.4f}  Prec: {svm_metrics['precision']:.4f}  Rec: {svm_metrics['recall']:.4f}")
    print(f"  Ensemble (LR+SVM+NB)→ Acc: {ens_metrics['accuracy']:.4f}  F1: {ens_metrics['f1_macro']:.4f}  Prec: {ens_metrics['precision']:.4f}  Rec: {ens_metrics['recall']:.4f}")
    print(f"  Naive Bayes Q-type  → Acc: {nb_metrics['accuracy']:.4f}")
    print(f"  K-Means             → Silhouette: {km_metrics['silhouette']:.4f}  Purity: {km_metrics['purity']:.4f}")
    print(f"  Label Propagation   → F1: {lp_metrics['f1_macro']:.4f}  Acc: {lp_metrics['accuracy']:.4f}")
    print("\n  All models saved.")
    print("  Run evaluate.py next for BLEU / ROUGE / METEOR scores.")

    import json
    comparison = {
        "Logistic Regression": {
            "accuracy": lr_metrics["accuracy"],
            "f1_macro": lr_metrics["f1_macro"],
            "precision": lr_metrics["precision"],
            "recall": lr_metrics["recall"],
            "confusion_matrix": lr_metrics["confusion_matrix"],
        },
        "SVM (calibrated)": {
            "accuracy": svm_metrics["accuracy"],
            "f1_macro": svm_metrics["f1_macro"],
            "precision": svm_metrics["precision"],
            "recall": svm_metrics["recall"],
            "confusion_matrix": svm_metrics["confusion_matrix"],
        },
        "Ensemble (LR+SVM+NB)": {
            "accuracy": ens_metrics["accuracy"],
            "f1_macro": ens_metrics["f1_macro"],
            "precision": ens_metrics["precision"],
            "recall": ens_metrics["recall"],
            "confusion_matrix": ens_metrics["confusion_matrix"],
        },
        "K-Means (unsupervised)": {
            "silhouette": km_metrics["silhouette"],
            "purity": km_metrics["purity"],
            "inertia": km_metrics["inertia"],
        },
        "Label Propagation (semi-sup)": {
            "f1_macro": lp_metrics["f1_macro"],
            "accuracy": lp_metrics["accuracy"],
        },
    }
    processed_dir = os.path.join(BASE, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    comp_path = os.path.join(processed_dir, "model_a_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison table saved → {comp_path}")

    return {
        "lr": lr_metrics,
        "svm": svm_metrics,
        "ensemble": ens_metrics,
        "nb": nb_metrics,
        "kmeans": km_metrics,
        "label_propagation": lp_metrics,
    }


if __name__ == "__main__":
    run_full_training()