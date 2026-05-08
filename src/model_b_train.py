








import os
import sys
import re
import numpy as np
import joblib
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, r2_score,
)
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    BASE,
    load_race_data,
    clean_text,
    fit_tfidf,
    split_sentences,
    word_overlap_score,
    extract_candidate_phrases,
)

SAVE_DIR = os.path.join(BASE, "models", "model_b", "traditional")
os.makedirs(SAVE_DIR, exist_ok=True)


                                                               
                                
                                                               

def build_distractor_features(candidate, correct_answer, article, vectorizer):










    vc   = vectorizer.transform([clean_text(candidate)])
    va   = vectorizer.transform([clean_text(correct_answer)])
    vart = vectorizer.transform([clean_text(article)])

    cos_to_ans = cosine_similarity(vc, va)[0, 0]
    cos_to_art = cosine_similarity(vc, vart)[0, 0]

    chars_c  = set(candidate.lower())
    chars_a  = set(correct_answer.lower())
    char_jac = len(chars_c & chars_a) / max(len(chars_c | chars_a), 1)

    art_tok  = clean_text(article).split()
    cand_tok = clean_text(candidate).split()
    pfreq    = sum(art_tok.count(t) for t in cand_tok) / max(len(art_tok), 1)

    return np.array([cos_to_ans, cos_to_art, char_jac, pfreq], dtype=np.float32)


def _normalize_candidate(candidate):
    candidate = re.sub(r"\s+", " ", str(candidate)).strip()
    candidate = candidate.strip(" \t\n\r,;:.!?-–—()[]{}")
    return candidate


def _candidate_looks_plausible(candidate):
    words = candidate.split()
    if len(words) == 0 or len(words) > 8:
        return False
    if len(words) == 1:
        token = words[0]
        return len(token) > 3 or token[0].isupper() or any(ch.isdigit() for ch in token)
    return True


def _extract_contextual_candidates(article, correct_answer, top_n=60):

    sentences = split_sentences(article)
    candidates = []
    answer_norm = clean_text(correct_answer)

    named_entity_pattern = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    numeric_pattern = re.compile(
        r"\b\d{1,4}(?:\.\d+)?(?:\s?(?:million|billion|thousand|percent|%|km|kilometers?|years?|months?|days?))?\b",
        re.IGNORECASE,
    )
    split_pattern = re.compile(r"\b(?:and|or|but|because|while|however|although|which|that|who|where|when|since|after|before|then|so)\b|[;,:()]")

    for sentence in sentences:
        sentence_norm = clean_text(sentence)
        fragments = [fragment.strip() for fragment in split_pattern.split(sentence) if fragment.strip()]
        for fragment in fragments:
            fragment = _normalize_candidate(fragment)
            if _candidate_looks_plausible(fragment):
                candidates.append(fragment)

        for match in named_entity_pattern.findall(sentence):
            match = _normalize_candidate(match)
            if _candidate_looks_plausible(match):
                candidates.append(match)

        for match in numeric_pattern.findall(sentence):
            match = _normalize_candidate(match)
            if _candidate_looks_plausible(match):
                candidates.append(match)

        if answer_norm and answer_norm in sentence_norm:
            answer_parts = re.split(r"\b(?:and|or|but|because|while|however|although|which|that|who|where|when|since|after|before|then|so)\b|[;,:()]", sentence)
            for part in answer_parts:
                part = _normalize_candidate(part)
                if _candidate_looks_plausible(part) and clean_text(part) != answer_norm:
                    candidates.append(part)

    if not candidates:
        candidates = extract_candidate_phrases(article, top_n=top_n)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate_norm = clean_text(candidate)
        if not candidate_norm or candidate_norm == answer_norm:
            continue
        if candidate_norm in seen:
            continue
        if answer_norm and (candidate_norm in answer_norm or answer_norm in candidate_norm):
            continue
        seen.add(candidate_norm)
        unique_candidates.append(candidate)
        if len(unique_candidates) >= top_n:
            break

    return unique_candidates


def _score_distractor_candidate(candidate, correct_answer, article, vectorizer, question=""):
    tfidf_candidate = vectorizer.transform([clean_text(candidate)])
    tfidf_answer    = vectorizer.transform([clean_text(correct_answer)])
    tfidf_article   = vectorizer.transform([clean_text(article)])
    tfidf_question   = vectorizer.transform([clean_text(question)]) if question else None

    cos_to_ans  = cosine_similarity(tfidf_candidate, tfidf_answer)[0, 0]
    cos_to_art  = cosine_similarity(tfidf_candidate, tfidf_article)[0, 0]
    cos_to_ques = cosine_similarity(tfidf_candidate, tfidf_question)[0, 0] if tfidf_question is not None else 0.0

    candidate_len = max(len(candidate.split()), 1)
    answer_len    = max(len(clean_text(correct_answer).split()), 1)
    length_score  = 1.0 - min(abs(candidate_len - answer_len) / max(answer_len, 4), 1.0)

    candidate_words = set(clean_text(candidate).split())
    answer_words    = set(clean_text(correct_answer).split())
    overlap        = len(candidate_words & answer_words) / max(len(candidate_words | answer_words), 1)

    features = build_distractor_features(candidate, correct_answer, article, vectorizer)
    plausibility = float(np.clip(features[1], 0.0, 1.0))

    score = (
        0.35 * plausibility
        + 0.25 * cos_to_art
        + 0.15 * length_score
        - 0.45 * cos_to_ans
        - 0.20 * overlap
        - 0.05 * cos_to_ques
    )
    return score


                                                               
                        
                                                               

def build_distractor_training_data(df, vectorizer, max_rows=3000):






    X, y = [], []
    for _, row in df.head(max_rows).iterrows():
        correct      = row["answer"]
        correct_text = str(row[correct])
        wrong_opts   = [str(row[o]) for o in ["A", "B", "C", "D"] if o != correct]

                                                
        for opt in wrong_opts:
            X.append(build_distractor_features(opt, correct_text, row["article"], vectorizer))
            y.append(1)

                                                     
        for cand in extract_candidate_phrases(row["article"], top_n=10)[:3]:
            X.append(build_distractor_features(cand, correct_text, row["article"], vectorizer))
            y.append(0)

    return np.array(X, dtype=np.float32), np.array(y)


def build_hint_training_data(df, vectorizer, max_rows=3000):













    X, y = [], []
    for _, row in df.head(max_rows).iterrows():
        correct_text = str(row[row["answer"]]).lower()
        question     = str(row["question"])
        sentences    = split_sentences(row["article"])

        for i, sent in enumerate(sentences):
            overlap = word_overlap_score(sent, question)
            slen    = min(len(sent.split()) / 50.0, 1.0)
            pos     = i / max(len(sentences), 1)

            vs  = vectorizer.transform([clean_text(sent)])
            vq  = vectorizer.transform([clean_text(question)])
            cos = cosine_similarity(vs, vq)[0, 0]

            X.append([overlap, slen, pos, cos])
            y.append(1 if correct_text in sent.lower() else 0)

    return np.array(X, dtype=np.float32), np.array(y)


                                                               
                                                      
                                                               

def generate_distractors(article, question, correct_answer, artifacts_b, n=3):










    tfidf = artifacts_b["tfidf"]
    rf    = artifacts_b["rf"]

    candidates = _extract_contextual_candidates(article, correct_answer, top_n=60)
    if len(candidates) < n * 3:
        candidates.extend(extract_candidate_phrases(article, top_n=50))

                                              
    seen = set()
    candidates = [c for c in candidates if not (clean_text(c) in seen or seen.add(clean_text(c)))]

    scored     = []

    for cand in candidates:
        if clean_text(cand) == clean_text(correct_answer):
            continue
        feats = build_distractor_features(cand, correct_answer, article, tfidf)
        rf_prob = rf.predict_proba(feats.reshape(1, -1))[0][1]
        bonus   = _score_distractor_candidate(cand, correct_answer, article, tfidf, question=question)
        scored.append((cand, 0.55 * rf_prob + 0.45 * bonus))

    scored.sort(key=lambda x: x[1], reverse=True)

                                                                              
    selected = []
    for cand, _ in scored:
        is_diverse = all(word_overlap_score(cand, sel) < 0.35 for sel in selected)
        not_too_similar_to_answer = word_overlap_score(cand, correct_answer) < 0.65
        if is_diverse:
            if not_too_similar_to_answer:
                selected.append(cand)
        if len(selected) >= n:
            break

    if len(selected) < n:
        for cand, _ in scored:
            if cand in selected:
                continue
            if word_overlap_score(cand, correct_answer) >= 0.85:
                continue
            selected.append(cand)

    while len(selected) < n:
        filler = clean_text(correct_answer).title() if correct_answer else "Passage detail"
        selected.append(f"Related detail: {filler}")

    return selected[:n]


def generate_hints(article, question, artifacts_b, n_hints=3):








    tfidf   = artifacts_b["tfidf"]
    lr_hint = artifacts_b["lr_hint"]

    sentences = split_sentences(article)
    if not sentences:
        return ["No hints available."] * n_hints

    scored = []
    for i, sent in enumerate(sentences):
        overlap = word_overlap_score(sent, question)
        slen    = min(len(sent.split()) / 50.0, 1.0)
        pos     = i / max(len(sentences), 1)
        vs      = tfidf.transform([clean_text(sent)])
        vq      = tfidf.transform([clean_text(question)])
        cos     = cosine_similarity(vs, vq)[0, 0]

        feats = np.array([[overlap, slen, pos, cos]], dtype=np.float32)
        prob  = lr_hint.predict_proba(feats)[0][1]
        scored.append((sent, prob))

    scored.sort(key=lambda x: x[1], reverse=True)

                                                                     
                                                          
                                               
    top_hints = [s for s, _ in scored[:n_hints]]
    hints     = list(reversed(top_hints))

    while len(hints) < n_hints:
        hints.append("Read the passage carefully for more context.")

    return hints[:n_hints]


                                                               
                        
                                                               

def train_model_b(max_rows=6000):










    print("=" * 55)
    print("MODEL B — TRAINING PIPELINE")
    print("=" * 55)

                                                                
    print("\n[STEP 1] Loading data and fitting TF-IDF...")
    train_df   = load_race_data("train").head(max_rows)
    corpus     = train_df["article"].apply(clean_text).tolist()
    tfidf_path = os.path.join(SAVE_DIR, "tfidf_modelb.pkl")
    vectorizer = fit_tfidf(corpus, save_path=tfidf_path)
    print(f"    Articles   : {len(corpus)}")
    print(f"    Vocabulary : {len(vectorizer.vocabulary_)}")

                                                                
    print("\n[STEP 2] Building distractor training data...")
    X_d, y_d = build_distractor_training_data(train_df, vectorizer, max_rows=3000)
    print(f"    Samples       : {len(X_d)}")
    print(f"    Positive rate : {y_d.mean():.3f}")

    print("\n    Training RandomForest distractor ranker...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_d, y_d)

    rf_preds = rf.predict(X_d)
    rf_f1    = f1_score(y_d, rf_preds, average="macro")
    rf_prec  = precision_score(y_d, rf_preds, average="macro", zero_division=0)
    rf_rec   = recall_score(y_d, rf_preds, average="macro", zero_division=0)
    rf_acc   = accuracy_score(y_d, rf_preds)
    rf_cm    = confusion_matrix(y_d, rf_preds).tolist()
    print(f"    Train F1        : {rf_f1:.4f}")
    print(f"    Train Precision : {rf_prec:.4f}")
    print(f"    Train Recall    : {rf_rec:.4f}")
    print(f"    Train Accuracy  : {rf_acc:.4f}")
    print(f"    Confusion Matrix:\n{confusion_matrix(y_d, rf_preds)}")

    rf_path = os.path.join(SAVE_DIR, "rf_distractor_ranker.pkl")
    joblib.dump(rf, rf_path)
    print(f"    Saved → {rf_path}")

                                                                
    print("\n[STEP 3] Building hint training data...")
    X_h, y_h = build_hint_training_data(train_df, vectorizer, max_rows=3000)
    print(f"    Samples       : {len(X_h)}")
    print(f"    Positive rate : {y_h.mean():.3f}")

    print("\n    Training LR hint scorer...")
    lr_hint = LogisticRegression(
        max_iter=500,
        C=1.0,
        class_weight="balanced",
    )
    lr_hint.fit(X_h, y_h)

    hint_preds = lr_hint.predict(X_h)
    hint_proba = lr_hint.predict_proba(X_h)[:, 1]
    hint_f1    = f1_score(y_h, hint_preds, average="macro")
    hint_prec  = precision_score(y_h, hint_preds, average="macro", zero_division=0)
    hint_rec   = recall_score(y_h, hint_preds, average="macro", zero_division=0)
    hint_acc   = accuracy_score(y_h, hint_preds)
    hint_r2    = r2_score(y_h, hint_proba)
    hint_cm    = confusion_matrix(y_h, hint_preds).tolist()
    print(f"    Train F1        : {hint_f1:.4f}")
    print(f"    Train Precision : {hint_prec:.4f}")
    print(f"    Train Recall    : {hint_rec:.4f}")
    print(f"    Train Accuracy  : {hint_acc:.4f}")
    print(f"    R² Score        : {hint_r2:.4f}  (relevance score correlation)")
    print(f"    Confusion Matrix:\n{confusion_matrix(y_h, hint_preds)}")

    lr_path = os.path.join(SAVE_DIR, "lr_hint_scorer.pkl")
    joblib.dump(lr_hint, lr_path)
    print(f"    Saved → {lr_path}")

                                                                
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 55)
    print(f"  RF Distractor Ranker → Acc: {rf_acc:.4f}  F1: {rf_f1:.4f}  Prec: {rf_prec:.4f}  Rec: {rf_rec:.4f}")
    print(f"  LR Hint Scorer       → Acc: {hint_acc:.4f}  F1: {hint_f1:.4f}  Prec: {hint_prec:.4f}  Rec: {hint_rec:.4f}  R²: {hint_r2:.4f}")
    print("\n  All models saved.")
    print("  Run evaluate.py to get BLEU / ROUGE / METEOR scores.")

                                                                
    import json
    model_b_metrics = {
        "rf_distractor_ranker": {
            "accuracy":         rf_acc,
            "f1_macro":         rf_f1,
            "precision":        rf_prec,
            "recall":           rf_rec,
            "confusion_matrix": rf_cm,
        },
        "lr_hint_scorer": {
            "accuracy":         hint_acc,
            "f1_macro":         hint_f1,
            "precision":        hint_prec,
            "recall":           hint_rec,
            "r2_score":         hint_r2,
            "confusion_matrix": hint_cm,
        },
    }
    processed_dir = os.path.join(BASE, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    mb_path = os.path.join(processed_dir, "model_b_metrics.json")
    with open(mb_path, "w") as f:
        json.dump(model_b_metrics, f, indent=2)
    print(f"\n  Model B metrics saved → {mb_path}")

    return {
        "rf_distractor":  {"f1": rf_f1,   "precision": rf_prec,  "recall": rf_rec,  "accuracy": rf_acc,  "r2": None},
        "lr_hint_scorer": {"f1": hint_f1, "precision": hint_prec, "recall": hint_rec, "accuracy": hint_acc, "r2": hint_r2},
    }


if __name__ == "__main__":
    train_model_b(max_rows=6000)