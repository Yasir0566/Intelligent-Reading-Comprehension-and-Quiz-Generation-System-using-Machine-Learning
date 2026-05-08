import os
import re
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_AUX_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "can", "could", "shall", "should",
    "may", "might", "must",
}


def load_race_data(split="train"):
    path = os.path.join(BASE, "data", "raw", f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()
    df.dropna(subset=["article", "question", "A", "B", "C", "D", "answer"], inplace=True)
    return df.reset_index(drop=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    tokens = clean_text(text).split()
    return " ".join(w for w in tokens if w not in STOPWORDS)


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def word_overlap_score(text_a, text_b):
    set_a = set(remove_stopwords(text_a).split())
    set_b = set(remove_stopwords(text_b).split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_candidate_phrases(article, top_n=30):
    cleaned = remove_stopwords(article)
    tokens = cleaned.split()
    freq = Counter(tokens)
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        freq[bigram] = freq.get(bigram, 0) + 1
    candidates = [t for t, _ in freq.most_common(top_n * 2) if len(t) > 3]
    return candidates[:top_n]


def build_onehot_features(texts, vocabulary=None, max_features=5000):
    if vocabulary is not None:
        vectorizer = CountVectorizer(vocabulary=vocabulary, binary=True, stop_words="english")
        matrix = vectorizer.transform(texts)
    else:
        vectorizer = CountVectorizer(max_features=max_features, binary=True, stop_words="english", min_df=2)
        matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.vocabulary_


def fit_tfidf(texts, save_path=None):
    n_docs = len(texts) if texts is not None else 0
    if n_docs < 2:
        min_df = 1
        max_df = 1.0
    else:
        min_df = 2
        max_df = 0.95

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
    )
    vectorizer.fit(texts or [])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(vectorizer, save_path)
    return vectorizer


def load_tfidf(path):
    return joblib.load(path)


def compute_cosine_features(df, vectorizer):
    features = []
    for _, row in df.iterrows():
        art_vec = vectorizer.transform([clean_text(row["article"])])
        q_vec = vectorizer.transform([clean_text(row["question"])])
        for opt in ["A", "B", "C", "D"]:
            opt_vec = vectorizer.transform([clean_text(str(row[opt]))])
            features.append([
                cosine_similarity(art_vec, opt_vec)[0, 0],
                cosine_similarity(q_vec, opt_vec)[0, 0],
                cosine_similarity(art_vec, q_vec)[0, 0],
            ])
    return np.array(features, dtype=np.float32)


def build_full_feature_matrix(texts, cosine_feats, vectorizer):
    X_tfidf = vectorizer.transform(texts)
    X_cos = csr_matrix(cosine_feats)
    return hstack([X_tfidf, X_cos])


def scale_features(X_train, X_val=None, save_path=None):
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    return X_train_scaled, X_val_scaled, scaler


WH_TEMPLATES = {
    "what": "What {predicate}?",
    "who": "Who {predicate}?",
    "where": "Where {predicate}?",
    "when": "When {predicate}?",
    "why": "Why {predicate}?",
    "how": "How {predicate}?",
}


def _pick_wh_word(answer_text):
    a = answer_text.lower().strip()
    if any(h in a for h in ["man", "woman", "he", "she", "person"]):
        return "who"
    if any(h in a for h in ["city", "country", "island", "ocean"]):
        return "where"
    if any(h in a for h in ["day", "year", "century", "monday"]):
        return "when"
    if any(h in a for h in ["because", "reason", "cause"]):
        return "why"
    if any(h in a for h in ["way", "method", "manner"]):
        return "how"
    return "what"


def _make_real_question(sentence, answer, wh_word):
    sent = sentence.strip().rstrip(".")
    wh = wh_word.capitalize()
    if answer and answer.lower() in sent.lower():
        # replace answer with wh-word
        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        parts = pattern.split(sent, maxsplit=1)
        if len(parts) >= 2:
            before, after = parts[0].strip(), parts[1].strip()
            if after:
                return f"{wh} {after}?"
            if before:
                return f"{wh} {before}?"
    words = sent.split()
    if len(words) > 3:
        predicate = " ".join(words[1:])
        return f"{wh} {predicate}?"
    return f"{wh} does the passage discuss?"


def generate_questions_for_row(row, vectorizer, top_k=3):
    article = str(row.get("article", ""))
    answer_key = str(row.get("answer", "A"))
    correct_answer = str(row.get(answer_key, "")).strip() if answer_key in row else ""

    sentences = split_sentences(article)
    if not sentences:
        return ["What is the main topic of this passage?"]

    wh = _pick_wh_word(correct_answer) if correct_answer else "what"

    best_sentences = []
    if correct_answer:
        containing = [s for s in sentences if correct_answer.lower() in s.lower()]
        if containing:
            best_sentences = containing[:top_k]
        else:
            try:
                ans_vec = vectorizer.transform([clean_text(correct_answer)])
                scored = []
                for sent in sentences:
                    sv = vectorizer.transform([clean_text(sent)])
                    score = cosine_similarity(sv, ans_vec)[0, 0]
                    scored.append((sent, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                best_sentences = [s for s, _ in scored[:top_k]]
            except Exception:
                best_sentences = sentences[:top_k]
    else:
        best_sentences = sentences[:top_k]

    filtered = [s for s in best_sentences if 5 <= len(s.split()) <= 50]
    if not filtered:
        filtered = best_sentences

    generated = [_make_real_question(s, correct_answer, wh) for s in filtered]
    generated = [g if g.strip().endswith('?') else g.strip() + '?' for g in generated]
    return generated if generated else [f"{wh.capitalize()} does the passage discuss?"]


if __name__ == "__main__":
    print("preprocessing module loaded")
