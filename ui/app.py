import os
import sys
import json
import time
import random
import re
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import BASE, load_race_data, clean_text, split_sentences, extract_candidate_phrases, generate_questions_for_row
from inference import load_model_a_artifacts, load_model_b_artifacts, verify_answer, generate_distractors, generate_hints
from evaluate import compute_bleu, compute_rouge, compute_meteor

st.set_page_config(
    page_title="Reading Comprehension & Quiz System",
    layout="wide",
)
@st.cache_resource
def get_model_a():
    return load_model_a_artifacts()


@st.cache_resource
def get_model_b():
    return load_model_b_artifacts()


@st.cache_data
def get_random_race_sample():
    try:
        df = load_race_data("test")
        row = df.sample(1, random_state=None).iloc[0]
        return row
    except FileNotFoundError:
        return None
defaults = {
    "article": "",
    "question": "",
    "options": {"A": "", "B": "", "C": "", "D": ""},
    "correct_answer": "",
    "loaded_row": None,
    "generated": False,
    "selected_option": None,
    "answer_checked": False,
    "hints_viewed": 0,
    "reveal_requested": False,
    "session_log": [],
    "current_screen": "Screen 1",
    "inference_result": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
st.sidebar.title("Navigation")
screen = st.sidebar.radio(
    "Go to",
    ["Screen 1 - Article Input", "Screen 2 - Quiz View", "Screen 3 - Hint Panel", "Screen 4 - Analytics"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("Model status")

models_loaded = False
try:
    artifacts_a = get_model_a()
    artifacts_b = get_model_b()
    st.sidebar.success("Model A loaded")
    st.sidebar.success("Model B loaded")
    models_loaded = True
except FileNotFoundError as e:
    st.sidebar.error(f"Models not found. Run preprocessing.py, model_a_train.py, and model_b_train.py first.\n\n{e}")

eval_metrics = None
eval_path = os.path.join(BASE, "data", "processed", "eval_metrics.json")
if os.path.exists(eval_path):
    with open(eval_path) as f:
        eval_metrics = json.load(f)
def run_inference(article, question, options, correct_answer):
    if not models_loaded:
        st.error("Models are not loaded. Check the sidebar.")
        return None

    with st.spinner("Running inference"):
        result = verify_answer(article, question, options, artifacts_a)
        distractors = generate_distractors(
            article, question, str(options[correct_answer]), artifacts_b, n=3
        )
        hints = generate_hints(article, question, artifacts_b, n_hints=3)
        row_like = {
            "article": article,
            "question": question,
            "answer": correct_answer,
            "A": options["A"],
            "B": options["B"],
            "C": options["C"],
            "D": options["D"],
        }
        generated_q = generate_questions_for_row(row_like, artifacts_a["tfidf"], top_k=1)
        gen_q = generated_q[0] if generated_q else question

        bleu1 = compute_bleu([question], [gen_q], n=1)
        rougeL = compute_rouge([question], [gen_q])["rougeL"]
        meteor = compute_meteor([question], [gen_q])

        return {
            "predicted_answer": result["predicted_answer"],
            "probabilities":    result["probabilities"],
            "question_type":    result["question_type"],
            "latency_ms":       result["latency_ms"],
            "distractors":      distractors,
            "hints":            hints,
            "generated_question": gen_q,
            "bleu1":  bleu1,
            "rougeL": rougeL,
            "meteor": meteor,
        }


def _normalize_phrase(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.strip(" .,;:!?()[]{}\"'")


def _looks_like_bad_fragment(text):
    t = _normalize_phrase(text)
    if not t:
        return True
    words = t.split()
    if len(words) == 1 and len(words[0]) <= 3:
        return True
    bad_starts = {
        "for", "with", "from", "into", "onto", "about", "after", "before",
        "during", "because", "although", "while", "if", "when", "where",
    }
    return words[0].lower() in bad_starts


def _pick_anchor_sentence(article):
    sentences = split_sentences(article)
    if not sentences:
        return ""

    def score_sentence(s):
        words = s.split()
        if len(words) < 8 or len(words) > 40:
            return -1.0
        score = 0.0
        if re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", s):
            score += 1.0
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", s):
            score += 0.8
        if re.search(r"\b(is|was|were|are|has|have|includes|consists|contains)\b", s, flags=re.IGNORECASE):
            score += 0.6
        score += max(0.0, 1.0 - abs(len(words) - 18) / 18)
        return score

    return max(sentences, key=score_sentence)


def _extract_answer_from_sentence(sentence, article):
    year_match = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", sentence)
    if year_match:
        return year_match.group(1)

    proper_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", sentence)
    proper_matches = [m for m in proper_matches if m.lower() not in {"The Game", "National University"}]
    if proper_matches:
        return _normalize_phrase(max(proper_matches, key=len))

    phrase_pool = extract_candidate_phrases(sentence, top_n=20)
    for phrase in phrase_pool:
        p = _normalize_phrase(phrase)
        if len(p.split()) >= 2 and not _looks_like_bad_fragment(p):
            return p

    article_pool = extract_candidate_phrases(article, top_n=20)
    for phrase in article_pool:
        p = _normalize_phrase(phrase)
        if len(p.split()) >= 2 and not _looks_like_bad_fragment(p):
            return p

    words = [w for w in sentence.split() if len(w) > 4]
    return _normalize_phrase(" ".join(words[:3])) if words else "main idea"


def _build_type_aware_distractors(answer_text, sentence, article, n=3):
    distractors = []
    answer_clean = clean_text(answer_text)

    year_match = re.fullmatch(r"(1[6-9]\d{2}|20\d{2})", answer_text.strip())
    if year_match:
        year = int(year_match.group(1))
        for delta in [1, 2, 3, 5, 10]:
            for sign in [-1, 1]:
                cand = str(year + sign * delta)
                if cand != answer_text and cand not in distractors:
                    distractors.append(cand)
                if len(distractors) >= n:
                    return distractors[:n]

    proper_pool = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", article)
    for cand in proper_pool:
        cand_n = _normalize_phrase(cand)
        if not cand_n or clean_text(cand_n) == answer_clean:
            continue
        if _looks_like_bad_fragment(cand_n):
            continue
        if cand_n not in distractors:
            distractors.append(cand_n)
        if len(distractors) >= n:
            return distractors[:n]

    phrase_pool = extract_candidate_phrases(article, top_n=80)
    ans_len = max(1, len(answer_text.split()))
    for cand in phrase_pool:
        cand_n = _normalize_phrase(cand)
        if not cand_n or clean_text(cand_n) == answer_clean:
            continue
        if _looks_like_bad_fragment(cand_n):
            continue
        if abs(len(cand_n.split()) - ans_len) > 2:
            continue
        overlap = len(set(clean_text(cand_n).split()) & set(answer_clean.split()))
        if overlap > max(1, ans_len // 2):
            continue
        if cand_n not in distractors:
            distractors.append(cand_n)
        if len(distractors) >= n:
            return distractors[:n]

    model_b_fallback = generate_distractors(article, "", answer_text, artifacts_b, n=max(6, n))
    for cand in model_b_fallback:
        cand_n = _normalize_phrase(cand)
        if not cand_n or clean_text(cand_n) == answer_clean:
            continue
        if _looks_like_bad_fragment(cand_n):
            continue
        if cand_n not in distractors:
            distractors.append(cand_n)
        if len(distractors) >= n:
            return distractors[:n]

    while len(distractors) < n:
        distractors.append(f"Alternative detail {len(distractors) + 1}")

    return distractors[:n]


def _build_cloze_question(sentence, answer_text):
    if not sentence:
        return "According to the passage, what is the best answer?"

    replaced = re.sub(re.escape(answer_text), "____", sentence, count=1, flags=re.IGNORECASE)
    if replaced == sentence:
        return f"According to the passage, what best completes this statement: {sentence.rstrip('.')}?"
    return f"Fill in the blank based on the passage: {replaced.rstrip('.')}?"


def build_auto_quiz(article, source_row=None):
    if not models_loaded:
        return None

    if source_row is not None:
        options = {
            "A": str(source_row.get("A", "")),
            "B": str(source_row.get("B", "")),
            "C": str(source_row.get("C", "")),
            "D": str(source_row.get("D", "")),
        }
        correct_answer = str(source_row.get("answer", "A")).strip().upper()
        question = str(source_row.get("question", "")).strip()

        if question and all(options.values()) and correct_answer in options:
            return {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
            }

    anchor_sentence = _pick_anchor_sentence(article)
    correct_answer_text = _extract_answer_from_sentence(anchor_sentence, article)
    distractors = _build_type_aware_distractors(correct_answer_text, anchor_sentence, article, n=3)

    option_texts = [correct_answer_text] + distractors[:3]
    random.shuffle(option_texts)

    labels = ["A", "B", "C", "D"]
    options = {label: option_texts[i] for i, label in enumerate(labels)}
    correct_answer = next(
        label for label, text in options.items()
        if clean_text(text) == clean_text(correct_answer_text)
    )

    question = _build_cloze_question(anchor_sentence, correct_answer_text)

    return {
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
    }


if "Screen 1" in screen:
    st.title("Screen 1 - Article Input")
    st.markdown(
        "Paste a reading passage or load a sample from the RACE dataset. Submit to run Model A and Model B."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if "article_input" not in st.session_state:
            st.session_state["article_input"] = st.session_state["article"]

        article_input = st.text_area(
            "Reading passage",
            key="article_input",
            height=250,
            placeholder="Paste your passage here...",
        )

    with col2:
        st.markdown("Sample data")
        if st.button("Load random RACE sample"):
            sample = get_random_race_sample()
            if sample is not None:
                st.session_state["article"] = str(sample["article"])
                st.session_state["article_input"] = str(sample["article"])
                st.session_state["loaded_row"] = sample.to_dict()
                st.session_state["generated"] = False
                st.session_state["answer_checked"] = False
                st.session_state["hints_viewed"] = 0
                st.session_state["reveal_requested"] = False
                st.rerun()
            else:
                st.warning("Could not load RACE data. Check data/raw/test.csv.")

    st.markdown("---")
    if st.button("Submit", type="primary"):
        if not article_input.strip():
            st.error("Please enter a reading passage.")
        else:
            source_row = st.session_state.get("loaded_row")
            if source_row is not None:
                source_article = str(source_row.get("article", ""))
                if clean_text(article_input) != clean_text(source_article):
                    source_row = None

            quiz = build_auto_quiz(article_input, source_row)
            if quiz is None:
                st.error("Models are not loaded. Check the sidebar.")
            else:
                st.session_state["article"] = article_input
                st.session_state["question"] = quiz["question"]
                st.session_state["options"] = quiz["options"]
                st.session_state["correct_answer"] = quiz["correct_answer"]

                result = run_inference(
                    article_input,
                    quiz["question"],
                    quiz["options"],
                    quiz["correct_answer"],
                )

                if result:
                    st.session_state["inference_result"] = result
                    st.session_state["generated"] = True
                    st.session_state["answer_checked"] = False
                    st.session_state["selected_option"] = None
                    st.session_state["hints_viewed"] = 0
                    st.session_state["reveal_requested"] = False
                    st.session_state["session_log"].append({
                        "question": quiz["question"],
                        "predicted": result["predicted_answer"],
                        "correct": quiz["correct_answer"],
                        "latency_ms": round(result["latency_ms"], 1),
                        "bleu1": result["bleu1"],
                        "rougeL": result["rougeL"],
                        "meteor": result["meteor"],
                    })

                    st.success("Quiz generated. Navigate to Screen 2, 3, or 4.")

elif "Screen 2" in screen:
    st.title("Screen 2 - Quiz View")

    if not st.session_state["generated"]:
        st.info("No quiz loaded yet. Go to Screen 1 and click Submit.")
    else:
        result = st.session_state["inference_result"]
        options = st.session_state["options"]
        correct = st.session_state["correct_answer"]

        st.markdown("### Passage excerpt")
        excerpt = st.session_state["article"][:600]
        st.markdown(f"> {excerpt}{'...' if len(st.session_state['article']) > 600 else ''}")

        st.markdown("---")
        st.markdown(f"### Question\n**{st.session_state['question']}**")

        st.markdown("---")
        st.markdown("### Options")
        selected = st.radio(
            "Select your answer",
            ["A", "B", "C", "D"],
            format_func=lambda o: f"**{o}.** {options[o]}",
            index=None if st.session_state["selected_option"] is None
                  else ["A", "B", "C", "D"].index(st.session_state["selected_option"]),
        )

        if selected:
            st.session_state["selected_option"] = selected

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check answer", type="primary"):
                if not st.session_state["selected_option"]:
                    st.warning("Please select an option first.")
                else:
                    st.session_state["answer_checked"] = True

        if st.session_state["answer_checked"]:
            user_ans = st.session_state["selected_option"]
            if user_ans == correct:
                st.success(f"Correct. The answer is {correct}: {options[correct]}")
            else:
                st.error(
                    f"Incorrect. You chose {user_ans}: {options[user_ans]}\n\n"
                    f"The correct answer is {correct}: {options[correct]}"
                )

            st.markdown("---")
            st.markdown("### Model A prediction")
            pred = result["predicted_answer"]
            pred_correct = pred == correct
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted answer", pred, delta="Correct" if pred_correct else "Incorrect")
            with col_b:
                st.metric("Question type", result["question_type"].capitalize())

            st.markdown("Option confidence scores")
            prob_data = pd.DataFrame({
                "Option": list(result["probabilities"].keys()),
                "Confidence": [round(v, 4) for v in result["probabilities"].values()],
            })
            st.bar_chart(prob_data.set_index("Option"))

        with col2:
            if st.button("Reset quiz"):
                st.session_state["answer_checked"] = False
                st.session_state["selected_option"] = None
                st.session_state["hints_viewed"] = 0
                st.session_state["reveal_requested"] = False
                st.rerun()

elif "Screen 3" in screen:
    st.title("Screen 3 - Hint Panel")

    if not st.session_state["generated"]:
        st.info("No quiz loaded yet. Go to Screen 1 and click Submit.")
    else:
        result = st.session_state["inference_result"]
        hints = result.get("hints", [])
        correct = st.session_state["correct_answer"]
        options = st.session_state["options"]

        st.markdown(f"### {st.session_state['question']}")
        st.markdown("---")
        st.markdown(
            "Hints are revealed one at a time, from most general to most specific. Reveal answer appears only after all three hints have been viewed."
        )

        if st.button("Show hint 1"):
            st.session_state["hints_viewed"] = max(st.session_state["hints_viewed"], 1)

        if st.session_state["hints_viewed"] >= 1:
            st.info(f"Hint 1: {hints[0] if hints else 'No hint available.'}")

        if st.session_state["hints_viewed"] >= 1:
            if st.button("Show hint 2"):
                st.session_state["hints_viewed"] = max(st.session_state["hints_viewed"], 2)

        if st.session_state["hints_viewed"] >= 2:
            st.warning(f"Hint 2: {hints[1] if len(hints) > 1 else 'No hint available.'}")

        if st.session_state["hints_viewed"] >= 2:
            if st.button("Show hint 3"):
                st.session_state["hints_viewed"] = max(st.session_state["hints_viewed"], 3)

        if st.session_state["hints_viewed"] >= 3:
            st.error(f"Hint 3: {hints[2] if len(hints) > 2 else 'No hint available.'}")

        st.markdown("---")
        if st.session_state["hints_viewed"] < 3:
            remaining = 3 - st.session_state["hints_viewed"]
            st.caption(f"View {remaining} more hint(s) to unlock reveal answer.")
        else:
            if st.button("Reveal answer", type="primary"):
                st.session_state["reveal_requested"] = True

            if st.session_state["reveal_requested"]:
                st.success(f"The correct answer is {correct}: {options[correct]}")

        st.markdown("---")
        st.markdown("### Generated distractors")
        distractors = result.get("distractors", [])
        for i, d in enumerate(distractors, 1):
            st.markdown(f"- Distractor {i}: {d}")

elif "Screen 4" in screen:
    st.title("Screen 4 - Analytics Dashboard")

    st.markdown("### Question generation quality (RACE test set)")
    st.caption(
        "Tracks BLEU, ROUGE, and METEOR scores for generation tasks. Pre-computed by running evaluate.py on the RACE test set."
    )

    if eval_metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("BLEU-1", eval_metrics.get("bleu1", "N/A"))
        col2.metric("BLEU-2", eval_metrics.get("bleu2", "N/A"))
        col3.metric("ROUGE-1", eval_metrics.get("rouge1", "N/A"))

        col4, col5, col6 = st.columns(3)
        col4.metric("ROUGE-2", eval_metrics.get("rouge2", "N/A"))
        col5.metric("ROUGE-L", eval_metrics.get("rougeL", "N/A"))
        col6.metric("METEOR", eval_metrics.get("meteor", "N/A"))

        import plotly.express as px
        metric_names = ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
        metric_values = [
            eval_metrics.get("bleu1", 0),
            eval_metrics.get("bleu2", 0),
            eval_metrics.get("rouge1", 0),
            eval_metrics.get("rouge2", 0),
            eval_metrics.get("rougeL", 0),
            eval_metrics.get("meteor", 0),
        ]
        fig = px.bar(
            x=metric_names, y=metric_values,
            title="Generation metrics - RACE test set",
            labels={"x": "Metric", "y": "Score"},
            range_y=[0, 1],
            color=metric_names,
            color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=".4f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "Evaluation metrics not found. Run python src/evaluate.py to generate them."
        )

    st.markdown("---")

    st.markdown("### Session log")
    log = st.session_state["session_log"]

    if not log:
        st.info("No quiz attempts yet in this session. Use Screen 1 to start.")
    else:
        log_df = pd.DataFrame(log)

        log_df = log_df.rename(columns={
            "question": "Question",
            "predicted": "Predicted",
            "correct": "Correct Answer",
            "latency_ms": "Latency (ms)",
            "bleu1": "BLEU-1",
            "rougeL": "ROUGE-L",
            "meteor": "METEOR",
        })

        st.dataframe(log_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### Inference latency")
        avg_lat = log_df["Latency (ms)"].mean()
        max_lat = log_df["Latency (ms)"].max()
        col1, col2 = st.columns(2)
        col1.metric("Average latency", f"{avg_lat:.1f} ms")
        col2.metric("Max latency", f"{max_lat:.1f} ms")

        correct_preds = (log_df["Predicted"] == log_df["Correct Answer"]).sum()
        total_preds = len(log_df)
        session_acc = correct_preds / total_preds if total_preds > 0 else 0
        st.metric("Model A session accuracy", f"{session_acc:.2%}", help="Fraction of attempts where Model A predicted the correct option.")

        st.markdown("---")
        st.markdown("### Export session results")
        csv = log_df.to_csv(index=False)
        st.download_button(
            label="Download session log as CSV",
            data=csv,
            file_name="session_results.csv",
            mime="text/csv",
        )