import streamlit as st
import os
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import load_race_data


def render_article_input():
    st.markdown("## Article Input")
    st.markdown("Paste a reading passage below or load a random sample from the RACE dataset.")

    if "loaded_row" not in st.session_state:
        st.session_state.loaded_row = None

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### Quick load")
        if st.button("Random RACE sample", use_container_width=True):
            try:
                df = load_race_data("test")
                idx = random.randint(0, min(len(df) - 1, 999))
                st.session_state.loaded_row = df.iloc[idx].to_dict()
                st.success(f"Loaded row #{idx}")
            except FileNotFoundError:
                st.error("test.csv not found in data/raw/. Add the RACE dataset first.")

        if st.session_state.loaded_row is not None:
            row = st.session_state.loaded_row
            st.markdown("Reference question")
            st.info(row.get("question", ""))
            st.markdown("Options")
            for opt in ["A", "B", "C", "D"]:
                marker = "" if opt != row.get("answer", "") else " (correct)"
                st.write(f"{opt}: {row.get(opt, '')}{marker}")

    with col1:
        default_text = ""
        if st.session_state.loaded_row is not None:
            default_text = st.session_state.loaded_row.get("article", "")

        article = st.text_area(
            "Reading Passage",
            value=default_text,
            height=300,
            placeholder="Paste your reading passage here...",
            key="article_text_area",
        )

        char_count = len(article)
        word_count = len(article.split()) if article.strip() else 0
        st.caption(f"{word_count} words, {char_count} characters")

    st.markdown("---")

    submitted = False
    question = None
    options = None
    correct = None

    if article.strip():
        if st.button("Submit", type="primary", use_container_width=True):
            if len(article.split()) < 30:
                st.warning("Passage seems very short. Please provide at least 30 words for good results.")
            else:
                submitted = True

        if st.session_state.loaded_row is not None:
            row = st.session_state.loaded_row
            question = row.get("question")
            options = {o: row.get(o, "") for o in ["A", "B", "C", "D"]}
            correct = row.get("answer")
    else:
        st.info("Enter or load a reading passage above to get started.")

    return article, question, options, correct, submitted