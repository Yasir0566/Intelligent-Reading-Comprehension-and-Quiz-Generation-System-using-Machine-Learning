import streamlit as st


def render_hint_panel(hints, correct_answer, options):
    st.markdown("## Hints")
    st.markdown("Use hints progressively from general to specific. Try to answer before revealing.")

    if "hints_revealed" not in st.session_state:
        st.session_state.hints_revealed = 0
    if "answer_revealed" not in st.session_state:
        st.session_state.answer_revealed = False

    hint_labels = [
        ("Hint 1 - General Clue", "#1565c0", "#bbdefb"),
        ("Hint 2 - More Specific", "#f57f17", "#fff9c4"),
        ("Hint 3 - Near Explicit", "#b71c1c", "#ffcdd2"),
    ]

    for i in range(3):
        label, border_color, bg_color = hint_labels[i]
        is_unlocked = i < st.session_state.hints_revealed

        with st.expander(label, expanded=is_unlocked):
            if is_unlocked and i < len(hints):
                st.markdown(
                    f"""
                    <div style="background:{bg_color}22; border-left:3px solid {border_color};
                                border-radius:6px; padding:12px 16px; color:#e0e0e0;">
                        {hints[i]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background:#1e1e2e; border-radius:6px; padding:12px;
                                text-align:center; color:#555; font-style:italic;">
                        Unlock by clicking the button below
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.session_state.hints_revealed < 1:
            if st.button("Reveal hint 1", use_container_width=True):
                st.session_state.hints_revealed = 1
                st.rerun()

    with btn_col2:
        if 1 <= st.session_state.hints_revealed < 2:
            if st.button("Reveal hint 2", use_container_width=True):
                st.session_state.hints_revealed = 2
                st.rerun()

    with btn_col3:
        if 2 <= st.session_state.hints_revealed < 3:
            if st.button("Reveal hint 3", use_container_width=True):
                st.session_state.hints_revealed = 3
                st.rerun()

    if st.session_state.hints_revealed >= 3:
        st.markdown("---")
        if not st.session_state.answer_revealed:
            if st.button("Reveal answer", type="primary", use_container_width=True):
                st.session_state.answer_revealed = True
                st.rerun()
        else:
            st.success(f"Answer: {correct_answer} - {options.get(correct_answer, '')}")