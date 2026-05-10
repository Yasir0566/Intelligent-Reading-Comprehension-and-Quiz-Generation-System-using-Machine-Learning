import streamlit as st


def render_quiz_view(question, options, correct_answer, predicted_answer, probabilities, question_type):
    st.markdown("## Quiz")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-left: 4px solid #4fc3f7;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <span style="color:#4fc3f7; font-size:0.75rem; font-weight:600; 
                         letter-spacing:1px; text-transform:uppercase;">
                Question · {question_type.capitalize()} type
            </span>
            <p style="color:#ffffff; font-size:1.1rem; margin:8px 0 0 0; line-height:1.6;">
                {question}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "user_answer" not in st.session_state:
        st.session_state.user_answer = None
    if "checked" not in st.session_state:
        st.session_state.checked = False

    st.markdown("Choose your answer")

    opt_cols = st.columns(2)
    for i, opt in enumerate(["A", "B", "C", "D"]):
        with opt_cols[i % 2]:
            label = f"**{opt}.** {options.get(opt, '')}"
            if st.button(label, key=f"opt_{opt}", use_container_width=True):
                st.session_state.user_answer = opt
                st.session_state.checked = False

    if st.session_state.user_answer:
        st.markdown(f"Selected: Option {st.session_state.user_answer}")

    if st.button("Check answer", type="primary", use_container_width=True):
        st.session_state.checked = True

    if st.session_state.checked and st.session_state.user_answer:
        user_ans = st.session_state.user_answer
        is_correct = user_ans == correct_answer

        if is_correct:
            st.markdown(
                f"""
                <div style="background:#1b5e20; border:1px solid #4caf50; border-radius:8px;
                            padding:16px; margin-top:12px;">
                    <h3 style="color:#69f0ae; margin:0;">Correct</h3>
                    <p style="color:#c8e6c9; margin:8px 0 0 0;">
                        You selected <strong>{user_ans}</strong>. That is correct.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background:#b71c1c; border:1px solid #ef5350; border-radius:8px;
                            padding:16px; margin-top:12px;">
                    <h3 style="color:#ff8a80; margin:0;">Incorrect</h3>
                    <p style="color:#ffcdd2; margin:8px 0 0 0;">
                        You selected <strong>{user_ans}</strong>. 
                        The correct answer is <strong>{correct_answer}</strong>: 
                        {options.get(correct_answer, '')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f"Model A prediction: {predicted_answer}  (Confidence: {probabilities.get(predicted_answer, 0):.1%})")

        import plotly.express as px

        fig = px.bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            labels={"x": "Option", "y": "Confidence"},
            title="Model A confidence per option",
            color=list(probabilities.keys()),
            color_discrete_sequence=["#4fc3f7", "#81c784", "#ffb74d", "#e57373"],
            range_y=[0, 1],
            text_auto=".2%",
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)