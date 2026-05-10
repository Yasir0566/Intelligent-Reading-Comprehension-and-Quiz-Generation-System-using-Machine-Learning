







import os
import sys
import pytest
import numpy as np

                                                               
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

                                                                
SAMPLE_ARTICLE = """
The sun is the star at the center of the Solar System. It is a nearly perfect sphere of 
hot plasma, with internal convective motion that generates a magnetic field via a dynamo 
process. It is by far the most important source of energy for life on Earth. Its diameter 
is about 1.39 million kilometers, and its mass is about 330,000 times that of Earth.
The Sun has been an object of veneration in many cultures throughout human history.
It formed approximately 4.6 billion years ago from the gravitational collapse of matter 
within a region of a large molecular cloud.
"""

SAMPLE_QUESTION = "What is the sun?"

SAMPLE_OPTIONS = {
    "A": "A planet in the Solar System",
    "B": "The star at the center of the Solar System",
    "C": "A moon orbiting Earth",
    "D": "A comet passing through the galaxy",
}

SAMPLE_CORRECT = "B"

SAMPLE_ROW = {
    "article": SAMPLE_ARTICLE,
    "question": SAMPLE_QUESTION,
    "A": SAMPLE_OPTIONS["A"],
    "B": SAMPLE_OPTIONS["B"],
    "C": SAMPLE_OPTIONS["C"],
    "D": SAMPLE_OPTIONS["D"],
    "answer": SAMPLE_CORRECT,
}


                                                               
                     
                                                               

class TestPreprocessing:


    def test_clean_text_lowercases(self):
        from preprocessing import clean_text
        result = clean_text("Hello World!")
        assert result == result.lower(), "clean_text should return lowercase"

    def test_clean_text_removes_punctuation(self):
        from preprocessing import clean_text
        result = clean_text("Hello, World! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_clean_text_collapses_whitespace(self):
        from preprocessing import clean_text
        result = clean_text("  hello   world  ")
        assert "  " not in result
        assert result == result.strip()

    def test_split_sentences_returns_list(self):
        from preprocessing import split_sentences
        sentences = split_sentences(SAMPLE_ARTICLE)
        assert isinstance(sentences, list)
        assert len(sentences) > 0

    def test_split_sentences_nonempty(self):
        from preprocessing import split_sentences
        sentences = split_sentences(SAMPLE_ARTICLE)
        for s in sentences:
            assert len(s.strip()) > 0

    def test_word_overlap_score_identical(self):
        from preprocessing import word_overlap_score
        score = word_overlap_score("the quick brown fox", "the quick brown fox")
        assert score == 1.0, "Identical strings should have overlap 1.0"

    def test_word_overlap_score_disjoint(self):
        from preprocessing import word_overlap_score
        score = word_overlap_score("apple banana cherry", "dog elephant fish")
        assert score == 0.0, "Completely different strings should have overlap 0.0"

    def test_word_overlap_score_partial(self):
        from preprocessing import word_overlap_score
        score = word_overlap_score("the quick brown fox", "the slow red fox")
        assert 0.0 < score < 1.0

    def test_extract_candidate_phrases_returns_list(self):
        from preprocessing import extract_candidate_phrases
        phrases = extract_candidate_phrases(SAMPLE_ARTICLE, top_n=10)
        assert isinstance(phrases, list)
        assert len(phrases) <= 10

    def test_extract_candidate_phrases_nonempty(self):
        from preprocessing import extract_candidate_phrases
        phrases = extract_candidate_phrases(SAMPLE_ARTICLE, top_n=5)
        assert len(phrases) > 0


                                                               
                                                  
                                                               

class TestEvaluationMetrics:


    REFS = [
        "What is the sun in the Solar System?",
        "How large is the diameter of the sun?",
        "When did the sun form in the galaxy?",
    ]
    HYPS = [
        "What is the star at the center of the Solar System?",
        "How large is the sun's diameter in kilometers?",
        "When did the sun approximately form in space?",
    ]

    def test_compute_bleu_range(self):
        from evaluate import compute_bleu
        score = compute_bleu(self.REFS, self.HYPS, n=1)
        assert 0.0 <= score <= 1.0, f"BLEU-1 must be in [0,1], got {score}"

    def test_compute_bleu2_range(self):
        from evaluate import compute_bleu
        score = compute_bleu(self.REFS, self.HYPS, n=2)
        assert 0.0 <= score <= 1.0, f"BLEU-2 must be in [0,1], got {score}"

    def test_compute_bleu_perfect(self):
        from evaluate import compute_bleu
        refs = ["what is the sun"]
        hyps = ["what is the sun"]
        score = compute_bleu(refs, hyps, n=1)
        assert score == 1.0, "Perfect match should give BLEU=1.0"

    def test_compute_rouge_keys(self):
        from evaluate import compute_rouge
        result = compute_rouge(self.REFS, self.HYPS)
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result

    def test_compute_rouge_range(self):
        from evaluate import compute_rouge
        result = compute_rouge(self.REFS, self.HYPS)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} not in [0,1]"

    def test_compute_rouge_perfect(self):
        from evaluate import compute_rouge
        refs = ["what is the sun in the solar system"]
        hyps = ["what is the sun in the solar system"]
        result = compute_rouge(refs, hyps)
        assert result["rouge1"] == 1.0
        assert result["rougeL"] == 1.0

    def test_compute_meteor_range(self):
        from evaluate import compute_meteor
        score = compute_meteor(self.REFS, self.HYPS)
        assert 0.0 <= score <= 1.0, f"METEOR must be in [0,1], got {score}"

    def test_compute_meteor_perfect(self):
        from evaluate import compute_meteor
        refs = ["what is the sun"]
        hyps = ["what is the sun"]
        score = compute_meteor(refs, hyps)
        assert score == 1.0, "Perfect match should give METEOR=1.0"

    def test_evaluate_generation_returns_all_keys(self):
        from evaluate import evaluate_generation
        metrics = evaluate_generation(self.REFS, self.HYPS, label="Test")
        required_keys = ["label", "bleu1", "bleu2", "rouge1", "rouge2", "rougeL", "meteor"]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_evaluate_generation_length_mismatch_raises(self):
        from evaluate import evaluate_generation
        with pytest.raises(AssertionError):
            evaluate_generation(["ref1", "ref2"], ["hyp1"])

    def test_bleu_better_with_closer_match(self):

        from evaluate import compute_bleu
        refs = ["what is the sun in the solar system"]
        hyp_close = ["what is the sun in the solar system"]
        hyp_far   = ["completely different unrelated sentence about nothing"]
        score_close = compute_bleu(refs, hyp_close, n=1)
        score_far   = compute_bleu(refs, hyp_far,   n=1)
        assert score_close >= score_far


                                                               
                           
                                                               

class TestQuestionGeneration:


    def _get_vectorizer(self):

        from preprocessing import fit_tfidf, clean_text
        texts = [clean_text(SAMPLE_ARTICLE)]
        return fit_tfidf(texts)

    def test_generate_questions_returns_list(self):
        from preprocessing import generate_questions_for_row
        vectorizer = self._get_vectorizer()
        import pandas as pd
        row = pd.Series(SAMPLE_ROW)
        result = generate_questions_for_row(row, vectorizer, top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_questions_are_strings(self):
        from preprocessing import generate_questions_for_row
        vectorizer = self._get_vectorizer()
        import pandas as pd
        row = pd.Series(SAMPLE_ROW)
        result = generate_questions_for_row(row, vectorizer, top_k=3)
        for q in result:
            assert isinstance(q, str)
            assert len(q) > 0

    def test_generate_questions_end_with_question_mark(self):
        from preprocessing import generate_questions_for_row
        vectorizer = self._get_vectorizer()
        import pandas as pd
        row = pd.Series(SAMPLE_ROW)
        result = generate_questions_for_row(row, vectorizer, top_k=3)
        for q in result:
            assert q.strip().endswith("?"), f"Question should end with '?': {q}"

    def test_generate_questions_top_k_respected(self):
        from preprocessing import generate_questions_for_row
        vectorizer = self._get_vectorizer()
        import pandas as pd
        row = pd.Series(SAMPLE_ROW)
        result = generate_questions_for_row(row, vectorizer, top_k=2)
        assert len(result) <= 2


                                                               
                             
                                                               

class TestDistractorGeneration:


    def _get_modelb_artifacts(self):

        try:
            import sys, os
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
            from inference import load_model_b_artifacts
            return load_model_b_artifacts()
        except FileNotFoundError:
            pytest.skip("Model B not trained yet — run model_b_train.py first")

    def test_generate_distractors_returns_list(self):
        from model_b_train import generate_distractors
        artifacts_b = self._get_modelb_artifacts()
        result = generate_distractors(
            SAMPLE_ARTICLE, SAMPLE_QUESTION,
            SAMPLE_OPTIONS[SAMPLE_CORRECT], artifacts_b, n=3
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_distractors_are_strings(self):
        from model_b_train import generate_distractors
        artifacts_b = self._get_modelb_artifacts()
        result = generate_distractors(
            SAMPLE_ARTICLE, SAMPLE_QUESTION,
            SAMPLE_OPTIONS[SAMPLE_CORRECT], artifacts_b, n=3
        )
        for d in result:
            assert isinstance(d, str)

    def test_generate_hints_returns_list(self):
        from model_b_train import generate_hints
        artifacts_b = self._get_modelb_artifacts()
        result = generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION, artifacts_b, n_hints=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_hints_are_strings(self):
        from model_b_train import generate_hints
        artifacts_b = self._get_modelb_artifacts()
        result = generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION, artifacts_b, n_hints=3)
        for h in result:
            assert isinstance(h, str)
            assert len(h) > 0


                                                               
                           
                                                               

class TestAnswerVerification:


    def _get_modela_artifacts(self):
        try:
            from inference import load_model_a_artifacts
            return load_model_a_artifacts()
        except FileNotFoundError:
            pytest.skip("Model A not trained yet — run model_a_train.py first")

    def test_verify_answer_returns_dict(self):
        from inference import verify_answer
        artifacts_a = self._get_modela_artifacts()
        result = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS, artifacts_a)
        assert isinstance(result, dict)

    def test_verify_answer_has_required_keys(self):
        from inference import verify_answer
        artifacts_a = self._get_modela_artifacts()
        result = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS, artifacts_a)
        assert "predicted_answer" in result
        assert "probabilities"    in result
        assert "question_type"    in result
        assert "latency_ms"       in result

    def test_verify_answer_predicted_is_valid_label(self):
        from inference import verify_answer
        artifacts_a = self._get_modela_artifacts()
        result = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS, artifacts_a)
        assert result["predicted_answer"] in ["A", "B", "C", "D"]

    def test_verify_answer_probabilities_sum_to_one(self):
        from inference import verify_answer
        artifacts_a = self._get_modela_artifacts()
        result = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS, artifacts_a)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.1, f"Probs should sum ~1.0, got {total}"

    def test_verify_answer_latency_positive(self):
        from inference import verify_answer
        artifacts_a = self._get_modela_artifacts()
        result = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS, artifacts_a)
        assert result["latency_ms"] > 0


                                                               
             
                                                               

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])