# Reading Comprehension & Quiz System (RACE)

This project builds a traditional ML pipeline on the RACE dataset to verify multiple-choice answers, generate quiz questions, and produce distractors and hints. It includes a Streamlit UI for interactive quizzes plus analytics dashboards for BLEU/ROUGE/METEOR and model performance.

## What it does
Model A focuses on answer verification (binary correct/incorrect classification per option) and question-type classification. Model B focuses on distractor ranking and hint relevance. A lightweight question-generation routine produces WH-questions from passage sentences, and the evaluation pipeline reports standard NLG metrics.

## Key features
1. **Model A**: TF-IDF + cosine features, supervised classifiers (LR, calibrated SVM, NB) and a soft-vote ensemble.
2. **Model B**: RF-based distractor ranker and LR-based hint scorer with contextual candidate extraction.
3. **Streamlit UI**: 4-screen workflow (article input, quiz view, hints, analytics) with session logs and exports.

## Project structure
| Path | Description |
| --- | --- |
| `src/` | Core pipeline: preprocessing utilities, training, inference, evaluation |
| `ui/` | Streamlit app and reusable UI components |
| `data/raw/` | RACE CSVs (`train.csv`, `val.csv`, `test.csv`) |
| `data/processed/` | Feature matrices, metrics, and experiment outputs |
| `models/` | Trained model artifacts for Model A and Model B |
| `notebooks/` | EDA and experimentation notebooks |
| `tests/` | Pytest suite for preprocessing, evaluation, and inference |
| `report/` | Final report document |

## Data format (RACE CSVs)
Place the RACE dataset CSVs in `data/raw/`. Required columns:

| Column | Description |
| --- | --- |
| `article` | Reading passage |
| `question` | Multiple-choice question |
| `A`, `B`, `C`, `D` | Answer options |
| `answer` | Correct option label (`A`-`D`) |

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `data/raw/train.csv`, `data/raw/val.csv`, and `data/raw/test.csv` exist.

NLTK resources are downloaded automatically at runtime.

## Training and evaluation
Model A expects precomputed feature matrices in `data/processed/` (`X_train.npz`, `X_val.npz`, `y_train.npy`, `y_val.npy`) and a TF-IDF vectorizer (`models/model_a/traditional/tfidf_vectorizer.pkl`). The repo includes these artifacts; if you want to regenerate them, use the notebooks or your own preprocessing pipeline based on `src/preprocessing.py`.

Run the training pipelines:
```bash
python src/model_a_train.py
python src/model_b_train.py
```

Run generation evaluation:
```bash
python src/evaluate.py
```

Outputs:
| Path | Produced by |
| --- | --- |
| `models/model_a/traditional/*.pkl` | `model_a_train.py` |
| `models/model_b/traditional/*.pkl` | `model_b_train.py` |
| `data/processed/model_a_comparison.json` | `model_a_train.py` |
| `data/processed/model_b_metrics.json` | `model_b_train.py` |
| `data/processed/eval_metrics.json` | `evaluate.py` |

## Run the Streamlit app
```bash
streamlit run ui/app.py
```

The app loads trained artifacts from `models/` and metrics from `data/processed/`. If those files are missing, the sidebar will show instructions.

## Use the inference module in Python
```python
from inference import (
    load_model_a_artifacts,
    load_model_b_artifacts,
    verify_answer,
    generate_distractors,
    generate_hints,
)

artifacts_a = load_model_a_artifacts()
artifacts_b = load_model_b_artifacts()

result = verify_answer(article, question, options, artifacts_a)
distractors = generate_distractors(article, question, correct_answer, artifacts_b, n=3)
hints = generate_hints(article, question, artifacts_b, n_hints=3)
```

## Tests
```bash
pytest
```

## Notebooks and report
* `notebooks/EDA.ipynb` and `notebooks/experiments.ipynb` document exploratory analysis and experiments.
* `report/Final_Report.docx` contains the final project report.
