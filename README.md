# Reasoning Difficulty Estimator (RDE)

A machine learning system that predicts question difficulty (`easy` / `medium` / `hard`) and routes inference through ARC (Adaptive Reasoning Controller) to reduce token usage.

## Pipeline

1. Download datasets.
2. Extract reasoning signals and embeddings.
3. Label difficulty.
4. Train RDE.
5. Run ARC controller and evaluate savings.

## Tech Stack

- Python, PyTorch
- Hugging Face Transformers
- Scikit-learn
- spaCy

## How To Run

```bash
python scripts/01_download_data.py
python scripts/02_extract_signals.py
python scripts/03_label_difficulty.py
python scripts/05_train_rde.py
python scripts/06_test_rde.py
python scripts/10_evaluate.py
```

## ARC React UI

Launch the local HTML/React dashboard:

```bash
python scripts/arc_web_server.py --backend hf_small
```

Then open:

```bash
http://127.0.0.1:8501
```

## Structure

- `data/`
- `models/`
- `scripts/`
- `webui/`
