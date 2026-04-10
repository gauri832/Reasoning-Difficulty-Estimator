#  Reasoning Difficulty Estimator (RDE)

A machine learning system that predicts the difficulty of questions (easy / medium / hard) using:

- Transformer-based embeddings
- Attention entropy & perplexity signals
- Neural network classifier (PyTorch)

##  Pipeline

1. Download datasets (GSM8K, MMLU, BBH, etc.)
2. Extract reasoning signals + embeddings
3. Label difficulty
4. Train RDE model
5. Predict difficulty for new questions

## Results

- Accuracy: ~80%
- Strong performance on semantic understanding
- Improved hard-class detection using embeddings

## Tech Stack

- Python, PyTorch
- HuggingFace Transformers
- Scikit-learn
- SpaCy

##  How to run

```bash
python scripts/01_download_data.py
python scripts/02_extract_signals.py
python scripts/03_label_difficulty.py
python scripts/05_train_rde.py
python scripts/06_test_rde.py

# Structure

arc_project/
├── data/
├── models/
├── scripts/
└── README.md


