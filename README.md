# XGBoost Spam Classification

A binary text classifier that detects spam vs. non-spam (ham) messages using TF-IDF feature extraction and an XGBoost classifier.

## Dataset

`spamdata.csv` contains 5,572 labeled SMS/text messages with two columns:

| Column  | Description                          |
|---------|---------------------------------------|
| `label` | `0` = not spam (ham), `1` = spam      |
| `text`  | Raw message text                      |

The dataset is imbalanced: ~87% ham vs. ~13% spam. 403 duplicate rows are removed during preprocessing.

## Pipeline

1. **Load & clean** — read the CSV, drop duplicates, check for nulls/empty strings.
2. **Split** — 75/25 stratified train/test split (`random_state=42`).
3. **Vectorize** — `TfidfVectorizer` with `max_features=5000` and unigrams + bigrams (`ngram_range=(1, 2)`).
4. **Handle class imbalance** — `scale_pos_weight` computed from the training set's negative/positive ratio.
5. **Train** — `XGBClassifier` (binary logistic objective, 300 estimators, depth 4, learning rate 0.05, with subsampling and L1/L2 regularization).
6. **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, classification report, confusion matrix, and ROC curve.
7. **Inference** — a `check_spam(message)` helper predicts label and spam probability for new, unseen text.

## Results

Evaluated on the held-out test set (1,293 messages):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9745 |
| Precision | 0.9333 |
| Recall    | 0.8589 |
| F1 Score  | 0.8946 |
| ROC-AUC   | 0.9778 |

```
              precision    recall  f1-score   support

    NOT SPAM       0.98      0.99      0.99      1130
        SPAM       0.93      0.86      0.89       163

    accuracy                           0.97      1293
   macro avg       0.96      0.93      0.94      1293
weighted avg       0.97      0.97      0.97      1293
```

## Requirements

- Python 3
- pandas
- scikit-learn
- xgboost
- matplotlib

Install with:

```bash
pip install pandas scikit-learn xgboost matplotlib
```

## Usage

Open and run [notebook.ipynb](notebook.ipynb) end to end. It loads `spamdata.csv`, trains the model, prints evaluation metrics/plots, and demonstrates predictions on sample messages via `check_spam()`:

```python
result, probability = check_spam("Congratulations! You have won a free iPhone. Call now to claim your prize!")
print(result, probability)
# SPAM 0.98
```
