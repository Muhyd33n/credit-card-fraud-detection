# Credit Card Fraud Detection

A machine learning project detecting fraudulent credit card transactions using real-world, 
anonymized transaction data — built to explore how to handle severe class imbalance, a common 
challenge in fintech risk/fraud modeling.

## The Problem

Fraud is rare — in this dataset, only 0.17% of transactions (492 out of 284,807) are fraudulent. 
This makes standard classification approaches misleading: a model predicting "not fraud" for 
every transaction would score 99.8% accuracy while catching zero fraud. This project focuses on 
handling that imbalance properly and evaluating with metrics that actually reflect real-world 
usefulness.

## Dataset

- Source: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions from European cardholders, September 2013
- Features `V1`–`V28` are PCA-anonymized for privacy; `Amount` and `Time` are raw

## Approach

1. Explored class imbalance and distribution of `Amount`/`Time` by fraud vs. non-fraud
2. Scaled `Amount`/`Time`, split data with stratified sampling to preserve class ratio
3. Trained and compared two models:
   - Logistic Regression (baseline, `class_weight='balanced'`)
   - Random Forest (`class_weight='balanced'`, 100 trees)
4. Evaluated using precision, recall, F1, and a precision-recall curve — not accuracy, 
   which is misleading on imbalanced data
5. Examined feature importance to identify which anonymized features drove predictions

## Results

| Model | Precision (fraud) | Recall (fraud) | F1-score |
|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 |
| Random Forest | 0.91 | 0.79 | 0.84 |

Random Forest achieved an **Average Precision score of 0.863**, indicating strong performance 
across all decision thresholds, not just the default.

## Key Takeaway

The two models represent different points on the precision/recall trade-off — logistic regression 
catches more fraud but with far more false alarms; random forest is more selective. In practice, 
the right choice depends on business priorities: cost of missed fraud vs. cost of alert fatigue 
on a fraud review team.

## Tools

Python, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn

## How to Run

1. Clone this repo
2. Create a virtual environment and install requirements: `pip install -r requirements.txt`
3. Download the dataset from the Kaggle link above and place `creditcard.csv` in the `data/` folder
4. Open `notebooks/01_explore_data.ipynb`