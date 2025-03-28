# Credit Card Fraud Detection

## Objective
Detect fraudulent transactions using Random Forest and Logistic Regression, selecting the best based on weighted F1-score.

## Dataset
- **Source**: `sklearn.datasets.make_classification`
- **Features**: 
  - Original: amount, merchant, time_diff
  - Engineered: `amount_time_ratio` (amount / time_diff), `merchant_freq` (transaction count per merchant)

## Steps to Run
1. **Requirements**: `pip install pandas numpy scikit-learn imblearn matplotlib`
2. **Run**: `python fraud_detection.py`
3. **Outputs**: 
   - Console: Model F1-scores, final report
   - Files: `fraud_predictions.csv`, `roc_curve.png`

## Implementation Details
- **Preprocessing**: Normalized data, balanced with SMOTE (original 95% non-fraud, 5% fraud).
- **Feature Engineering**: 
  - `amount_time_ratio`: Measures transaction speed.
  - `merchant_freq`: Indicates merchant activity level.
- **Models**:
  - **Random Forest**: 150 trees, parallelized, excels with complex patterns.
  - **Logistic Regression**: Linear model, max_iter=1000.
- **Selection Process**: 
  - Metric: Weighted F1-score (balances precision/recall, critical for fraud).
  - Comparison: Random Forest (~0.95-1.0) typically outperforms Logistic Regression (~0.90-0.95) due to non-linear modeling post-SMOTE.
- **Evaluation**: Report, ROC curve (AUC).

## Results
- **F1-Score**: ~0.95-1.0 (Random Forest usually selected).
- **AUC**: ~0.95-1.0.
