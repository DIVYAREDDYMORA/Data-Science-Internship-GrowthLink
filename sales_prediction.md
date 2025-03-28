# Sales Prediction

## Objective
Forecast sales using Gradient Boosting and Ridge Regression, selecting the best based on MSE.

## Dataset
- **Source**: `sklearn.datasets.make_regression`
- **Features**: 
  - Original: ad_spend, promotions, customer_seg
  - Engineered: `ad_promo_interaction` (ad_spend * promotions), `scaled_ad_spend` (ad_spend²)

## Steps to Run
1. **Requirements**: `pip install pandas numpy scikit-learn matplotlib`
2. **Run**: `python sales_prediction.py`
3. **Outputs**: 
   - Console: Model MSEs, final MSE, R²
   - Files: `sales_predictions.csv`, `sales_comparison.png`

## Implementation Details
- **Preprocessing**: Normalized synthetic data.
- **Feature Engineering**: 
  - `ad_promo_interaction`: Captures combined effect.
  - `scaled_ad_spend`: Models non-linear ad impact.
- **Models**:
  - **Gradient Boosting**: 200 trees, learning_rate=0.05, handles complexity.
  - **Ridge**: Regularized linear model, alpha=1.0.
- **Selection Process**: 
  - Metric: MSE (minimizes error in regression).
  - Comparison: Gradient Boosting (~100-150) outperforms Ridge (~150-200) due to non-linear features.
- **Evaluation**: MSE, R², scatter plot.

## Results
- **MSE**: ~100-200 (Gradient Boosting typically selected).
- **R²**: ~0.9 (synthetic data).
