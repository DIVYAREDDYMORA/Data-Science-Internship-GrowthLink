# Movie Rating Prediction

## Objective
Predict movie ratings using Gradient Boosting and Linear Regression, selecting the best based on MSE.

## Dataset
- **Source**: Simulated (no `sklearn` equivalent)
- **Features**: 
  - Original: user_id, movie_id, genre_score, watch_time
  - Engineered: `user_avg_rating` (mean rating per user), `movie_popularity` (rating frequency), `time_rating_interaction` (watch_time * rating)

## Steps to Run
1. **Requirements**: `pip install pandas numpy scikit-learn matplotlib`
2. **Run**: `python movie_rating.py`
3. **Outputs**: 
   - Console: Model MSEs, final MSE, R²
   - Files: `movie_predictions.csv`, `rating_comparison.png`

## Implementation Details
- **Preprocessing**: Normalized features with `StandardScaler`.
- **Feature Engineering**: 
  - `user_avg_rating`: Captures user bias.
  - `movie_popularity`: Indicates movie appeal.
  - `time_rating_interaction`: Models viewing duration’s impact on rating.
- **Models**:
  - **Gradient Boosting**: 150 trees, learning_rate=0.05, excels with non-linear patterns.
  - **Linear Regression**: Simple baseline assuming linear relationships.
- **Selection Process**: 
  - Metric: MSE (minimizes prediction error in regression).
  - Comparison: Gradient Boosting typically has lower MSE (~0.5-0.7) due to handling interactions; Linear Regression (~0.8-1.0) struggles with non-linearity.
- **Evaluation**: MSE, R², scatter plot visualization.

## Results
- **MSE**: ~0.5-1.0 (Gradient Boosting usually selected).
- **R²**: ~0.1-0.3 (synthetic data limits fit).
