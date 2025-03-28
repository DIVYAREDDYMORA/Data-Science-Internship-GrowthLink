# Titanic Survival Prediction

## Objective
Predict whether a passenger survived the Titanic disaster by comparing Random Forest and Logistic Regression, selecting the best model based on accuracy.

## Dataset
- **Source**: `seaborn.load_dataset('titanic')`
- **Features**: 
  - Original: Pclass, sex, age, fare, sibsp, parch
  - Engineered: `family_size` (sibsp + parch + 1), `fare_per_person` (fare / family_size), `is_alone` (binary flag)

## Steps to Run
1. **Requirements**: `pip install pandas scikit-learn seaborn matplotlib`
2. **Run**: `python titanic_survival.py`
3. **Outputs**: 
   - Console: Model accuracies, final report, feature importance (if RF)
   - Files: `titanic_predictions.csv`, `confusion_matrix.png`

## Implementation Details
- **Preprocessing**: 
  - Encoded `sex` (male=0, female=1).
  - Filled missing `age` with median.
  - Normalized features using `StandardScaler`.
- **Feature Engineering**: 
  - `family_size`: Captures total family members.
  - `fare_per_person`: Reflects individual economic contribution.
  - `is_alone`: Highlights solo travelers’ survival patterns.
- **Models**:
  - **Random Forest**: 200 trees, max_depth=10, parallelized (`n_jobs=-1`) for efficiency and non-linear modeling.
  - **Logistic Regression**: Linear model with max_iter=1000 for convergence.
- **Selection Process**: 
  - Metric: Accuracy (suitable for balanced dataset, ~60% died, 40% survived).
  - Comparison: Random Forest typically outperforms (~82-85%) due to non-linear feature interactions; Logistic Regression (~78-80%) serves as a simpler baseline.
- **Evaluation**: Accuracy, classification report, confusion matrix visualization.

## Results
- **Accuracy**: ~80-85% (Random Forest usually selected).
- **Key Insight**: Features like `fare`, `sex`, and `fare_per_person` are critical predictors.
