# Iris Flower Classification

## Objective
Classify Iris species using Random Forest and SVM, selecting the best based on accuracy.

## Dataset
- **Source**: `sklearn.datasets.load_iris`
- **Features**: 
  - Original: sepal length, sepal width, petal length, petal width
  - Engineered: `petal_area` (length * width), `sepal_ratio` (length / width)

## Steps to Run
1. **Requirements**: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. **Run**: `python iris_classification.py`
3. **Outputs**: 
   - Console: Model accuracies, final report, feature importance (if RF)
   - Files: `iris_predictions.csv`, `confusion_matrix.png`

## Implementation Details
- **Preprocessing**: Normalized features with `StandardScaler`.
- **Feature Engineering**: 
  - `petal_area`: Enhances species differentiation.
  - `sepal_ratio`: Captures shape variations.
- **Models**:
  - **Random Forest**: 100 trees, parallelized, robust to small datasets.
  - **SVC**: RBF kernel for non-linear boundaries.
- **Selection Process**: 
  - Metric: Accuracy (balanced dataset, 50 samples per class).
  - Comparison: Random Forest (~95-100%) often edges out SVC (~90-95%) due to feature importance modeling.
- **Evaluation**: Accuracy, report, confusion matrix.

## Results
- **Accuracy**: ~95-100% (Random Forest typically selected).
- **Key Insight**: Petal features dominate classification.
