import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns_vis

# Load and preprocess data
data = sns.load_dataset('titanic')
features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch', 'survived']
data = data[features].copy()
data['sex'] = data['sex'].map({'male': 0, 'female': 1})  # Encode gender
data['age'].fillna(data['age'].median(), inplace=True)   # Handle missing values

# Feature engineering
data['family_size'] = data['sibsp'] + data['parch'] + 1  # Total family on board
data['fare_per_person'] = data['fare'] / data['family_size']  # Economic contribution
data['is_alone'] = (data['family_size'] == 1).astype(int)  # Solo traveler flag

# Define features and target
X = data.drop('survived', axis=1)
y = data['survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features for consistent scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Random Forest Classifier (ensemble, non-linear)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Model 2: Logistic Regression (linear, interpretable)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Model selection based on accuracy (suitable for balanced data)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
selected_model = rf_model if rf_accuracy > lr_accuracy else lr_model
model_name = "Random Forest" if rf_accuracy > lr_accuracy else "Logistic Regression"
print(f"Selected Model: {model_name} (Accuracy: {max(rf_accuracy, lr_accuracy) * 100:.2f}%)")

# Final predictions with selected model
y_pred = selected_model.predict(X_test_scaled)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Detailed output
print("\nFinal Model Classification Report:\n", report)

# Visualization: Confusion Matrix
plt.figure(figsize=(6, 4))
sns_vis.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance (if Random Forest selected)
if model_name == "Random Forest":
    feat_importance = pd.DataFrame({'Feature': X.columns, 'Importance': selected_model.feature_importances_})
    print("\nFeature Importance:\n", feat_importance.sort_values('Importance', ascending=False))

# Save predictions to CSV
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('titanic_predictions.csv', index=False)
