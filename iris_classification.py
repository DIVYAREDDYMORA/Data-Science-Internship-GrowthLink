import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from sklearn
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                    columns=iris['feature_names'] + ['target'])

# Feature engineering
data['petal_area'] = data['petal length (cm)'] * data['petal width (cm)']  # Petal size metric
data['sepal_ratio'] = data['sepal length (cm)'] / data['sepal width (cm)']  # Shape indicator

# Define features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Random Forest Classifier (ensemble, robust)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Model 2: Support Vector Classifier (kernel-based, non-linear)
svc_model = SVC(kernel='rbf', random_state=42)
svc_model.fit(X_train_scaled, y_train)
svc_pred = svc_model.predict(X_test_scaled)
svc_accuracy = accuracy_score(y_test, svc_pred)

# Model selection based on accuracy (balanced classes)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"SVC Accuracy: {svc_accuracy * 100:.2f}%")
selected_model = rf_model if rf_accuracy > svc_accuracy else svc_model
model_name = "Random Forest" if rf_accuracy > svc_accuracy else "SVC"
print(f"Selected Model: {model_name} (Accuracy: {max(rf_accuracy, svc_accuracy) * 100:.2f}%)")

# Final predictions
y_pred = selected_model.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=iris['target_names'])
conf_matrix = confusion_matrix(y_test, y_pred)

# Detailed output
print("\nFinal Classification Report:\n", report)

# Visualization: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False, 
            xticklabels=iris['target_names'], yticklabels=iris['target_names'])
plt.title(f'Confusion Matrix - {model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance (if Random Forest)
if model_name == "Random Forest":
    feat_importance = pd.DataFrame({'Feature': X.columns, 'Importance': selected_model.feature_importances_})
    print("\nFeature Importance:\n", feat_importance.sort_values('Importance', ascending=False))

# Save predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('iris_predictions.csv', index=False)
