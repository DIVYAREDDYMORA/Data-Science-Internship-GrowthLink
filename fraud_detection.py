import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Generate synthetic fraud data
X, y = make_classification(n_samples=1000, n_features=3, n_classes=2, weights=[0.95, 0.05], random_state=42)
data = pd.DataFrame(X, columns=['amount', 'merchant', 'time_diff'])
data['fraud'] = y
data['amount_time_ratio'] = data['amount'] / (data['time_diff'] + 1e-6)  # Transaction speed
data['merchant_freq'] = data.groupby('merchant')['amount'].transform('count')  # Merchant usage frequency

# Define features and target
X = data.drop('fraud', axis=1)
y = data['fraud']

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Random Forest Classifier (ensemble, robust)
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_report = classification_report(y_test, rf_pred, output_dict=True)
rf_f1 = rf_report['weighted avg']['f1-score']

# Model 2: Logistic Regression (linear, probabilistic)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_report = classification_report(y_test, lr_pred, output_dict=True)
lr_f1 = lr_report['weighted avg']['f1-score']

# Model selection based on weighted F1-score (balances precision/recall)
print(f"Random Forest F1-Score: {rf_f1:.2f}")
print(f"Logistic Regression F1-Score: {lr_f1:.2f}")
selected_model = rf_model if rf_f1 > lr_f1 else lr_model
model_name = "Random Forest" if rf_f1 > lr_f1 else "Logistic Regression"
print(f"Selected Model: {model_name} (F1-Score: {max(rf_f1, lr_f1):.2f})")

# Final predictions
y_pred = selected_model.predict(X_test_scaled)
y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]
report = classification_report(y_test, y_pred)

# Detailed output
print("\nFinal Classification Report:\n", report)

# Visualization: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {model_name}')
plt.legend(loc='best')
plt.savefig('roc_curve.png')
plt.close()

# Save predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('fraud_predictions.csv', index=False)
