import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic sales data
X, y = make_regression(n_samples=500, n_features=3, noise=10, random_state=42)
data = pd.DataFrame(X, columns=['ad_spend', 'promotions', 'customer_seg'])
data['sales'] = y
data['ad_promo_interaction'] = data['ad_spend'] * data['promotions']  # Synergy effect
data['scaled_ad_spend'] = data['ad_spend'] ** 2  # Non-linear ad impact

# Define features and target
X = data.drop('sales', axis=1)
y = data['sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Gradient Boosting Regressor (non-linear, ensemble)
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_mse = mean_squared_error(y_test, gb_pred)

# Model 2: Ridge Regression (linear, regularized)
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)
ridge_mse = mean_squared_error(y_test, ridge_pred)

# Model selection based on MSE (lower is better)
print(f"Gradient Boosting MSE: {gb_mse:.2f}")
print(f"Ridge Regression MSE: {ridge_mse:.2f}")
selected_model = gb_model if gb_mse < ridge_mse else ridge_model
model_name = "Gradient Boosting" if gb_mse < ridge_mse else "Ridge Regression"
print(f"Selected Model: {model_name} (MSE: {min(gb_mse, ridge_mse):.2f})")

# Final predictions
y_pred = selected_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Detailed output
print(f"Final MSE: {mse:.2f}")
print(f"Final R² Score: {r2:.2f}")

# Visualization: Predicted vs Actual Sales
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title(f'Predicted vs Actual Sales - {model_name}')
plt.savefig('sales_comparison.png')
plt.close()

# Save predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('sales_predictions.csv', index=False)
