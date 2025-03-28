import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Simulated dataset with enhanced feature engineering
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'user_id': np.random.randint(1, 100, n_samples),
    'movie_id': np.random.randint(1, 200, n_samples),
    'genre_score': np.random.uniform(0, 1, n_samples),  # Simulated genre preference
    'watch_time': np.random.normal(120, 30, n_samples),  # Viewing duration in minutes
    'rating': np.random.uniform(1, 5, n_samples)  # Target variable
})
data['user_avg_rating'] = data.groupby('user_id')['rating'].transform('mean')  # User rating bias
data['movie_popularity'] = data.groupby('movie_id')['rating'].transform('count') / n_samples  # Popularity index
data['time_rating_interaction'] = data['watch_time'] * data['rating']  # Interaction effect

# Define features and target
X = data.drop('rating', axis=1)
y = data['rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Gradient Boosting Regressor (non-linear, ensemble)
gb_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_mse = mean_squared_error(y_test, gb_pred)

# Model 2: Linear Regression (simple, linear)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_mse = mean_squared_error(y_test, lr_pred)

# Model selection based on MSE (lower is better)
print(f"Gradient Boosting MSE: {gb_mse:.2f}")
print(f"Linear Regression MSE: {lr_mse:.2f}")
selected_model = gb_model if gb_mse < lr_mse else lr_model
model_name = "Gradient Boosting" if gb_mse < lr_mse else "Linear Regression"
print(f"Selected Model: {model_name} (MSE: {min(gb_mse, lr_mse):.2f})")

# Final predictions
y_pred = selected_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Detailed output
print(f"Final MSE: {mse:.2f}")
print(f"Final R² Score: {r2:.2f}")

# Visualization: Predicted vs Actual Ratings
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([1, 5], [1, 5], 'r--')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title(f'Predicted vs Actual Ratings - {model_name}')
plt.savefig('rating_comparison.png')
plt.close()

# Save predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('movie_predictions.csv', index=False)
