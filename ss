import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
# Load your dataset
# Replace 'your_dataset.csv' with your actual dataset path
data = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# If there are missing values, you might want to fill or drop them
# data = data.fillna(method='ffill')

# Define features (X) and target (y)
X = data[['charge_capacity', 'discharge_capacity', 'temperature', 'voltage']]
y = data['cycles']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
# Initialize the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
# Predict on the test data
y_pred = model.predict(X_test)
# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Battery Life Cycles')
plt.show()
import joblib

# Save the trained model to a file
joblib.dump(model, 'battery_life_predictor.pkl')

# Load the model later using:
# model = joblib.load('battery_life_predictor.pkl')
