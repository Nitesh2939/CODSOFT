# Loading necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset and show first 10 rows
df = pd.read_csv("advertising.csv")
print("First 10 rows of the dataset:\n", df.head(10))

# Drop unnecessary columns (none in this dataset, but code is included if needed)
# df = df.drop(['UnnecessaryColumn'], axis=1)

# Calculate statistical values rounded to 3 decimal places
print("\nStatistical Summary:\n", df.describe().round(3))

# Check for null values
print("\nNull values in each column:\n", df.isnull().sum())

# Fill null values with mean (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Extract full dataset info
print("\nDataset Info:")
print(df.info())

# Check shape of dataset
print("\nShape of dataset:", df.shape)

# Visualization of Sales by Advertisement Spending
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(data=df, x='TV', y='Sales', color='blue')
plt.title("TV Ad Spend vs Sales")

plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Radio', y='Sales', color='green')
plt.title("Radio Ad Spend vs Sales")

plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Newspaper', y='Sales', color='red')
plt.title("Newspaper Ad Spend vs Sales")

plt.tight_layout()
plt.show()

# No categorical columns present, so no need for One-Hot Encoding
# If they were, we would use pd.get_dummies(df)

# Split into features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions with both models
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

# Evaluate both models
print("\nModel Evaluation:")

print("\nLinear Regression:")
print("MAE:", round(mean_absolute_error(y_test, lr_preds), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, lr_preds)), 3))
print("R2 Score:", round(r2_score(y_test, lr_preds), 3))

print("\nRandom Forest Regressor:")
print("MAE:", round(mean_absolute_error(y_test, rf_preds), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, rf_preds)), 3))
print("R2 Score:", round(r2_score(y_test, rf_preds), 3))

# Compare accuracy (R2 score difference)
lr_r2 = r2_score(y_test, lr_preds)
rf_r2 = r2_score(y_test, rf_preds)
accuracy_diff = abs(lr_r2 - rf_r2)

print("\nAccuracy Difference between models (R²):", round(accuracy_diff, 4))

# Final statement
print("\n✅ Sales prediction project completed successfully.")
