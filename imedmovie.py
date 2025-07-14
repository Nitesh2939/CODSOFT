# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")

# Show the first 10 rows
print(df.head(10))

# Describe the dataset (round to 3 decimals)
print(df.describe().round(3))

# Check null values
print("Null values before cleaning:\n", df.isnull().sum())

# Drop all rows with null values
df = df.dropna()

# Show dataset info
print("\nDataset Info:")
print(df.info())

# Shape of cleaned dataset
print("\nShape of data:", df.shape)

# Convert duration from "110 min" to 110 (int)
df['Duration'] = df['Duration'].str.replace(' min', '').astype(int)

# Clean 'Year' column: extract digits from "(2016)"
df['Year'] = df['Year'].str.extract('(\d{4})')  # extract year like 2016
df['Year'] = df['Year'].astype(float)

# Clean 'Votes' column: remove commas and convert to float
df['Votes'] = df['Votes'].str.replace(',', '')
df['Votes'] = df['Votes'].astype(float)

# Drop unnecessary columns
df = df.drop(['Name', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)

# Visualize top genres
plt.figure(figsize=(12, 5))
sns.countplot(y='Genre', data=df, order=df['Genre'].value_counts().index[:10])
plt.title("Top 10 Genres of Indian Movies")
plt.show()

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Genre', 'Director'], drop_first=True)

# Define features (X) and target (y)
X = df.drop(['Rating'], axis=1)
y = df['Rating']

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate both models
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluation for Linear Regression
print("\nLinear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R²:", r2_score(y_test, y_pred_lr))

# Evaluation for Random Forest
print("\nRandom Forest Regressor:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R²:", r2_score(y_test, y_pred_rf))
