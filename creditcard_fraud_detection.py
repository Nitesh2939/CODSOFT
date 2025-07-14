# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset as a CSV file
df = pd.read_csv("creditcard.csv")  # Make sure the CSV file is in the same folder
print("First 10 rows of the dataset:\n", df.head(10))  # Show first 10 rows

# Drop the unnecessary 'Time' column
df.drop(columns=['Time'], inplace=True)

# Calculate statistical values and round them to 3 decimal places
print("\nStatistical Summary:\n", df.describe().round(3))

# Check for null values in each column
print("\nNull Values Count:\n", df.isnull().sum())

# Handle null values by filling them with the column mean (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Extract all information about the dataset
print("\nDataset Info:")
print(df.info())

# Check the shape of the dataset
print("\nShape of dataset:", df.shape)

# Visualize the class distribution (fraud vs non-fraud)
sns.countplot(x='Class', data=df)
plt.title("Transaction Class Distribution (0 = Genuine, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Normalize the 'Amount' column using StandardScaler
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Data preprocessing (No categorical data, so no one-hot encoding needed here)

# Split the cleaned data into features (X) and label (y)
X = df.drop('Class', axis=1)  # Independent variables
y = df['Class']              # Dependent variable

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate the Logistic Regression model
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_predictions))
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", round(lr_accuracy, 4))

# Evaluate the Random Forest model
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", round(rf_accuracy, 4))

# Compare accuracy
accuracy_difference = abs(lr_accuracy - rf_accuracy)
print("\nAccuracy Difference between models:", round(accuracy_difference, 4))

# Final message
print("\n✅ Credit Card Fraud Classification Generated Successfully!")
