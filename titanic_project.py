# Titanic Survival Prediction Project

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import seaborn as sns  # For visualization
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # To split data into training and test sets
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import accuracy_score, confusion_matrix  # For model evaluation

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')  # Read the CSV file
df.head(10)  # Display the first 10 rows of the dataset

# Drop unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  # Drop columns not useful for prediction

# Display statistical summary
print(df.describe().round(3))  # Show stats rounded to 3 decimal places

# Check for null values
print(df.isnull().sum())  # Show count of null values per column

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Visualize survival data
plt.figure(figsize=(14, 5))  # Set figure size
plt.subplot(1, 3, 1)
sns.countplot(x='Survived', data=df)  # Count of survivors vs non-survivors
plt.title('Survival Count')

plt.subplot(1, 3, 2)
sns.countplot(x='Survived', hue='Sex', data=df)  # Survival by gender
plt.title('Survival by Gender')

plt.subplot(1, 3, 3)
sns.countplot(x='Survived', hue='Pclass', data=df)  # Survival by class
plt.title('Survival by Class')
plt.tight_layout()
plt.show()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)  # Convert categorical to numerical

# Split data into features and target
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80/20 split

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=200)  # Create logistic regression instance
log_model.fit(X_train, y_train)  # Train the model

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)  # Create random forest instance
rf_model.fit(X_train, y_train)  # Train the model

# Predict and evaluate Logistic Regression
y_pred_log = log_model.predict(X_test)  # Predict with logistic regression
acc_log = accuracy_score(y_test, y_pred_log)  # Accuracy of logistic regression
conf_log = confusion_matrix(y_test, y_pred_log)  # Confusion matrix

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)  # Predict with random forest
acc_rf = accuracy_score(y_test, y_pred_rf)  # Accuracy of random forest
conf_rf = confusion_matrix(y_test, y_pred_rf)  # Confusion matrix

# Print results
print(f"Logistic Regression Accuracy: {acc_log:.4f}")
print("Confusion Matrix (Logistic Regression):")
print(conf_log)

print(f"Random Forest Accuracy: {acc_rf:.4f}")
print("Confusion Matrix (Random Forest):")
print(conf_rf)

# Print accuracy difference
print(f"Accuracy Difference: {abs(acc_log - acc_rf):.4f}")  # Compare models

# Final statement
print("Model training and prediction complete. Titanic survival classification generated.")
