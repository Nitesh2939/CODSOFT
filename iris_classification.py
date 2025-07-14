# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("IRIS.csv")
print("First 10 rows of the dataset:\n", df.head(10))

# Describe dataset
print("\nStatistical Summary:\n", df.describe().round(3))

# Check for nulls
print("\nNull Values Count:\n", df.isnull().sum())

# Dataset Info
print("\nDataset Info:")
print(df.info())

# Shape and unique values
print("\nShape of dataset:", df.shape)
print("Unique species:", df['species'].unique())

# Visualization: Count of species
plt.figure(figsize=(6, 4))
sns.countplot(x='species', data=df)
plt.title("Species Count")
plt.show()

# Pair plot of features by species
sns.pairplot(df, hue='species')
plt.suptitle("Pair Plot of Iris Features by Species", y=1.02)
plt.show()

# Label encoding for species (target)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])  # setosa=0, versicolor=1, virginica=2

# Split data into features and target
X = df.drop('species', axis=1)
y = df['species']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

# Fit and evaluate
print("\nModel Evaluation:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy comparison
lr_acc = accuracy_score(y_test, models['Logistic Regression'].predict(X_test))
dt_acc = accuracy_score(y_test, models['Decision Tree'].predict(X_test))
knn_acc = accuracy_score(y_test, models['KNN'].predict(X_test))

print("\nAccuracy Comparison:")
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"Decision Tree: {dt_acc:.4f}")
print(f"KNN: {knn_acc:.4f}")
print("Accuracy Difference:", round(max(lr_acc, dt_acc, knn_acc) - min(lr_acc, dt_acc, knn_acc), 4))

# Final statement
print("\n✅ Iris species classification successfully completed.")
