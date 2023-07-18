# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load the Dataset
iris_data = load_iris()

# Step 3: Explore the Dataset
print("Features:", iris_data.feature_names)
print("Number of samples:", len(iris_data.data))
print("Number of classes:", len(iris_data.target_names))

# Step 4: Preprocess the Dataset (Not required for the Iris dataset)

# Step 5: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

# Step 6: Train a Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 9: Fine-tune the Model (Optional)

# Step 10: Deploy the Model (Not covered in this code)

