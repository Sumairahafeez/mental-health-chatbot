import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("Student Mental health.csv")

# Drop timestamp
df = df.drop(columns=["Timestamp"])

# Rename for easier handling
df.columns = [
    "Gender", "Age", "Course", "StudyYear", "CGPA", "MaritalStatus",
    "Depression", "Anxiety", "PanicAttack", "SoughtTreatment"
]

# Encode labels
df["Depression"] = df["Depression"].map({"Yes": 1, "No": 0})
df["Anxiety"] = df["Anxiety"].map({"Yes": 1, "No": 0})
df["PanicAttack"] = df["PanicAttack"].map({"Yes": 1, "No": 0})

# Encode categorical variables
df = pd.get_dummies(df, columns=["Gender", "Course", "StudyYear", "CGPA", "MaritalStatus"])

# Features & target
X = df.drop(columns=["Depression", "Anxiety", "PanicAttack", "SoughtTreatment"])
y = df[["Depression", "Anxiety", "PanicAttack"]]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Store model accuracies
model_accuracies = {}

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    model_accuracies[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.4f}")

# Find the best model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy: {model_accuracies[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_mental_health_model.pkl")
print(f"Saved the best model: {best_model_name} to 'best_mental_health_model.pkl'")