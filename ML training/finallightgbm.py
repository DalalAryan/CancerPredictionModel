import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from lightgbm import LGBMClassifier

# Load the data
df = pd.read_csv("ham10000_segmentation_analysis.csv")

# Binary target: 1 if dx in ["mel", "bcc", "akiec"], else 0
df["dx_binary"] = df["dx"].apply(lambda x: 1 if x in ["mel", "bcc", "akiec"] else 0)
y_dx = df["dx_binary"]

# Set features (drop original and encoded targets, and non-informative columns)
X = df.drop(columns=[
    "dx", "dx_type", "dx_binary",
    "Filename", "lesion_id", "image_id"
])

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category")

# Remove columns with all missing or constant values
X = X.dropna(axis=1, how='all')
for col in X.columns:
    if X[col].nunique() <= 1:
        X = X.drop(columns=[col])

# Train/test split
X_train, X_test, y_dx_train, y_dx_test = train_test_split(
    X, y_dx, test_size=0.2, random_state=42
)

params_2 = {
    "n_estimators": 300,
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
    # Keep any other parameters you're already using
}

# --- Model: Predict dx_binary ---
model_dx = LGBMClassifier(objective="binary", class_weight="balanced", **params_2)
model_dx.fit(X_train, y_dx_train, categorical_feature=categorical_cols) 
pred_dx = model_dx.predict(X_test)
# Get predicted probabilities
pred_probs = model_dx.predict_proba(X_test)[:, 1]

# Custom threshold (e.g., 0.3 instead of 0.5)
threshold = 0.26
custom_pred_dx = (pred_probs >= threshold).astype(int)
acc_dx = accuracy_score(y_dx_test, pred_dx)
print(f"Accuracy for dx (mel/bcc/akiec vs other): {acc_dx:.4f}")

from sklearn.metrics import confusion_matrix, classification_report

# Evaluate custom threshold
print(f"\nConfusion Matrix (threshold = {threshold}):")
print(confusion_matrix(y_dx_test, custom_pred_dx))

print("\nClassification Report (custom threshold):")
print(classification_report(y_dx_test, custom_pred_dx, target_names=["Benign", "Malignant"]))

from sklearn.metrics import classification_report, accuracy_score

# Predict on training set
train_preds = model_dx.predict(X_train)
train_probs = model_dx.predict_proba(X_train)[:, 1]

print("\n📋 Training Classification Report (default threshold):")
print(classification_report(y_dx_train, train_preds, target_names=["Benign", "Malignant"]))
import joblib
joblib.dump(model_dx, "finalmodels/lgb_model.pkl")
print("✅ Saved LightGBM model to finalmodels/lgb_model.pkl")

joblib.dump(X.columns.tolist(), "finalmodels/lgb_features.pkl")
print("✅ Saved LightGBM feature columns to finalmodels/lgb_features.pkl")
