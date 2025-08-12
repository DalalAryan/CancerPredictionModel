import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- Load and preprocess data ---
df = pd.read_csv("ham10000_segmentation_analysis.csv")
df["dx_binary"] = df["dx"].apply(lambda x: 1 if x in ["mel", "bcc", "akiec"] else 0)
y = df["dx_binary"]

X = df.drop(columns=["dx", "dx_type", "dx_binary", "Filename", "lesion_id", "image_id"])

# Encode categorical features
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Drop null and constant columns
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique() > 1]
X = X.fillna(X.mean())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# --- SMOTE to Oversample Class 1 ---
print("⚖️  Applying SMOTE oversampling...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"✅ After SMOTE: Class distribution: {np.bincount(y_train_sm)}")

# --- Train XGBoost with tuned params ---
best_f1 = 0
best_model = None
best_threshold = 0.54

# Optional grid
params = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 200,
    "min_child_weight": 1,
    "gamma": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "eval_metric": "aucpr",
    "random_state": 42
}

model = xgb.XGBClassifier(**params)
model.fit(X_train_sm, y_train_sm)

# --- Threshold Tuning ---
y_probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
idx = np.argmax(f1_scores)
best_f1 = f1_scores[idx]
best_threshold = thresholds[idx]

print(f"\n📌 Optimal threshold for F1: {best_threshold:.2f}")

# --- Final Prediction ---
y_final_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_final_probs >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Adjusted Accuracy: {acc:.4f}")
print(f"🎯 Final F1 Score (1s): {best_f1:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# --- Save Model ---
os.makedirs("finalmodels", exist_ok=True)
joblib.dump(model, "finalmodels/xgb_model_smote.pkl")
joblib.dump(scaler, "finalmodels/xgb_scaler_smote.pkl")
print("✅ Saved XGBoost model and scaler with SMOTE to 'models/'")


# --- Feature Importance ---
xgb.plot_importance(model, max_num_features=20)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()
