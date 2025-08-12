import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

# 1. Load and preprocess data
df = pd.read_csv("ham10000_segmentation_analysis.csv")
df["dx_binary"] = df["dx"].apply(lambda x: 1 if x in ["mel", "bcc", "akiec"] else 0)
y = df["dx_binary"]

X = df.drop(columns=["dx", "dx_type", "dx_binary", "Filename", "lesion_id", "image_id"])
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique() > 1].fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Define SVM pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        probability=True,
        class_weight={0: 1, 1: 1.25},
        random_state=42
    ))
])

# 3. Grid search for best C/gamma
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.1, 1],
}

grid = GridSearchCV(
    pipeline, param_grid, cv=5,
    scoring='f1', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print("🔧 Best Parameters:", grid.best_params_)

# 4. Evaluate
best_model = grid.best_estimator_
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Optional: Tune threshold to boost recall
threshold = 0.5
y_thresh = (y_proba >= threshold).astype(int)

print("\n✅ Threshold-tuned Classification Report (threshold = 0.35):")
print(classification_report(y_test, y_thresh))

# Accuracy and AUC
acc = accuracy_score(y_test, y_thresh)
auc = roc_auc_score(y_test, y_proba)
print(f"🔹 Accuracy: {acc:.4f}")
print(f"🔹 ROC AUC: {auc:.4f}")

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_thresh)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix (threshold=0.35)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 6. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.plot(rec, prec, label="PR Curve")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()

import joblib

joblib.dump(best_model, "finalmodels/svm_model.pkl")
print("✅ Saved SVM model to finalmodels/svm_model.pkl")