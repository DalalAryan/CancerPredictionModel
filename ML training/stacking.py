import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("ham10000_segmentation_analysis.csv")
df["dx_binary"] = df["dx"].apply(lambda x: 1 if x in ["mel", "bcc", "akiec"] else 0)
y = df["dx_binary"]

# Prepare features
X = df.drop(columns=["dx", "dx_type", "dx_binary", "Filename", "lesion_id", "image_id"])
original_columns = X.columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique() > 1].fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Restore DataFrame structure for LightGBM
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
for col in categorical_cols:
    if col in X_train_df.columns:
        X_train_df[col] = X_train_df[col].astype("category")
        X_test_df[col] = X_test_df[col].astype("category")

# Load custom trained models
models_dir = "finalmodels"
xgb_model = joblib.load(os.path.join(models_dir, "xgb_model_smote.pkl"))
lgb_model = joblib.load(os.path.join(models_dir, "lgb_model.pkl"))
svm_model = joblib.load(os.path.join(models_dir, "svm_model.pkl"))
cart_model = joblib.load(os.path.join(models_dir, "cart_model.pkl"))

# Predict helper
def get_model_preds(model, X, is_lgb=False, is_cart=False):
    if is_cart:
        return model.predict(X)  # use hard labels
    elif is_lgb:
        return model.predict_proba(X)[:, 1]  # soft prob
    else:
        return model.predict_proba(X)[:, 1]

# Base model predictions
train_preds = np.column_stack([
    get_model_preds(xgb_model, X_train),
    get_model_preds(lgb_model, X_train_df, is_lgb=True),
    get_model_preds(svm_model, X_train),
    get_model_preds(cart_model, X_train),
])

test_preds = np.column_stack([
    get_model_preds(xgb_model, X_test),
    get_model_preds(lgb_model, X_test_df, is_lgb=True),
    get_model_preds(svm_model, X_test),
    get_model_preds(cart_model, X_test),
])



# Meta-model: XGBoost
meta_model = xgb.XGBClassifier(
    objective="binary:logistic",
    learning_rate=0.05,
    n_estimators=100,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

meta_model.fit(train_preds, y_train)
final_probs = meta_model.predict_proba(test_preds)[:, 1]
final_preds = (final_probs >= 0.5).astype(int)

# Evaluation
acc = accuracy_score(y_test, final_preds)
print(f"\n✅ Final Stacking Accuracy: {acc:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, final_preds))

# Save meta-model
joblib.dump(meta_model, os.path.join(models_dir, "stacking_meta_model.pkl"))
print("✅ Saved stacking meta-model to finalmodels/stacking_meta_model.pkl")

