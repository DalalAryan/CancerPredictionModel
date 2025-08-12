import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Data Loading and Preprocessing ---
df = pd.read_csv("ham10000_segmentation_analysis.csv")

# Binary target: 1 if dx in ["mel", "bcc", "akiec"], else 0
df["dx_binary"] = df["dx"].apply(lambda x: 1 if x in ["mel", "bcc", "akiec"] else 0)
y = df["dx_binary"]

# Drop non-informative columns
X = df.drop(columns=[
    "dx", "dx_type", "dx_binary",
    "Filename", "lesion_id", "image_id"
])

# Replace whitespace in column names
X.columns = X.columns.str.replace(r"\s+", "_", regex=True)

# Encode categorical columns
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Remove columns with all missing or constant values
X = X.dropna(axis=1, how='all')
for col in X.columns:
    if X[col].nunique() <= 1:
        X = X.drop(columns=[col])

# Fill any remaining NaNs with column mean
X = X.fillna(X.mean())

# Standardize features (optional for trees, but keeps parity with NN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- CART Model ---
cart = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=6,           # you can tune this
    min_samples_split=10,  # you can tune this
    random_state=42
)
cart.fit(X_train, y_train)

# --- Evaluation ---
y_pred = cart.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"CART Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib

def get_predictions(X):
    model = joblib.load('cart.pkl')  # or your trained model directly
    return model.predict(X)

import joblib

joblib.dump(cart, "finalmodels/cart_model.pkl")
print("✅ Saved CART model to finalmodels/cart_model.pkl")

