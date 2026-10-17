"""
Minimal machine learning flow — read every comment before running.

Run from Training folder:
    python examples/see_ml_flow.py
"""

# --- Step 0: imports ---
# numpy: fast arrays of numbers (features and labels)
# sklearn: classic ML tools (split data, train model, metrics)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- Step 1: make fake "transaction-like" data ---
# Each ROW is one example. Each COLUMN is one feature (a number the model sees).
# In a real project, columns come from SQL, CSV, or feature engineering code.

rng = np.random.default_rng(42)
n_samples = 400

# feature 0: amount-like, feature 1: hour-like, feature 2: noise
X = rng.normal(size=(n_samples, 3))

# label: 1 if simple rule fires, else 0 (pretend we know true fraud from history)
y = ((X[:, 0] + 0.6 * X[:, 1]) > 0).astype(int)

# --- Step 2: split ---
# train = model learns here; test = pretend "new" data the model never saw

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --- Step 3: create and train model ---
# RandomForest: many decision trees vote (similar family to XGBoost in interviews)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)  # "fit" = training: adjust internal rules from data

# --- Step 4: predict on held-out test set ---

y_pred = model.predict(X_test)

# --- Step 5: measure quality ---

print("=== Training finished. Metrics on test data ===\n")
print(classification_report(y_test, y_pred))

# --- Step 5b: one production-style call ---
# New row must have SAME 3 features in SAME order as training.

new_transaction = np.array([[1.2, -0.5, 0.1]])
score = model.predict(new_transaction)
print("New sample predicted class:", score[0])

print("\n=== How this relates to the drawing project ===")
print("Table ML (this script): rows = objects, columns = features, label = category.")
print("Drawing MVP: rules/OpenCV find regions; optional ML classifies symbol crops.")
print("Deep learning: neural net learns from pixels when rules/templates fail.")
