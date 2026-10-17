"""
ML classique : même idée que XGBoost sur transactions, mais version minimale sklearn.
Classical ML: same idea as XGBoost on transactions, minimal sklearn version.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Features fictives : montant, heure, nb_marchands / fake features
rng = np.random.default_rng(42)
n = 500
X = rng.normal(size=(n, 3))
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("=== ML pipeline / pipeline ML ===")
print("1. Données + labels  →  2. split  →  3. fit  →  4. predict  →  5. métriques")
print()
print(classification_report(y_test, pred))

print("=== DL (images) en une phrase / DL for images in one line ===")
print("FR: pixels → CNN (souvent pré-entraînée) → fine-tune → même preprocessing en API")
print("EN: pixels → CNN (often pretrained) → fine-tune → same preprocessing in API")
