import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
# ── 1. Load & Explore ──────────────────────────────────────────
df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print("\nClass Distribution:")
print(df["Class"].value_counts())
print("\nFirst 5 rows:")
print(df.head())
# ── 2. Preprocess ──────────────────────────────────────────────
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"]   = scaler.fit_transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
# ── 3. Handle Class Imbalance ──────────────────────────────────
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE: ", pd.Series(y_train_res).value_counts().to_dict())
print("\nTraining Random Forest... (this may take a minute)")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)
print("Training complete!")

# ── 5. Evaluate ────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ── 6. Confusion Matrix Plot ───────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ── 7. ROC Curve Plot ──────────────────────────────────────────
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("\nDone! Plots saved as confusion_matrix.png and roc_curve.png")