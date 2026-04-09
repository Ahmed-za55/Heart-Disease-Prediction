import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)

# ================== Load Data ==================
df = pd.read_csv("heart.csv")
df = df.drop_duplicates()  # ✅ هنا مباشرة
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# ================== EDA ==================
print("\nTarget Distribution:\n", df['target'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='target', data=df)
plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df.drop('target', axis=1))
plt.title("Outlier Detection (Boxplot)")
plt.xticks(rotation=90)
plt.show()

# ================== Features & Target ==================
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================== Models ==================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    print(f"\n{name}")
    print(f"CV Recall: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Test Accuracy:", round(acc, 2))
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    if acc > best_score:
        best_score = acc
        best_model = model

# ================== ROC Curve ==================
y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_score = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {roc_score:.2f}")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve (Best Model)")
plt.legend()
plt.show()

print("\nBest Model AUC:", round(roc_score, 2))

# ================== Save ==================
joblib.dump(best_model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model & Scaler saved!")