import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ================== Load Data ==================
df = pd.read_csv("heart.csv")

print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# ================== EDA ==================
print("\nTarget Distribution:\n", df['target'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='target', data=df)
plt.title("Target Distribution")
plt.show()

# ================== Correlation Heatmap ==================
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ================== Outliers (Boxplot) ==================
plt.figure(figsize=(12,6))
sns.boxplot(data=df.drop('target', axis=1))
plt.title("Outlier Detection (Boxplot)")
plt.xticks(rotation=90)
plt.show()

# ================== Features & Target ==================
X = df.drop('target', axis=1)
y = df['target']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================== Models ==================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    if acc > best_score:
        best_score = acc
        best_model = model

# ================== ROC Curve (Best Model) ==================
y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_score = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {roc_score:.2f}")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve (Best Model)")
plt.legend()
plt.show()

print("\nBest Model AUC:", roc_score)

