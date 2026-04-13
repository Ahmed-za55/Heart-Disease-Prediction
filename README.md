# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🚀 Project Overview

This project uses Machine Learning to predict the risk of heart disease based on medical attributes such as blood pressure, cholesterol, chest pain type, and heart rate.

The goal is to assist early medical diagnosis and reduce the risk of undetected heart disease.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI / Kaggle Heart Disease Dataset |
| Original Samples | 1025 patients |
| After Cleaning | 302 unique patients |
| Features | 13 medical attributes |
| Target | 0 → No Disease / 1 → Disease |

---

## 📌 Features Description

| Feature | Description |
|---------|-------------|
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `thalach` | Maximum heart rate achieved |
| `oldpeak` | ST depression |
| `ca` | Number of major vessels |

---

## ⚙️ Data Preprocessing

- ✅ Removed 723 duplicate rows (70% of original data)
- ✅ Train/Test split **before** scaling (prevents data leakage)
- ✅ Feature scaling using `StandardScaler` on train data only
- ✅ Outlier detection using Boxplots

---

## 🤖 Machine Learning Models

- Logistic Regression
- Random Forest Classifier

---

## 📈 Model Evaluation

### ⚠️ Why Recall matters in Medical ML?
In healthcare, **Recall is more important than Accuracy** because missing a sick patient (False Negative) can be life-threatening.

### ✅ Cross-Validation
Models evaluated using **5-Fold Cross-Validation** on training data for reliable performance estimates.

---

## 🏆 Results

| Model | Accuracy | CV Recall | AUC |
|-------|----------|-----------|-----|
| Logistic Regression | ~77% | ~90% | - |
| Random Forest | ~84% | ~92% | 0.88 |

> ⚠️ Earlier versions reported 99% accuracy due to 723 duplicate rows. After cleaning, results reflect true model performance.

---

## 📊 Visualizations

| Visualization | Output |
|---------------|--------|
| Target Distribution | ![](./images/Figure_1.png) |
| Correlation Heatmap | ![](./images/Figure_2.png) |
| Outlier Detection | ![](./images/Figure_3.png) |
| Logistic Regression - Confusion Matrix | ![](./images/Figure_4.png) |
| Random Forest - Confusion Matrix | ![](./images/Figure_5.png) |
| ROC Curve | ![](./images/Figure_6.png) |

---

## 📉 Key Insights

- Chest pain type (`cp`) is the strongest predictor of heart disease
- Maximum heart rate (`thalach`) significantly impacts diagnosis
- Dataset contained 70% duplicate rows — cleaning was critical for honest evaluation

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

---

## 🚀 How to Run

```bash
git clone https://github.com/Ahmed-za55/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
pip install -r requirements.txt
python main.py
```

---

## 💾 Saved Artifacts

After running, two files are saved:
- `heart_model.pkl` → Trained Random Forest model
- `scaler.pkl` → Fitted StandardScaler for new predictions

---

## 👨‍💻 Author

**Ahmed Sameh** — Faculty of Artificial Intelligence, Kafrelsheikh University