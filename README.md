# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🚀 Project Overview
This project uses Machine Learning to predict the risk of heart disease based on medical attributes such as blood pressure, cholesterol, chest pain type, and heart rate.

The goal is to assist early medical diagnosis and reduce the risk of undetected heart disease.

---

## 📊 Dataset
- **Source:** UCI / Kaggle Heart Disease Dataset
- **Original Samples:** 1025 patients
- **After Removing Duplicates:** 302 unique patients
- **Features:** 13 medical attributes
- **Target:**
  - `0` → No Heart Disease
  - `1` → Heart Disease

---

## 📌 Features Description

| Feature | Description |
|---------|-------------|
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| thalach | Maximum heart rate achieved |
| oldpeak | ST depression |
| ca | Number of major vessels |

---

## ⚙️ Data Preprocessing
- ✅ Removed 723 duplicate rows (70% of original data)
- ✅ Train/Test split **before** scaling (prevents data leakage)
- ✅ Feature scaling using `StandardScaler` on train only
- ✅ Outlier detection using Boxplots

---

## 🤖 Machine Learning Models
- Logistic Regression
- Random Forest Classifier

---

## 📈 Model Evaluation

### ⚠️ Why Recall matters in Medical ML?
In healthcare prediction, **Recall is more important than Accuracy** because missing a sick patient (False Negative) can be life-threatening.

### ✅ Cross-Validation (5-Fold)
Models were evaluated using 5-Fold Cross-Validation on training data to ensure reliable performance estimates.

---

## 🏆 Results

| Model | Accuracy | CV Recall |
|-------|----------|-----------|
| Logistic Regression | ~80% | ~90% |
| Random Forest | ~88% | ~92% |

> ⚠️ Note: Earlier versions of this project reported 98% accuracy due to duplicate rows in the dataset. After cleaning, results reflect true model performance.

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

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models & evaluation |
| Matplotlib / Seaborn | Visualizations |
| Joblib | Model saving |

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