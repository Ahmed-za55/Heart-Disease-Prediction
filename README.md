# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🚀 Project Overview

This project uses Machine Learning to predict the risk of heart disease based on medical attributes such as blood pressure, cholesterol, chest pain type, and heart rate.

The goal is to assist early medical diagnosis and reduce the risk of undetected heart disease.

> ✅ Includes a **Streamlit Web App** — enter patient data and get instant predictions!

---

## 🌐 Web App Preview

Run the app locally and interact with the model through a clean medical UI:

```bash
streamlit run app.py
```

**Features:**
- Input 13 medical attributes via an interactive form
- Instant prediction: HIGH RISK ⚠️ or LOW RISK ✅
- Probability percentage with visual bar
- Medical disclaimer footer

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI / Kaggle Heart Disease Dataset |
| Original Samples | 1025 patients |
| After Cleaning | 302 unique patients |
| Features | 13 medical attributes |
| Target | 0 → No Disease / 1 → Disease |

> ⚠️ The original dataset contained **723 duplicate rows (70%)** — removed before training to ensure honest evaluation.

---

## 📌 Features Description

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Sex (0 = Female, 1 = Male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–4) |
| `thal` | Thalassemia type (0–3) |

---

## ⚙️ Data Preprocessing

- ✅ Removed 723 duplicate rows (70% of original data)
- ✅ Train/Test split **before** scaling (prevents data leakage)
- ✅ Feature scaling using `StandardScaler` on train data only
- ✅ Outlier detection using Boxplots

---

## 🤖 Machine Learning Models

| Model | Type |
|-------|------|
| Logistic Regression | Baseline |
| Random Forest Classifier | Best Model ⭐ |

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
| Logistic Regression | ~77% | ~90% | — |
| Random Forest | ~84% | ~92% | **0.88** ⭐ |

---

## 📊 Visualizations

| Visualization | Output |
|---------------|--------|
| Target Distribution | ![](./images/Figure_1.png) |
| Correlation Heatmap | ![](./images/Figure_2.png) |
| Outlier Detection | ![](./images/Figure_3.png) |
| Logistic Regression — Confusion Matrix | ![](./images/Figure_4.png) |
| Random Forest — Confusion Matrix | ![](./images/Figure_5.png) |
| ROC Curve (AUC = 0.88) | ![](./images/Figure_6.png) |

---

## 📉 Key Insights

- Chest pain type (`cp`) is the strongest predictor of heart disease
- Maximum heart rate (`thalach`) significantly impacts diagnosis
- Dataset contained 70% duplicate rows — cleaning was critical for honest evaluation
- Random Forest outperforms Logistic Regression on this dataset

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Ahmed-za55/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python main.py
```

### 4. Launch the Web App
```bash
streamlit run app.py
```

> App runs at `http://localhost:8501`

---

## 💾 Saved Artifacts

After running `main.py`, two files are saved automatically:

| File | Description |
|------|-------------|
| `heart_model.pkl` | Trained Random Forest model |
| `scaler.pkl` | Fitted StandardScaler for new predictions |

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── images/
│   ├── Figure_1.png        ← Target Distribution
│   ├── Figure_2.png        ← Correlation Heatmap
│   ├── Figure_3.png        ← Outlier Detection
│   ├── Figure_4.png        ← LR Confusion Matrix
│   ├── Figure_5.png        ← RF Confusion Matrix
│   └── Figure_6.png        ← ROC Curve
│
├── main.py                 ← Training script
├── app.py                  ← Streamlit Web App
├── heart.csv               ← Dataset
├── heart_model.pkl         ← Saved model
├── scaler.pkl              ← Saved scaler
├── requirements.txt        ← Dependencies
└── README.md
```

---

## 👨‍💻 Author

**Ahmed Sameh** — Faculty of Artificial Intelligence, Kafrelsheikh University

[![GitHub](https://img.shields.io/badge/GitHub-Ahmed--za55-181717?style=flat&logo=github)](https://github.com/Ahmed-za55)