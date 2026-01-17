# 🎓 HSC Result Prediction using Machine Learning

A machine learning–based web application that predicts **HSC (Higher Secondary Certificate) results** for Bangladeshi students using historical academic and socio-economic data.
The system is built using a **Random Forest Regression model** and deployed with **Gradio** for an interactive user interface.

---

## 📌 Project Overview

Predicting student performance can help educators, students, and guardians understand potential academic outcomes early.This project uses machine learning to estimate a student's **HSC GPA** based on factors such as:

- Demographic information
- Family background
- Parental education and occupation
- Lifestyle factors
- SSC academic performance

⚠️ **Note:** This application is intended for **educational and research purposes only**.

---

## 🧠 Machine Learning Approach

- **Problem Type:** Regression
- **Target Variable:** HSC Result (GPA)
- **Model Used:** Random Forest Regressor

### Why Random Forest?

- Handles non-linear relationships well
- Robust to noise and overfitting
- Works effectively with mixed feature types
- Provides strong baseline performance for tabular data

---

## 🧾 Input Features

The model takes the following inputs:

### 🔢 Numerical Features

- Age
- Tuition Fee
- Time Spent with Friends
- SSC Result (GPA)

### 🔤 Categorical Features

- Gender
- Address (Urban / Rural)
- Family Size
- Parents' Living Status
- Mother’s Education Level
- Father’s Education Level
- Mother’s Occupation
- Father’s Occupation
- Relationship Status
- Smoker

All preprocessing (encoding, imputation, scaling) is handled using a **Scikit-learn Pipeline**.

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Libraries & Tools:**
  - NumPy
  - Pandas
  - Scikit-learn
  - Joblib
  - Gradio
  - Matplotlib & Seaborn (EDA)
  - ydata-profiling (EDA)

---

## 🗂️ Project Structure

hsc-result-prediction-ml/

│

├── data/

│   └── bangladesh_student_performance.csv

│

├── notebooks/

│   └── student_hsc_result.ipynb

│

├── models/

│   └── final_model.pkl

│

├── app/

│   └── app.py

│

├── requirements.txt

├── README.md

├── .gitignore

└── LICENSE

## 🚀 How to Run the Application

1️⃣ Clone the Repository

```bash
https://github.com/KzRaihan/hsc-result-prediction-ml.git
cd hsc-result-prediction-ml
```

2️⃣ Create a Virtual Environment

```bash
 conda create -n mlProject python=3.11 -y
```

3️⃣ Activate virtual environment

```bash
 conda activate mlProject
```

4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

5️⃣ Run the Gradio App

```bash
python app.py
```


## 📌 Future Improvements

* Add model explainability (SHAP / feature importance)
* Improve dataset size and diversity
* Deploy publicly on Hugging Face Spaces
* Add multilingual (Bangla) interface
* Compare with XGBoost and Neural Networks


## 👨‍💻 Author

**Md Kamruzzaman Raihan**

BSc in Computer Science & Engineering

Focus: Machine Learning, Deep Learning, Applied AI


## 📄 License

This project is licensed under the **MIT License** and it's open-source and free to use for learning purposes.
