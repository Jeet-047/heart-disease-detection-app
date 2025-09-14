# Heart Disease Prediction App

A web application for predicting the presence of heart disease using a trained **Random Forest** machine learning model. Built with **Flask**, this app provides a user-friendly interface for entering patient data and viewing prediction results.

---

## 🩺 Overview

This app allows users to input patient attributes such as age, chest pain type, cholesterol, and more. The backend uses a Random Forest model (`heart_disease_rf.pkl`) to predict whether the patient is likely to have heart disease.

---

## 🚀 Features

- **Modern UI**: Clean, responsive frontend using Tailwind CSS.
- **Instant Prediction**: Get results based on your input data.
- **Detailed Report**: See a summary of your input and the prediction.
- **Easy to Use**: Simple form for entering patient details.

---

## 📸 App Screenshots

> _Add your own screenshots here!_

| Home Page | Prediction Form | Prediction Report |
|-----------|----------------|------------------|
| ![Home](images/home.png) | ![Form](images/form.png) | ![Report](images/report.png) |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Jeet-047/heart-disease-detection-app.git
cd heart-disease-detection-app
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed.

```bash
pip install Flask numpy scikit-learn joblib
```

### 3. Add the Model File

Place your trained model file named `heart_disease_rf.pkl` in the project root directory.

> **Note:** If you don’t have this file, train a Random Forest model and save it using `joblib`.

### 4. Run the App

```bash
python app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 📝 Usage

1. **Home Page:** Click "Get Started" to go to the prediction form.
2. **Prediction Form:** Enter patient details (name, age, chest pain type, etc.).
3. **Submit:** Click "Predict" to view the result.
4. **Report:** See a summary of your input and the prediction outcome.

---

## ⚙️ Project Structure

```
heart-disease-detection-app/
│
├── app.py
├── heart_disease_rf.pkl
├── templates/
│   ├── index.html
│   ├── prediction_form.html
│   └── report.html
├── static/
│   └── (optional: images, CSS)
└── README.md
```

---

## 📊 Model Information

- **Algorithm:** Random Forest Classifier
- **Input Features:** Age, Chest Pain Type, Resting Blood Pressure, Cholesterol, Max Heart Rate, Oldpeak, Number of Major Vessels, Thalassemia

---

## ❗ Disclaimer

This app is for educational and demonstration purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## 📬 Contact

For questions or suggestions, open an issue or contact [Jeet-047](https://github.com/Jeet-047).

---
