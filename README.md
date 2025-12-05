# 🚕 NYC Taxi Trip Duration Prediction

A machine learning project for predicting the duration of New York City taxi trips using the **Kaggle NYC Taxi Trip Duration dataset**.
This repository includes data preprocessing, powerful ML models (CatBoost & Ridge), feature engineering, model evaluation, and a simple Streamlit interface for predictions.

---

## 🎯 Overview

This project implements a full end-to-end pipeline that predicts how long a taxi trip in NYC will take based on pickup/dropoff coordinates, datetime features, and trip characteristics.

It includes:

* Clean data preprocessing
* Geographic & time-based feature engineering
* Training multiple regression models
* Model evaluation & comparison
* A small UI for real-time predictions

**Key Technologies**

* Python 3.8+
* NumPy, Pandas
* Scikit-learn
* CatBoost
* Streamlit
* Matplotlib / Seaborn for analysis

---

## 📁 Project Structure

```text
.
├── data/
│   ├── train.csv                     # Local training data (ignored in Git)
│   └── val.csv                       # Local validation data (ignored in Git)
│
├── models/
│   ├── random_forest.pkl             # Random Forest model
│   └── catboost_nyc.cbm              # CatBoost model
│
├── notebook/
│   └── EDA.ipynb                     # Exploratory data analysis notebook
│
├── src/
│   ├── data_load.py                  # Data loading & cleaning utilities
│   ├── feature_engineering.py        # Custom feature engineering functions
│   ├── pipeline.py                   # End-to-end preprocessing pipeline
│   └── evaluation.py                 # Model evaluation utilities
│
├── app/
│   └── streamlit_app.py              # Streamlit prediction UI
│
├── .gitignore
└── README.md
```

---

## ✨ Features

### **🧼 Data Preprocessing**

* Removes invalid or extreme values:

  * Negative or unrealistic durations
  * Out-of-bound GPS coordinates
  * Invalid passenger counts
* Handles missing values
* Converts datetime into rich time features

---

### **🧪 Feature Engineering**

* **Haversine distance** (great-circle distance)
* **Manhattan distance**
* Latitude/longitude differences
* Time-based features:

  * Hour, weekday, month
  * Rush-hour flags
  * Weekend indicator
* Bearing angle (trip direction)
* Vendor ID & store-forward flag
* Cyclic encodings (sin/cos)

---

### **🤖 Models**

#### **CatBoostRegressor — Main Model**

* Handles categorical data efficiently
* High accuracy
* Low preprocessing requirements

#### **RandomForestRegressor — Feature Importance**

* Strong performance
* Good for feature importance insight

#### **RidgeRegressor — Baseline**

#### **Evaluation Metrics**

* **R²**
* **RMSE**
* **MAE**
* Handles skew via log-transformed targets

---

## 💻 Usage

### 🔹 **Train the CatBoost Model**

```bash
python train_catboost.py
```

The scripts will:

* Load and clean the dataset
* Apply feature engineering
* Train CatBoost
* Evaluate the model
* Save the trained model under `models/`

---

### 🔹 **Launch the Streamlit App**

```bash
streamlit run app/streamlit_app.py
```

The app provides:

* Input form for trip details
* Validation of NYC coordinates
* Real-time prediction of trip duration
* Simple interface for quick testing

---

## 🔬 Model Details

### **Algorithm**

* Main model: `CatBoostRegressor`
* Loss function: RMSE
* Target: `trip_duration` (seconds)

### **Evaluation**

`src/evaluation.py` computes:

* R²
* RMSE
* MAE

---

## 📊 Dataset

This project uses the official Kaggle dataset:

👉 **NYC Taxi Trip Duration Dataset**
[https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data)

The dataset is **not stored in this repository** due to size.
Place the downloaded files into:

```
/data/train.csv
/data/val.csv
```

---

## 👥 Authors

**Mahmoud Ashraf**
Machine Learning Engineer

**Mohamed Ehab**
Machine Learning Engineer
