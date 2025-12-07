# 🚕 NYC Taxi Trip Duration Prediction

**End-to-end Machine Learning Pipeline | Feature Engineering | CatBoost Modeling | Streamlit App**

This project predicts the duration of New York City taxi trips using the **Kaggle NYC Taxi Trip Duration Dataset**.
It includes a complete ML workflow: data cleaning, outlier removal, advanced feature engineering, model training, feature selection, evaluation, and a Streamlit interface for real-time predictions.

---

## 🎯 Project Overview

The goal is to estimate **trip duration (in seconds)** based on:

* Pickup & dropoff GPS coordinates
* Datetime features (hour, day, month)
* Distance measurements (Haversine, Manhattan)
* Traffic indicators (rush hour, working day)
* Trip direction (bearing)

The project includes:

* Clean preprocessing pipeline
* Custom feature engineering
* Multiple ML models (Ridge, Random Forest, CatBoost)
* Feature selection
* Evaluation metrics
* Streamlit app for live predictions

---

## 🧠 Final Model Performance (CatBoost)

| Metric    | Score        |
| --------- | ------------ |
| **Model** | CatBoost     |
| **R²**    | **0.802396** |
| **MSE**   | **0.117929** |
| **RMSE**  | **0.343408** |
| **RMSLE** | **0.05371**  |

🔥 **CatBoost achieved strong performance with minimal tuning**, thanks to its ability to capture nonlinear patterns in geospatial and time-based features.

---

## 📁 Project Structure

```
.
├── app/
│   └── streamlit_app.py           # Interactive prediction web app
│
├── data/
│   ├── train.csv                  # Training dataset (local only)
│   ├── train.zip
│   ├── val.csv                    # Validation dataset (local only)
│   └── val.zip
│
├── models/
│   ├── catboost_nyc.cbm           # Final CatBoost model
│   └── ridge.pkl                  # Ridge regression baseline
│
├── notebook/
│   └── EDA.ipynb                  # Full exploratory data analysis
│
├── src/
│   ├── data_load.py               # Data loading + outlier handling
│   ├── feature_engineering.py     # Feature engineering functions
│   ├── pipeline.py                # Main preprocessing pipeline
│   ├── evaluation.py              # Evaluation utilities (R2, RMSE, RMSLE)
│   ├── feature_selection.py       # Random Forest feature importance
│   ├── train_catboost.py          # Train final CatBoost model
│   └── train_ridge.py             # Train Ridge baseline
│
├── .gitignore
└── README.md
```

---

## ✨ Features

### 🧼 **Data Cleaning & Outlier Handling**

* Removes:

  * Negative/zero durations
  * Extremely long trips (> 12 hours)
  * Invalid passenger counts (0, 7)
  * Coordinates outside NYC
* Ensures clean and consistent training data

---

### 🧪 **Feature Engineering**

Implemented in `src/feature_engineering.py`, including:

#### **Distance Features**

* Haversine distance
* Manhattan distance
* Latitude & longitude deltas
* Distance ratio (Manhattan / Haversine)

#### **Geospatial Features**

* Trip bearing (direction)
* Trip midpoint latitude/longitude

#### **Datetime Features**

* Hour, day, month, weekday
* Working day indicator
* Rush hour indicator

#### **Cyclic Encodings**

* `hour_sin`, `hour_cos`
* `day_sin`, `day_cos`
* `weekday_sin`, `weekday_cos`
* `bearing_sin`, `bearing_cos`

#### **Target Transformation**

* `log_trip_duration = log(1 + trip_duration)`
  (reduces skew & improves model performance)

---

### 🤖 Models

#### **1️⃣ Ridge Regression — Baseline**

* Fast linear model
* Uses α = 1 (L2 regularization)
* Helps check basic linear relationships

#### **2️⃣ Random Forest — Feature Selection**

* Provides reliable feature importance
* Captures nonlinear interactions
* Helps determine strongest predictive features

#### **3️⃣ CatBoost — Final Model**

* Gradient boosting algorithm optimized for tabular data
* Handles categorical features natively
* Excellent accuracy with minimal tuning
* Best performance in this project

---

## 💻 Usage

### 🔹 **Train the CatBoost Model**

```bash
python src/train_catboost.py
```

### 🔹 **Train the Ridge Baseline**

```bash
python src/train_ridge.py
```

### 🔹 **Run Feature Selection**

```bash
python src/feature_selection.py
```

---

## 🌐 Run the Streamlit App

Start the live prediction interface:

```bash
streamlit run app/streamlit_app.py
```

The app allows:

* Manual entry of trip coordinates
* Example preset trips
* Map-based visualization
* Instant predictions using the CatBoost model

---

## 📊 Dataset

Dataset from Kaggle:

🔗 **NYC Taxi Trip Duration Dataset**
[https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data)

Place the CSV files inside:

```
data/train.csv
data/val.csv
```

(They are git-ignored to keep the repo lightweight.)

---

## 👥 Authors

**Mahmoud Ashraf**
Machine Learning Engineer

**Mohamed Ehab**
Machine Learning Engineer

