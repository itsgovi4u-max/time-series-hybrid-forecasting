# Advanced Time Series Forecasting with Prophet + LSTM Hybrid Model

## 📌 Project Overview
This project implements a hybrid time series forecasting model combining:
- Facebook Prophet (for trend and seasonality)
- LSTM Neural Network (for complex temporal patterns)

The goal is to compare:
1. Standalone Prophet Model  
2. Standalone LSTM Model  
3. Hybrid Prophet + LSTM Model  

and evaluate them using RMSE, MAE, and MAPE.

---

## 📊 Dataset
The dataset used in this project is taken from Kaggle:

Kaggle Dataset:  
Air Passengers Time Series Dataset  
https://www.kaggle.com/datasets/rakannimer/air-passengers

If required, the dataset can also be generated synthetically inside the code.

---

## 🧠 Methodology

### Step 1: Data Preparation
- Load time series dataset  
- Handle missing values  
- Normalize data for deep learning model  

### Step 2: Baseline Models
#### Model 1: Prophet
- Captures trend and seasonality  
- Produces baseline forecast  

#### Model 2: LSTM (Deep Learning)
- Uses sliding window technique  
- Learns temporal dependencies  

### Step 3: Hybrid Model
- Prophet removes trend/seasonality  
- LSTM learns patterns from residuals  
- Final forecast combines both outputs  

---

## 📈 Evaluation
Models are compared using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

Rolling cross-validation is used for robust evaluation.

---

## 📁 Files in Repository
- main.py → Complete runnable project code  
- README.md → Project explanation (this file)

---

## ✅ Conclusion
The hybrid Prophet + LSTM model provides better forecasting performance compared to individual models, especially for non-linear time series with multiple seasonal patterns.
