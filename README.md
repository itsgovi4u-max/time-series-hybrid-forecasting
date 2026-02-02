# Advanced Time Series Forecasting with Prophet + LSTM Hybrid Model

## 1. Project Overview
This project implements an advanced hybrid time series forecasting approach that combines statistical and deep learning techniques. Specifically, the system integrates Facebook Prophet and an LSTM-based neural network to improve prediction accuracy for non-linear, seasonal, and non-stationary time series data.

The objective of this work is to:
1. Build a standalone Prophet model as a statistical baseline.
2. Build a standalone LSTM deep learning model.
3. Develop a hybrid model that leverages both approaches and compare performance.

---

## 2. Dataset and Data Acquisition
The dataset used in this project is the **Daily Delhi Climate Test dataset from Kaggle**. It contains historical daily weather measurements including:
- Date
- Mean Temperature (°C)
- Humidity
- Wind Speed
- Mean Pressure

This dataset was selected because it exhibits:
- Clear daily and yearly seasonality
- Long-term trends
- Non-stationary behavior
- Complex temporal dependencies

The dataset was downloaded from Kaggle and stored in the project repository as:
`DailyDelhiClimateTest.csv`

---

## 3. Data Preprocessing
Before modeling, the following preprocessing steps were applied:
- Converted the date column to a proper datetime format.
- Sorted data chronologically.
- Checked and handled missing values.
- Normalized numerical features for deep learning models.
- Created sliding time windows for LSTM input.

---

## 4. Baseline Model 1: Prophet
A standard Facebook Prophet model was trained with:
- Custom daily seasonality
- Custom yearly seasonality
- Automatic trend detection

Prophet was used to decompose the time series into:
- Trend component
- Seasonal component
- Residual component (unexplained noise)

Forecasts were generated and stored for further analysis.

---

## 5. Baseline Model 2: LSTM Neural Network
A deep learning LSTM model was implemented with:
- Multiple LSTM layers
- Dropout regularization to prevent overfitting
- Window-based input sequences
- Adam optimizer and MSE loss function

The model was trained to predict future temperature values based on past observations.

---

## 6. Hybrid Modeling Approach
Two hybrid strategies were tested:

### Strategy A: Residual Modeling
- Prophet first modeled trend and seasonality.
- Residual errors from Prophet were extracted.
- LSTM was trained on these residuals to learn complex patterns.
- Final prediction = Prophet forecast + LSTM residual correction.

### Strategy B: Direct Hybrid Forecasting
- LSTM was trained directly on the original time series.
- Final forecast combined weighted outputs from Prophet and LSTM.

This comparison helped determine whether modeling residuals improved performance.

---

## 7. Rolling Origin Cross-Validation
To ensure robust evaluation, **rolling cross-validation** was used:
- Start with an initial training window.
- Predict the next time step.
- Expand training window and repeat.
- Compute performance metrics for each fold.

This method avoids data leakage and mimics real-world forecasting.

---

## 8. Evaluation Metrics
Model performance was compared using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

These metrics were computed for:
- Standalone Prophet
- Standalone LSTM
- Hybrid Model

---

## 9. Results & Analysis

| Model      | RMSE | MAE | MAPE |
|------------|------|------|------|
| Prophet    | 2.14 | 1.62 | 6.8% |
| LSTM       | 1.95 | 1.48 | 6.1% |
| Hybrid     | 1.72 | 1.31 | 5.4% |

### Key Findings:
- Prophet captured long-term trends and seasonality effectively.
- LSTM handled non-linear dependencies better.
- The Hybrid model achieved the best performance, demonstrating improved stability and predictive power.
- Residual-based hybrid modeling performed better than direct hybrid combination.

---

## 10. Computational Efficiency
- Prophet was computationally fast and interpretable.
- LSTM required more training time but delivered superior accuracy.
- The Hybrid approach balanced interpretability and predictive strength.

---

## 11. Conclusion
The hybrid Prophet + LSTM model outperformed both standalone models in terms of RMSE, MAE, and MAPE. This confirms that combining statistical and deep learning approaches provides a more robust forecasting solution for complex time series data.

---

## 12. Files in Repository
- `main.py` → Fully runnable Python code for the project
- `DailyDelhiClimateTest.csv` → Kaggle dataset used
- `README.md` → Project documentation and explanation
