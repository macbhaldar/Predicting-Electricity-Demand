# Predicting Electricity Demand During a Day Using Machine Learning

---

## Abstract

Accurate electricity demand forecasting is essential for efficient power system operation, cost optimization, and grid stability. This project investigates multiple machine learning approaches to predict short-term electricity demand within a day using synthetic but realistic time-series data. Starting from exploratory data analysis, we progressively apply feature engineering and build predictive models ranging from classical regression techniques to advanced deep learning architectures. Models evaluated include Random Fourier Features with Linear Regression, Support Vector Regression, Gaussian Process Regression, Multi-Layer Perceptron Neural Networks, and Long Short-Term Memory networks. Performance is assessed using Mean Squared Error and R² score. Results demonstrate that sequence-based LSTM models outperform tabular models by effectively capturing temporal dependencies inherent in electricity demand data.

---

## 1. Introduction

Electricity demand exhibits strong temporal patterns influenced by time of day, weather conditions, and human behavior such as weekday versus weekend activity. Accurate short-term load forecasting enables:

- Efficient power generation scheduling  
- Reduced operational costs  
- Improved grid reliability  
- Integration of renewable energy sources  

Traditional statistical models struggle with complex non-linear patterns present in demand data. Machine learning models, particularly kernel methods and neural networks, offer powerful alternatives. This project aims to compare these methods in a structured and reproducible manner.

---

## 2. Dataset Description

### 2.1 Data Generation

A synthetic dataset was created to simulate real-world electricity demand patterns. The dataset covers **30 days** with **30-minute resolution**, resulting in **1440 observations**.

### 2.2 Raw Dataset Features

| Feature | Description |
|------|------------|
| day | Day index (0–29) |
| hour | Hour of day (0–24) |
| temp | Ambient temperature (°C) |
| measured_demand | Electricity demand (target variable) |

---

## 3. Exploratory Data Analysis (EDA)

EDA revealed:

- Clear **daily periodicity** in electricity demand  
- Positive correlation between temperature and demand  
- Lower demand on weekends compared to weekdays  
- Presence of noise mimicking real-world measurement uncertainty  

Visual inspection confirmed that demand patterns are smooth but non-linear, motivating the use of non-linear models.

---

## 4. Feature Engineering

To enhance model performance, the following features were engineered:

### 4.1 Cyclical Time Encoding

To preserve periodicity:

- `sin(2πt)`
- `cos(2πt)`

### 4.2 Additional Features

- Normalized time of day  
- Weekend indicator (binary)  
- Temperature as a weather feature  

### 4.3 Final Feature Set

| Feature |
|------|
| time_normalized |
| sin_time |
| cos_time |
| temp |
| is_weekend |
| measured_demand |

---

## 5. Baseline Model: Random Fourier Features + Linear Regression

Random Fourier Features (RFF) approximate the RBF kernel, enabling non-linear learning using a linear model.

### Key Observations

- Linear regression alone underfits the data  
- RFF significantly improves performance  
- Increasing RFF components reduces training error  
- Demonstrates the **universal approximation property**

This model serves as a strong and computationally efficient baseline.

---

## 6. Support Vector Regression (SVR)

SVR with an RBF kernel was applied to capture non-linear relationships.

### Results

- Strong predictive performance  
- Sensitive to hyperparameters (C, γ, ε)  
- Grid search improved generalization  

SVR performs competitively but scales poorly for large datasets.

---

## 7. Gaussian Process Regression (GPR)

GPR provides probabilistic predictions with uncertainty estimation.

### Advantages

- Predictive mean and confidence intervals  
- Interpretable kernel parameters  

### Limitations

- O(N³) computational complexity  
- Suitable only for small-to-medium datasets  

Despite computational cost, GPR provided highly accurate predictions and valuable uncertainty quantification.

---

## 8. Neural Networks (MLP)

A feed-forward Multi-Layer Perceptron was implemented.

### Architecture

- Two hidden layers with ReLU activation  
- Adam optimizer  
- Mean Squared Error loss  

### Observations

- Learns complex non-linear interactions  
- Requires careful scaling and regularization  
- Competitive performance with kernel-based methods  

---

## 9. LSTM: Time-Series Forecasting

Long Short-Term Memory networks explicitly model temporal dependencies.

### Key Features

- Uses sequences of past demand  
- Captures long-term temporal patterns  
- Suitable for real-world load forecasting  

### Results

- Lowest prediction error among all models  
- Smooth and stable forecasts  
- Superior performance compared to tabular models  

---

## 10. Model Comparison

| Model | Strengths | Limitations |
|----|---------|------------|
| RFF + Linear Regression | Fast, interpretable | Limited temporal modeling |
| SVR | Strong non-linearity | Expensive for large data |
| GPR | Uncertainty estimation | Poor scalability |
| MLP | Flexible, powerful | Needs tuning |
| LSTM | Best performance | Higher complexity |

---

## 11. Conclusion

This project demonstrated a complete end-to-end pipeline for electricity demand forecasting. While classical and kernel-based models perform well, deep learning models—particularly LSTM—provide superior accuracy by modeling temporal dependencies. The results highlight the importance of selecting models aligned with data structure.

---
