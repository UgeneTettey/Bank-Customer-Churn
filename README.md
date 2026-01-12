# Bank-Customer-Churn
Simple ML project using modular programming

A retail bank wants to predict whether a customer will churn in the next 3 months, based on their account activity and demographics.

## **Assumptions**
- Churn is binary: **_1 = churned, 0 = active_**  
- Each row represents one customers  


## Data Flow
Raw data(csv) -> Data Loading -> Validation (structure + sanity checks) -> cleaning & Transoformation, Train/evaluate model -> Persist Model + metrics -> Predict on new data.
