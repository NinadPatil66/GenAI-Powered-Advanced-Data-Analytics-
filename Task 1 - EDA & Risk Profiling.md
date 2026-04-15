# Task 1 - EDA & Risk Profiling

## Goal:
- Conduct exploratory data analysis (EDA) using GenAI.
- Techniques to handle missing values and ensure data quality.
- Understanding customer risk factors for delinquency.
- Leverage synthetic data generation to enhance datasets when real data is insufficient.

## Steps in EDA:
EDA consists of four main steps, which can be enhanced with GenAI tools:
1) Understanding the Dataset
2) Identifying missing values and outliers
3) Understanding relationships between variables
4) Detecting patterns and risk factor

## Outcome:
- Key variables: Age, Income, Credit Score, Credit Utilization, Missed Payments, Debt-to-Income Ratio.
- Target variable is imbalanced.
- Imputation using median values for numerical data.
- Strong correlation between high credit utilization (>50%) and delinquency.
- Customers with 3+ missed payments in the past 6 months have a higher delinquency rate.
- Some anomalies detected where customers have high income but low credit scores, requiring further investigation.
