# Project Title - GenAI-Powered-Advanced-Data-Analytics:

## Goal:
- Reduce credit card delinquency rate by performing advanced data analytics & building AI/ML models using GenAI
- Develop a recommendation framework to help the Head of Collections at Geldium finance determine the best intervention strategies for at-risk customers.
- Ensure that AI-driven solutions are ethical, explainable, and effective in supporting responsible financial decision-making.

## Key risk factors for delinquency
- Payment history – Customers with a history of late or missed payments are more likely to default.
- Credit utilization rate – High usage of available credit can indicate financial stress and potential repayment issues.
- Debt-to-income (DTI) ratio – A high DTI suggests a customer may struggle to manage their financial obligations.
- Recent credit activity – A sudden increase in new credit accounts or loan applications may signal financial instability.
- Employment and income stability – Frequent job changes or inconsistent income can contribute to a higher risk of missed payments.
- Demographic trends – While AI models must avoid bias, certain patterns (e.g., younger customers with limited credit history) may require additional analysis.

## Synthetic Data:
In financial services, incomplete or inconsistent data can make it difficult to build reliable predictive models. When real-world data is limited, sensitive, or incomplete, synthetic data generation can help fill gaps, simulate scenarios, and improve dataset quality while maintaining privacy and compliance. Synthetic data is created using statistical models or AI-driven techniques to supplement missing values or expand datasets for testing. Traditional statistical simulation techniques such as Monte Carlo simulations, bootstrapping, and probabilistic modeling are often preferred due to their explainability, reproducibility, and ability to align with industry regulations. It must be strictly validated to ensure it accurately reflects real-world trends and does not introduce bias.

GenAI-generated synthetic data should be:
- Validated against real-world distributions to ensure accuracy.
- Cross-checked with statistical models to prevent introducing artificial patterns or biases.
- Used as a supplementary tool, not a primary data source, especially in regulatory environments.

Example GenAI prompt: “Generate synthetic payment history data for customers with missing records while ensuring that distributions align with historical patterns observed in the dataset (e.g., standard deviations, typical payment behaviors).”

While synthetic data can enhance dataset completeness, it must be used carefully to avoid:
- Introducing bias – Ensure synthetic records reflect realistic patterns.
- Misrepresenting risk factors – Avoid generating overly optimistic or pessimistic data.
- Compromising compliance – Validate that synthetic data aligns with industry regulations.

# Task 1 - EDA & Risk Profiling
- Conduct exploratory data analysis (EDA) using GenAI.
- Techniques to handle missing values and ensure data quality.
- Understanding customer risk factors for delinquency.
- Leverage synthetic data generation to enhance datasets when real data is insufficient.

## Steps in EDA:
EDA consists of four main steps, which can be enhanced with GenAI tools:

Step 1 - Understanding the Dataset
Ask yourself few questions:
1) What are the key variables (e.g., payment history, income levels, credit utilization)?
2) Are there categorical or numerical data points?
3) Are there missing or inconsistent values?

Step 2 - Identifying missing values and outliers
When handling missing data, it is critical to first analyze the reason for missingness—whether it is random, systematic, or indicative of underlying biases.

There three different missing data mechanisms:
•	Missing Completely at Random (MCAR): Data is considered MCAR when the reason for the missing values is unrelated to any other data in the dataset. The missingness happens by pure chance, without any pattern.
•	Missing at Random (MAR): Data is considered MAR when the reason values are missing is related to other information in the dataset that isn't missing. If we know the values of some complete variables, we can explain why other values are missing.
•	Missing Not at Random (MNAR): Data is considered MNAR when the reason it's missing is related to the missing value itself. In other words, the missingness depends on information we don’t have.

Step 3 - Understanding relationships between variables

Step 4 - Detecting patterns and risk factors

Example Prompt: 
Begin - 
I am uploading a CSV. I want you to act as a Senior Data Scientist. Do not just summarize the data; perform a deep Exploratory Data Analysis. 

1) Data Quality and Anomaly Check - "Summarize key patterns, outliers, missing values or any other inconsistencies in this dataset. Highlight any fields that might present problems for analysis or modeling, such as high null rates, suspicious distributions, or constant/near-constant columns."
2) Univariate Analysis - "For each variable in this dataset, describe its distribution. For numerical columns, report the range, mean, median, skewness, and presence of outliers. For categorical columns, report the number of unique values, the most/least frequent categories, and any signs of high cardinality. Flag any variables that may need transformation before modeling."
3) Bivariate & Correlation Analysis - "Analyze the relationships between variables in this dataset. Identify strongly correlated numerical pairs, and highlight any multicollinearity concerns. If a target variable exists, show how each feature relates to it individually. For categorical variables, note any strong associations using frequency breakdowns or statistical indicators."
4) Visualization & Feature Engineering Suggestions - "Based on the structure and distributions in this dataset, suggest: (a) the most effective chart types to visualize key patterns and relationships (e.g., histograms, heatmaps, box plots, scatter plots), and (b) potential feature engineering steps such as binning, encoding, log transforms, interaction terms, or datetime decomposition that could improve model performance. Keep suggestions generic and applicable regardless of domain."
5) Target-Driven Feature Relevance - "If a target variable is present or specified, identify the top 3–5 variables most likely to predict it based on this dataset. If no target is specified, identify the most statistically interesting or information-rich variables. Provide brief reasoning for each."

# Task 2 - Predicting Delinquency with AI
- Use GenAI tools to generate model logic for predicting customer delinquency.
- Choose and justify the best predictive approach. Common techniques for credit risk modeling (e.g., decision trees, logistic regression, neural networks). 
- Evaluating model performance and ethical considerations (bias, explainability, fairness).
- How to utilize GenAI tools like ChatGPT or Google Gemini to generate model code and refine predictions.

Predictive Modeling - Predictive modeling is the process of using historical data to forecast future outcomes. GenAI tools can assist analysts in building, testing, and refining models with less manual coding.

GenAI can help to:
- Suggest appropriate modeling techniques based on dataset characteristics.
- Draft a description of model logic, or even sample code.
- Assist in interpreting results and refining model performance.

## Steps in predictive modeling:
1) Selecting the right model
Logistic Regression - Great for binary predictions, Easy to interpret - Shows the impact of each variable on the outcome, Works well with structured data.
Decision Trees - Transparency (Easy to explain to stakeholders), Works with both numerical and categorical data, Shows which customer attributes are most predictive of delinquency.
Neural Networks - Uncover deep patterns in customer financial behavior, More accurate on large datasets than simpler models

2) Generating model code
   
3) Evaluating model performance
GenAI can help:
- Suggest evaluation metrics (e.g., accuracy, precision, recall).
- Interpret results and suggest improvements.
- Highlight ethical concerns, such as potential biases.

If your model is not performing well, there are several ways to improve it:

Feature engineering – Adjust the dataset by adding or removing variables that may be impacting model predictions. For example, including customer tenure or past delinquency trends may enhance predictive power.
Rebalancing the dataset – If the dataset is highly skewed (e.g., 95% non-delinquent, 5% delinquent), oversampling delinquent cases or undersampling non-delinquent cases can improve results.
Trying different models – Some algorithms work better with certain data structures. If logistic regression is underperforming, a decision tree may provide better results.
Hyperparameter tuning – Fine-tuning model parameters, such as adjusting the threshold for delinquency classification, can improve precision and recall scores.

Prompt Example:
1) "Based on this dataset, which predictive modeling techniques are best suited for identifying customers likely to miss payments? Explain why."
2) "Generate a decision tree model to predict delinquency risk based on income, credit utilization, and missed payments. Explain how the model determines risk categories."
3) "Generate a logistic regression model framework using this dataset to predict customer delinquency. Provide an explanation of each step, ensuring outputs are reviewed and refined for accuracy and fairness."
4) "Evaluate the performance of this predictive model using precision and recall. Identify any biases in the predictions."
5) "Create a basic neural network model for predicting delinquency risk. Compare its strengths and weaknesses against decision trees and logistic regression."
6) "Explain how logistic regression can be used to predict credit card delinquency. Generate a simple model using income, debt-to-income ratio, and payment history."
7) "Evaluate the performance of my credit risk model using precision, recall, and F1 score. Suggest improvements if needed."

# Task 3 - Business report and data storytelling for collections strategy

# Task 4 - Implementing an AI-driven collections strategy
How to design an AI-powered, autonomous debt-management system.
The role of agentic AI in automating financial decision-making.
Strategies for ensuring compliance, transparency, and fairness in AI-driven financial services.
How to align AI-driven insights with financial industry regulatory standards.
