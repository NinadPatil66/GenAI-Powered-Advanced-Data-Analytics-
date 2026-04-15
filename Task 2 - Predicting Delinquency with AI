# Task 2 - Predicting Delinquency with AI

## Goal:
- Use GenAI tools to generate model logic for predicting customer delinquency.
- Choose and justify the best predictive approach. Common techniques for credit risk modeling (e.g., decision trees, logistic regression, neural networks). 
- Evaluating model performance and ethical considerations (bias, explainability, fairness).

## Steps in Predictive Modeling:
1) Selecting the right model
Logistic Regression - If you need a probability-based approach that is easy to interpret. Great for binary predictions. Easy to interpret - Shows the impact of each variable on the outcome. Works well with structured data. 
Decision Trees - Transparency (Easy to explain to stakeholders), Clear risk segmentation, Works with both numerical and categorical data, Shows which customer attributes are most predictive of delinquency. 
Neural Networks - Uncover deep patterns in customer financial behavior, More accurate on large datasets than simpler models. It comes at the cost explainability.

2) Generating model code, refining and improving the predictions
   
3) Evaluating model performance
- Suggest evaluation metrics (e.g., accuracy, precision, recall, F1 score, confusion matrix).
- Interpret results and suggest improvements.
- Highlight ethical concerns, such as potential biases.

4) Bias, explainability, and fairness in credit risk modeling
- Bias occurs when a model systematically favors or disadvantages certain groups, often due to historical inequalities or imbalanced data. Types of bias includes historical bias, selection bias and proxy bias.
- Explainability ensures that decision-makers can understand and justify a model’s predictions.
- Fairness - Avoid systematic disadvantages for certain demographic groups, Be tested for disparate impact to ensure fairness, Use diverse and representative training data to prevent reinforcing biases.

Achieving truly responsible and unbiased financial decision-making also requires human oversight, regulatory compliance, and formal fairness audits 

## Optimizing Model Performance:
- Feature engineering – Adjust the dataset by adding or removing variables that may be impacting model predictions. For example, including customer tenure or past delinquency trends may enhance predictive power.
- Rebalancing the dataset – If the dataset is highly skewed (e.g., 95% non-delinquent, 5% delinquent), oversampling delinquent cases or undersampling non-delinquent cases can improve results.
- Trying different models – Some algorithms work better with certain data structures. If logistic regression is underperforming, a decision tree may provide better results.
- Hyperparameter tuning – Fine-tuning model parameters, such as adjusting the threshold for delinquency classification, can improve precision and recall scores.

## Outcome:
1) ML Algorithms Used: Logistic Regression, Decision Tress/Random Forest/XGBoost, Neural Network along with SHAP
2) Shortlisted: XGBoost with SHAP. It provides high performance and excels at capturing complex, non-linear patterns in noisy tabular data. It can prioritize high recall through specialized class weighting which aligns with the business need to minimize capital loss from defaults, 
while the integration of SHAP values provides the "explainability" required for regulatory transparency and fair lending audits.
3) Evaluation Metrics - Accuracy, Precision, Recall, F1 score, and Area Under the Precision, Recall Curve (AUCPRC). AUPRC ensures that the 16% class imbalance is effectively managed, ensuring high-risk accounts are captured without excessive false alarms.
4) Ethical Considerations: Avoid proxy bias, maintaining model transparency and fairness, and clearly communicating how model outputs influence decisions. 
