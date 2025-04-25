# TCS_BFSI_Hackathon

**Problem Statement: Predicting Credit Risk for Loan Applicants**

**Background:**
Financial institutions face significant challenges in assessing the creditworthiness of loan applicants. Accurate credit risk prediction is crucial for minimizing defaults and ensuring the stability of the lending system. The German Credit dataset provides a comprehensive set of features related to applicants' financial history, personal information, and loan details, making it an ideal resource for developing predictive models.

**Objective:**
Develop a machine learning model to predict the credit risk of loan applicants using the German Credit dataset. The model should classify applicants into two categories: good credit risk and bad credit risk. Additionally, provide insights into the key factors influencing credit risk and suggest strategies for improving the credit evaluation process.

**Requirements:**

**1. Data Exploration and Preprocessing:**

Analyze the dataset to understand the distribution of features and target variable.
Handle missing values, outliers, and perform necessary data cleaning.
Engineer new features that could enhance model performance.

**2. Model Development:**

Select appropriate machine learning algorithms for classification.
Train and validate the model using suitable evaluation metrics (e.g., accuracy, precision, recall, F1-score).
Optimize the model through techniques such as hyperparameter tuning and cross-validation.

**3. Model Interpretation and Insights:**

Interpret the model's predictions and identify the most influential features.
Create visualizations to communicate findings effectively.
Provide actionable insights and recommendations for improving the credit evaluation process.

**4. Presentation:**

Prepare a comprehensive report detailing the methodology, results, and conclusions. Explain why the implemented approach was selected.

You may use streamlit for UI. 

Submit the recording of the demo with voice over of what has been achieved along with the code.

**Methodology:**

**Credit Risk Prediction for Loan Applicants**

**Objective:**

The goal of this project is to develop a machine learning model capable of classifying loan applicants as either Good Credit Risk or Bad Credit Risk. This enables financial institutions to make informed decisions, minimize defaults, and strengthen the credit evaluation process.

**1. Data Exploration and Preprocessing**

**1.1 Dataset Loading**

The German Credit dataset is loaded from a .csv file.
Irrelevant columns like Unnamed: 0 are dropped.
Missing values in Saving accounts and Checking account columns are filled with 'unknown'.

**1.2 Data Cleaning and Encoding**

All categorical features are label encoded using LabelEncoder, which converts text labels into numerical values suitable for machine learning algorithms.
Numerical features (Age, Credit amount, and Duration) are standardized using StandardScaler to bring them onto a comparable scale.

**1.3 Target Variable Creation**

Instead of directly using a provided label, the model creates a custom binary target Risk:
Risk = 1 (Good credit) if Credit amount > median and Duration < median.
Risk = 0 (Bad credit) otherwise.
This logic helps simulate a real-world scenario where risk thresholds are determined based on data distribution.

**1.4 Train-Test Split**

The data is split into training and testing sets using an 80:20 ratio, maintaining stratification to balance both classes.

**2. Model Development**

**2.1 Model Selection: XGBoost Classifier**

XGBoost is chosen for its robustness, regularization capabilities, and high performance on tabular data.
The model is trained using a GridSearchCV with 3-fold cross-validation to optimize parameters:
n_estimators = 100
max_depth = 5
learning_rate = 0.1

**2.2 Model Evaluation**

Evaluation metrics used:
Accuracy
Precision
Recall
F1-Score
Classification report and Confusion Matrix are generated to measure predictive performance on unseen data.

**3. Model Interpretation and Insights**

**3.1 Feature Importance Analysis**

A Correlation Heatmap is used to visualize feature relationships with the target Risk.
The top 5 features most correlated with risk are listed, giving stakeholders an idea of influential factors.
XGBoost feature importance (based on gain) is also plotted to validate model-driven insights.

**3.2 Actionable Recommendations**

Based on feature importance, specific recommendations are provided:
Reduce high credit amounts.
Offer longer repayment durations.
Evaluate banking history through checking account status.
Improve model recall using class balancing techniques like SMOTE or class weighting.

**4. Predicting New Applicant Credit Risk (User Interface)**

**4.1 Streamlit Interface**

A form-based UI using Streamlit allows users to input:
Credit amount
Duration

**4.2 Prediction Output**

The inputs are scaled using the same StandardScaler used in training.
A dummy row is used to match input format with the trained model.
The model returns:
Good Credit Risk or Bad Credit Risk based on prediction.

**5. Explainable AI with SHAP**

**5.1 SHAP (SHapley Additive exPlanations)**

SHAP is used to explain why a specific prediction was made.
A waterfall plot visualizes how each feature pushed the prediction toward good or bad risk.
The top contributing features for that individual prediction are shown in a table for further interpretation.

**Results and Inference:**

![Screenshot 2025-04-25 224020](https://github.com/user-attachments/assets/00426361-e3b9-4d41-a9ee-c017222ae8f2)

The image of the classification implies that:

True Negatives (TN): 180

False Positives (FP): 0

False Negatives (FN): 1

True Positives (TP): 19

This implies the model correctly identified:

180 out of 180 negative class samples (label 0),

19 out of 20 positive class samples (label 1),

Only 1 false negative and no false positives.

The following confusion matrix is evident for that:

![Screenshot 2025-04-25 224040](https://github.com/user-attachments/assets/a5940b29-264e-4cc4-a183-40e9fca7d555)

**Correlation Heatmap Analysis**

![Screenshot 2025-04-25 224121](https://github.com/user-attachments/assets/de53d791-5c9d-46a9-a68c-75e4624bbfef)

A correlation heatmap was plotted to analyze the linear relationship between input features and the target variable ‘Risk’, which indicates the likelihood of credit default or non-default.

**Observations**

The heatmap displays Pearson correlation coefficients ranging from -1 to 1. A coefficient near 1 indicates a strong positive correlation, near -1 indicates a strong negative correlation, and around 0 indicates no correlation. The diagonal values are all 1.00, as each feature is perfectly correlated with itself.

**Target Variable Correlation: ‘Risk’**

The feature ‘Duration’ shows the highest negative correlation with the target variable ‘Risk’, having a correlation coefficient of -0.28. This implies that as the loan duration increases, the risk decreases, or vice versa, under the current dataset. Though the correlation is moderate, it suggests Duration is a notable predictor for classification.

**Feature-to-Feature Correlations**

‘Credit amount’ and ‘Duration’ are positively correlated (0.64):
Longer loan durations tend to be associated with higher credit amounts.
‘Job’ and ‘Credit amount’ show a weak positive correlation (0.29), indicating a mild trend where certain job types request larger credit amounts.
Most other features show very low to negligible correlation with one another, which is beneficial as it reduces multicollinearity and helps in more robust model training.

**Insights for Feature Selection**

Features like ‘Age’, ‘Sex’, ‘Saving accounts’, ‘Checking account’, and ‘Purpose’ show minimal correlation with ‘Risk’ (near 0), indicating that these features alone may not be strong predictors of risk.
However, they may still hold non-linear or interaction effects, which ensemble models like Gradient Boosting can capture.
The low inter-feature correlations indicate that redundancy among features is minimal, supporting the use of all features in the initial training phase.
