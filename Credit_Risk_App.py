import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Credit Risk Evaluation", layout="wide")

# Load data
@st.cache_data
def load_data():
    file_path = "C:/Users/Admin/german_credit_data.csv"
    df = pd.read_csv(file_path)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.fillna({'Saving accounts': 'unknown', 'Checking account': 'unknown'}, inplace=True)
    return df

df = load_data()
st.title("Credit Risk Evaluation with German Credit Dataset")

# Preprocess
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

numerical_cols = ['Age', 'Credit amount', 'Duration']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

credit_threshold = df['Credit amount'].median()
duration_threshold = df['Duration'].median()
df['Risk'] = ((df['Credit amount'] > credit_threshold) & (df['Duration'] < duration_threshold)).astype(int)

X = df.drop(columns=['Risk'])
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Model training
st.subheader("Model Training with XGBoost")
param_grid = {
    'n_estimators': [100],
    'max_depth': [5],
    'learning_rate': [0.1],
}
xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(xgb_base, param_grid, scoring='f1', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Classification Report
st.markdown("### Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.markdown("### Confusion Matrix")
fig1, ax1 = plt.subplots(figsize=(4, 3))  # Reduced size
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

# Correlation Heatmap
st.markdown("### Correlation with Risk")
corr_matrix = df.corr()
fig2, ax2 = plt.subplots(figsize=(6, 5))  # Reduced size
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Most influential factors
st.markdown("### Top Influential Factors")
correlations = corr_matrix['Risk'].drop('Risk')
most_important = correlations.abs().sort_values(ascending=False)
st.write(most_important.head(5))

# Feature Importance Plot
st.markdown("### Feature Importance (XGBoost)")
fig3, ax3 = plt.subplots(figsize=(6, 4))  # Reduced size
xgb.plot_importance(best_model, max_num_features=10, importance_type='gain', ax=ax3)
st.pyplot(fig3)

# Actionable Insights
st.markdown("### Actionable Insights & Recommendations")
if 'Credit amount' in most_important.index and most_important['Credit amount'] > 0.3:
    st.write("- High credit amount increases risk. *Consider stricter limits or require collateral.*")

if 'Duration' in most_important.index and most_important['Duration'] > 0.2:
    st.write("- Shorter loan duration is risky. *Encourage longer, manageable repayment periods.*")

if 'Checking account' in most_important.index:
    st.write("- Checking account status is predictive. *Verify banking and financial background thoroughly.*")

if report['1']['recall'] < 0.7:
    st.write("- Risk class recall is low. *Use SMOTE or class weights to reduce false negatives.*")

st.success("Model ready! Use this tool to evaluate and improve your credit approval process.")

# User Input for Prediction
st.markdown("## Predict Credit Risk for New Applicant")

with st.form("credit_risk_form"):
    user_credit_amount = st.number_input("Enter Credit Amount", min_value=0.0, format="%.2f")
    user_duration = st.number_input("Enter Duration (in months)", min_value=0.0, format="%.2f")
    submit_btn = st.form_submit_button("Predict Credit Risk")

if submit_btn:
    # Scale input
    dummy_row = X.iloc[0:1].copy()
    dummy_row[numerical_cols] = 0

    # Scale user input
    scaled = scaler.transform([[0, user_credit_amount, user_duration]])
    dummy_row[['Age', 'Credit amount', 'Duration']] = scaled[0]

    prediction = best_model.predict(dummy_row)[0]

    if prediction == 1:
        st.success("✅ Prediction: Good Credit Risk")
    else:
        st.error("❌ Prediction: Bad Credit Risk")

    # SHAP Explanation
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(dummy_row)

    st.markdown("### Prediction Explanation (SHAP)")
    fig4, ax4 = plt.subplots(figsize=(6, 4))  # Reduced size
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig4)

    st.markdown("### 🔍 Top Contributing Features")
    feature_contributions = pd.DataFrame({
        'Feature': dummy_row.columns,
        'SHAP Value': shap_values.values[0]
    }).sort_values(by='SHAP Value', key=np.abs, ascending=False)

    st.dataframe(feature_contributions.head(5))
