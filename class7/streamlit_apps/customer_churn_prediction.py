import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_squared_error, classification_report,confusion_matrix
import streamlit as st

# Load dataset
df3 = pd.read_csv("../datasets/customer_churn_dataset.csv")
df3.dropna(inplace=True)

# Convert 'churn' column to integers
df3['churn'] = df3['churn'].astype(str).str.strip().map({'False': 0, 'True':1})

# Convert categorical columns to numerical
df3 = pd.get_dummies(df3, columns=['international_plan', 'voice_mail_plan'],drop_first=True)

# Drop unnecessary columns
df3.drop(columns=['Id', 'state', 'phone_number'], inplace=True)

# Define features and target variable
X = df3.drop(columns=['churn'])
y = df3['churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)


coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_.flatten()  # or model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)


st.title('Customer Churn Prediction')

st.write(f'Model Mean Squared Error (Continuous Prediction): {mse:.2f}')

st.subheader('Classification Report')

# Parse the report more carefully
report_lines = report.split('\n')
data = []

# Process only the class-specific lines (0 and 1 for binary classification)
for line in report_lines[2:]:  # Skip the header and separator lines
    parts = line.split()
    if len(parts) >= 5 and parts[0].isdigit():  
        class_name = parts[0]
        precision = float(parts[1])
        recall = float(parts[2])
        f1 = float(parts[3])
        support = int(parts[4])
        data.append([class_name, precision, recall, f1, support])

# Create DataFrame
if data:
    report_df = pd.DataFrame(data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    # Format percentage columns
    for col in ['Precision', 'Recall', 'F1-Score']:
        report_df[col] = report_df[col].map('{:.1%}'.format)
    
    # Map class names to more readable labels
    report_df['Class'] = report_df['Class'].map({'0': 'No Churn', '1': 'Churn'})
    
    # Add styling to the dataframe
    st.dataframe(
        report_df.style.highlight_max(axis=0, subset=['Precision', 'Recall', 'F1-Score'], color='blue'),
        use_container_width=True
    )
    
    # Add overall accuracy from the report
    accuracy_line = [line for line in report_lines if 'accuracy' in line]
    if accuracy_line:
        accuracy_parts = accuracy_line[0].split()
        if len(accuracy_parts) >= 2:
            accuracy = float(accuracy_parts[1])
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
    
    # Add a short explanation
    st.caption("Higher values are better for all metrics. Support shows the number of samples in each class.")
else:
    
    st.code(report, language="text")

st.write('Model Coefficients:')
st.dataframe(coefficients)

# Confusion Matrix Visualization
st.subheader('Confusion Matrix Visualization')
fig, ax = plt.subplots()
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)


# Create sidebar inputs dynamically for each feature
input_features = {}
for col in X.columns:
    input_features[col] = st.sidebar.number_input(
        f"Enter {col}",
        float(X[col].min()),
        float(X[col].max())
    )

if st.sidebar.button('Predict Churn Risk'):
    input_data = np.array([input_features[col] for col in X.columns]).reshape(1, -1)
    pred_continuous = model.predict(input_data)[0]
    # Threshold to decide churn risk
    pred_class = 1 if pred_continuous >= 0.5 else 0
    st.sidebar.write(f"Predicted churn risk (continuous value): {pred_continuous:.2f}")
    st.sidebar.write(f"Predicted Churn: {'Yes' if pred_class == 1 else 'No'}")
    if pred_class == 1:
        st.sidebar.error("Alert: This customer is at high risk of churning!")
    else:
        st.sidebar.success("Low risk of churn.")