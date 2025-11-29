import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load data and model
# -----------------------------
data = pd.read_csv('Placement.csv')
model = joblib.load('model_campus_placement')

# -----------------------------
# Sidebar - User Inputs
# -----------------------------
st.sidebar.header("Student Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("M", "F"))
    ssc_p = st.sidebar.slider("10th Percentage (SSC)", 0, 100, 70)
    ssc_b = st.sidebar.selectbox("Board (SSC)", ("Central", "Others"))
    hsc_p = st.sidebar.slider("12th Percentage (HSC)", 0, 100, 80)
    hsc_b = st.sidebar.selectbox("Board (HSC)", ("Central", "Others"))
    hsc_s = st.sidebar.selectbox("Stream (HSC)", ("Science", "Commerce", "Arts"))
    degree_p = st.sidebar.slider("Degree Percentage", 0, 100, 65)
    degree_t = st.sidebar.selectbox("Degree Type", ("Sci&Tech", "Comm&Mgmt", "Others"))
    workex = st.sidebar.selectbox("Work Experience", ("Yes", "No"))
    etest_p = st.sidebar.slider("Etest Percentage", 0, 100, 60)
    specialisation = st.sidebar.selectbox("MBA Specialisation", ("Mkt&HR", "Mkt&Fin"))
    mba_p = st.sidebar.slider("MBA Percentage", 0, 100, 60)

    features = {
        'gender': 1 if gender == 'M' else 0,
        'ssc_p': ssc_p,
        'ssc_b': 1 if ssc_b == 'Central' else 0,
        'hsc_p': hsc_p,
        'hsc_b': 1 if hsc_b == 'Central' else 0,
        'hsc_s': 2 if hsc_s == 'Science' else 1 if hsc_s == 'Commerce' else 0,
        'degree_p': degree_p,
        'degree_t': 2 if degree_t == 'Sci&Tech' else 1 if degree_t == 'Comm&Mgmt' else 0,
        'workex': 1 if workex == 'Yes' else 0,
        'etest_p': etest_p,
        'specialisation': 1 if specialisation == 'Mkt&HR' else 0,
        'mba_p': mba_p
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# -----------------------------
# Main Panel
# -----------------------------
st.title("Campus Placement Prediction")

# Show top 5 Sci&Tech placed students
st.subheader("Top 5 Placed Sci&Tech Students by Salary")
top_students = data[(data['degree_t'] == "Sci&Tech") & (data['status'] == "Placed")].sort_values(by="salary", ascending=False).head()
st.dataframe(top_students)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Predict Placement Status")
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)[:,1] if hasattr(model, "predict_proba") else None

status = "Placed" if prediction[0] == 1 else "Not Placed"
st.write(f"**Prediction:** {status}")

if prediction_proba is not None:
    st.write(f"**Probability of Placement:** {prediction_proba[0]*100:.2f}%")

# -----------------------------
# Model Accuracies
# -----------------------------
st.subheader("Model Accuracies")
# You can hardcode or calculate if you want to dynamically include all
model_acc = pd.DataFrame({
    'Models':['LR','SVC','KNN','DT','RF','GB'],
    'Accuracy (%)':[90, 85, 82, 80, 92, 94]  # replace with your scores
})
st.table(model_acc)
