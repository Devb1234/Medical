import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Medical No-Show Predcitor",page_icon="📅",layout="wide")

with open("./models/no_show_model.pkl","rb") as f:
    model=pickle.load(f)

with open("./models/scaler.pkl","rb") as f:
    scaler=pickle.load(f)

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📅 Medical Appointment No-Show Predictor</h1>
    <p style='text-align: center;'>Predict whether a patient will attend their medical appointment.</p>
""",unsafe_allow_html=True)

st.sidebar.header("📝 Patient Details")

gender=st.sidebar.selectbox("Gender", ['Female', 'Male'])
age=st.sidebar.slider("Age", 0, 115, 35)
scholarship=st.sidebar.selectbox("Scholarship Received?", [0, 1])
hypertension=st.sidebar.selectbox("Hypertension?", [0, 1])
diabetes=st.sidebar.selectbox("Diabetes?", [0, 1])
alcoholism=st.sidebar.selectbox("Alcoholism?", [0, 1])
handicap=st.sidebar.selectbox("Handicap?", [0, 1])
sms_received=st.sidebar.selectbox("SMS Received?", [0, 1])
waiting_days=st.sidebar.slider("Waiting Days", 0, 100, 5)
appointment_weekday=st.sidebar.selectbox("Appointment Day", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
scheduled_weekday=st.sidebar.selectbox("Scheduled Day", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

gender=1 if gender=='Male' else 0
weekday_map={'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
appointment_weekday=weekday_map[appointment_weekday]
scheduled_weekday=weekday_map[scheduled_weekday]

input_data=pd.DataFrame([[
    gender, age, scholarship, hypertension, diabetes, alcoholism,
    handicap, sms_received, waiting_days, appointment_weekday, scheduled_weekday
]], columns=[
    'gender', 'age', 'scholarship', 'hipertension', 'diabetes',
    'alcoholism', 'handcap', 'sms_received', 'waiting_days',
    'appointment_weekday', 'scheduled_weekday'
])

input_data[['age', 'waiting_days']]=scaler.transform(input_data[['age', 'waiting_days']])

col1,col2,col3=st.columns([1,2,1])
with col2:
    if st.button("🔍 Predict"):
        prediction=model.predict(input_data)[0]
        probability=model.predict_proba(input_data)[0][1]

        st.markdown("---")
        if prediction==1:
            st.error(f"❌ Patient is likely to **miss** the appointment.\n\n**Probability:** {probability:.2f}")
            st.markdown("😟 Consider follow-up actions like reminders or incentives.")
        else:
            st.success(f"✅ Patient is likely to **attend** the appointment.\n\n**Probability:** {1 - probability:.2f}")
            st.markdown("👍 Great! Things look good for attendance.")

st.markdown("""
    <hr style="border:1px solid #ddd;">
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Made with ❤️ by Dev · Powered by Streamlit
    </div>
""", unsafe_allow_html=True)