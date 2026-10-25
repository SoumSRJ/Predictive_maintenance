import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_predictive_failure_model.joblib")
model = joblib.load(model_path)

st.title("Engine Failure Prediction App")
st.write("""
This application predicts the likelihood of an engine failing based on its operational parameters.
Enter the sensor and configuration data below to get a prediction.
""")

Engine_RPM     = st.number_input("Engine speed (RPM)", 250.0, 400.0, 298.0, 0.1)
Lub_Oil_Pressure = st.number_input("Lubricant Oil Pressure(Kpa)", 250.0, 500.0, 324.0, 0.1)
Fuel_Pressure = st.number_input("Fuel Pressure to engine(Kpa)", 250.0, 500.0, 324.0, 0.1)
Coolant_Pressure    = st.number_input("Coolant oressure (kpa)", 250.0, 500.0, 324.0, 0.1)
Lub_Oil_Temperature = st.number_input("Lubricant Oil Temperature (C)", 250.0, 500.0, 324.0, 0.1)
Coolant_Temperature = st.number_input("Coolant Temperature (C)", 250.0, 500.0, 324.0, 0.1)

input_data = pd.DataFrame([{
    "Engine speed (RPM)": Engine_RPM,
    "Lubricant Oil Pressure(Kpa)": Lub_Oil_Pressure,
    "Fuel Pressure to engine(Kpa)": Fuel_Pressure,
    "Coolant oressure (kpa)": Coolant_Pressure,
    "Lubricant Oil Temperature (C)": Lub_Oil_Temperature,
    "Coolant Temperature (C)": Coolant_Temperature,
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Engine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
