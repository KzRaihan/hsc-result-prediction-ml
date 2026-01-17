# ================================
# HSC Result Predictor (Gradio App)
# ================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Trained Pipeline
# -------------------------------
model = joblib.load("student_rf_pipeline.pkl")

# -------------------------------
# Prediction Function
# -------------------------------
def predict_gpa(
    gender, age, address, famsize, Pstatus,
    M_Edu, F_Edu, M_Job, F_Job,
    relationship, smoker, tuition_fee,
    time_friends, ssc_result
):
    # Basic input validation
    if age <= 0 or tuition_fee < 0 or not (0 <= ssc_result <= 5):
        return "❌ Please enter valid input values."

    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "M_Edu": M_Edu,
        "F_Edu": F_Edu,
        "M_Job": M_Job,
        "F_Job": F_Job,
        "relationship": relationship,
        "smoker": smoker,
        "tuition_fee": tuition_fee,
        "time_friends": time_friends,
        "ssc_result": ssc_result
    }])

    prediction = model.predict(input_df)[0]
    prediction = np.clip(prediction, 0, 5)

    return f"🎓 Predicted HSC GPA: {prediction:.2f}\n\n⚠️ This is a prediction based on historical data."

# -------------------------------
# Gradio Interface
# -------------------------------
inputs = [
    gr.Radio(["M", "F"], label="Gender"),
    gr.Number(label="Age", value=18),
    gr.Radio(["Urban", "Rural"], label="Address"),
    gr.Radio(["GT3", "LE3"], label="Family Size"),
    gr.Radio(["Together", "Apart"], label="Parents' Living Status"),
    gr.Slider(0, 4, step=1, label="Mother's Education Level"),
    gr.Slider(0, 4, step=1, label="Father's Education Level"),
    gr.Dropdown(
        ["At_home", "Health", "Services", "Teacher", "Other"],
        label="Mother's Occupation"
    ),
    gr.Dropdown(
        ["Teacher", "Other", "Services", "Health", "Business", "Farmer"],
        label="Father's Occupation"
    ),
    gr.Radio(["Yes", "No"], label="In a Relationship"),
    gr.Radio(["Yes", "No"], label="Smoker"),
    gr.Number(label="Monthly Tuition Fee (BDT)"),
    gr.Slider(1, 5, step=1, label="Time Spent with Friends"),
    gr.Number(label="SSC Result (GPA)")
]

app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs="text",
    title="📊 HSC Result Predictor (Bangladesh)",
    description="Predict HSC GPA using a Random Forest model trained on historical student data."
)

# -------------------------------
# Launch App
# -------------------------------
if __name__ == "__main__":
    app.launch()
