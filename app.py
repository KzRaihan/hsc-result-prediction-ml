# ================================
# HSC Result Predictor (Gradio App)
# ================================

import gradio as gr
import pandas as pd
import numpy as np
import pickle


# -------------------------------
# 1. Load Trained Model Pipeline
# -------------------------------
MODEL_PATH = "student_rf_pipeline.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")


# -------------------------------
# 2. Prediction Function
# -------------------------------
def predict_gpa(
    gender, age, address, famsize,
    Pstatus, M_Edu, F_Edu, M_Job, F_Job,
    relationship, smoker, tuition_fee,
    time_friends, ssc_result
):
    # -------- Input Validation --------
    if age is None or age <= 15 or age > 30:
        return "❌ Age must be between 15 and 30."

    if ssc_result is None or ssc_result < 2.50 or ssc_result > 5.00:
        return "❌ SSC Result must be between 2.50 and 5.00."

    if tuition_fee is None or tuition_fee < 0 or tuition_fee > 134168:
        return "❌ Please enter a valid tuition fee (0–134168 BDT)."

    # -------- Prepare Input Data --------
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

    # -------- Model Prediction --------
    try:
        prediction = model.predict(input_df)[0]
        prediction = np.clip(prediction, 0, 5)
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

    return f"🎓 Predicted HSC GPA: {prediction:.2f}"


# -------------------------------
# 3. Gradio Interface
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
    gr.Number(label="Monthly Tuition Fee (BDT)", value=0),
    gr.Slider(1, 5, step=1, label="Time Spent with Friends"),
    gr.Number(label="SSC Result (GPA)", value=4.0)
]

app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs="text",
    title="📊 HSC Result Predictor (Bangladesh)",
    description=(
        "Predict HSC GPA using a Random Forest model trained on "
        "Bangladesh student academic data."
    )
)


# -------------------------------
# 4. Launch App
# -------------------------------
if __name__ == "__main__":
    app.launch()
