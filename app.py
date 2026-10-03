import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(
        "models/hybrid_xgb_lgb_smoteenn.pkl"
    )
    return model


@st.cache_resource
def load_features():
    features = joblib.load(
        "models/input_features.pkl"
    )
    return features


model = load_model()
input_features = load_features()


# ============================================================
# HEADER
# ============================================================

st.title("❤️ Heart Disease Risk Prediction")

st.write(
    """
    A machine learning application for predicting cardiovascular
    disease risk using a hybrid Gradient Boosting model combining
    XGBoost and LightGBM.
    """
)

st.info(
    """
    **Disclaimer:** This application is intended for research and
    educational purposes only. The prediction is not a medical
    diagnosis and should not replace professional medical advice.
    """
)


# ============================================================
# INPUT FORM
# ============================================================

st.subheader("Patient Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age_years = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            options=[1, 2],
            format_func=lambda x:
                "Female (1)" if x == 1 else "Male (2)"
        )

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=165.0,
            step=1.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=65.0,
            step=1.0
        )

        ap_hi = st.number_input(
            "Systolic Blood Pressure (ap_hi)",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )

        ap_lo = st.number_input(
            "Diastolic Blood Pressure (ap_lo)",
            min_value=30,
            max_value=200,
            value=80,
            step=1
        )

    with col2:

        cholesterol = st.selectbox(
            "Cholesterol",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal (1)",
                2: "Above Normal (2)",
                3: "Well Above Normal (3)"
            }[x]
        )

        gluc = st.selectbox(
            "Glucose",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal (1)",
                2: "Above Normal (2)",
                3: "Well Above Normal (3)"
            }[x]
        )

        smoke = st.selectbox(
            "Smoking",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        alco = st.selectbox(
            "Alcohol Consumption",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        active = st.selectbox(
            "Physical Activity",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

    submitted = st.form_submit_button(
        "🔍 Predict Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    bmi = weight / ((height / 100) ** 2)

    pressure_diff = ap_hi - ap_lo

    # Age in the original dataset is represented in days.
    age_days = age_years * 365.25

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            age_days,
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            bmi,
            pressure_diff
        ]],
        columns=input_features
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_no_risk = probabilities[0]
    probability_risk = probabilities[1]

    # --------------------------------------------------------
    # Display Result
    # --------------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ The model predicts a higher cardiovascular "
            "disease risk category."
        )

    else:

        st.success(
            "✅ The model predicts a lower cardiovascular "
            "disease risk category."
        )

    st.metric(
        "Predicted Risk Probability",
        f"{probability_risk * 100:.2f}%"
    )

    st.progress(float(probability_risk))

    # --------------------------------------------------------
    # Additional Information
    # --------------------------------------------------------

    st.subheader("Input Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "Height",
            "Weight",
            "Systolic BP",
            "Diastolic BP",
            "Cholesterol",
            "Glucose",
            "Smoking",
            "Alcohol",
            "Physical Activity",
            "BMI",
            "Pressure Difference"
        ],
        "Value": [
            f"{age_years:.0f} years",
            gender,
            f"{height:.1f} cm",
            f"{weight:.1f} kg",
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            f"{bmi:.2f}",
            f"{pressure_diff:.0f}"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The prediction is generated by the trained hybrid "
        "XGBoost + LightGBM model."
    )
