"""
Streamlit demo — Crop Recommendation Classifier
------------------------------------------------
Run with:  streamlit run app.py
Expects the trained artifacts (rf_crop_model.joblib, label_encoder.joblib)
in ../models/ relative to this file. Adjust MODEL_DIR below if you copy
this app elsewhere.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent.parent / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_RANGES = {
    "N":           dict(label="Nitrogen (N, kg/ha)",     min=0.0,  max=140.0, default=50.0,  step=1.0),
    "P":           dict(label="Phosphorus (P, kg/ha)",   min=5.0,  max=145.0, default=53.0,  step=1.0),
    "K":           dict(label="Potassium (K, kg/ha)",    min=5.0,  max=205.0, default=48.0,  step=1.0),
    "temperature": dict(label="Temperature (°C)",        min=8.0,  max=44.0,  default=25.0,  step=0.1),
    "humidity":    dict(label="Relative humidity (%)",   min=14.0, max=100.0, default=71.0,  step=0.5),
    "ph":          dict(label="Soil pH",                 min=3.5,  max=10.0,  default=6.5,   step=0.1),
    "rainfall":    dict(label="Rainfall (mm)",            min=20.0, max=300.0, default=103.0, step=1.0),
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "rf_crop_model.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return model, label_encoder


def main():
    st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="centered")
    st.title("🌱 Crop Recommendation Classifier")
    st.caption(
        "Educational prototype — enter soil and climate readings to get a "
        "recommended crop, trained on the Kaggle Crop Recommendation Dataset."
    )

    try:
        model, label_encoder = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run `python train.py` from the project "
            "root first, so that models/rf_crop_model.joblib and "
            "models/label_encoder.joblib exist."
        )
        st.stop()

    st.subheader("Input conditions")
    col1, col2 = st.columns(2)
    values = {}
    for i, feat in enumerate(FEATURES):
        cfg = FEATURE_RANGES[feat]
        target_col = col1 if i % 2 == 0 else col2
        values[feat] = target_col.slider(
            cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
            value=cfg["default"], step=cfg["step"],
        )

    if st.button("Recommend crop", type="primary"):
        X = pd.DataFrame([values])[FEATURES]
        pred_id = model.predict(X)[0]
        pred_crop = label_encoder.inverse_transform([pred_id])[0]
        proba = model.predict_proba(X)[0]

        st.success(f"Recommended crop: **{pred_crop.capitalize()}**")

        top_n = 5
        top_idx = np.argsort(proba)[::-1][:top_n]
        top_df = pd.DataFrame({
            "crop": label_encoder.inverse_transform(top_idx),
            "probability": proba[top_idx],
        })
        st.subheader(f"Top {top_n} candidate crops")
        st.bar_chart(top_df.set_index("crop")["probability"])
        st.dataframe(top_df.style.format({"probability": "{:.1%}"}), hide_index=True)

    with st.expander("About this model"):
        st.markdown(
            "- **Model:** Random Forest classifier (300 trees), scikit-learn\n"
            "- **Test accuracy:** ~99.3% · **5-fold CV accuracy:** ~99.5%\n"
            "- **Features:** N, P, K, temperature, humidity, pH, rainfall\n"
            "- **Classes:** 22 crop types, 100 balanced samples each in training data\n"
            "- This is an educational prototype, not an agronomic decision tool. "
            "Always validate recommendations against local agricultural guidance."
        )


if __name__ == "__main__":
    main()
