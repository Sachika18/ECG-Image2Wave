import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os

from app import ECGPredictor

st.set_page_config(page_title="ECG Digitizer", layout="wide")

# Load model
@st.cache_resource
def load_model():
    try:
        return ECGPredictor("resnet_unet_best.pth")
    except Exception as e:
        # Surface the error in Streamlit UI so user sees why app may be blank
        st.error(f"Failed to initialize model: {e}")
        # Return None so the rest of the script can handle missing model gracefully
        return None

predictor = load_model()

if predictor is None:
    st.stop()

st.title("📈 ECG Image → Digital Signal Converter")
st.write("Upload ECG panel images and convert them into 12-lead digital ECG waveform.")

uploaded_files = st.file_uploader(
    "Upload ECG PNG panels (multiple files)",
    type=["png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded.")

    # Save files temporarily
    panel_paths = []
    for f in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.write(f.read())
        temp_file.close()
        panel_paths.append(temp_file.name)

    # Predict
    st.info("Running model… this may take a few seconds.")
    wave = predictor.predict(panel_paths)   # (12, T)

    st.success("Digitization complete!")

    # Plot waveform
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    lead_idx = st.selectbox("Select lead to visualize:", list(range(12)),
                            format_func=lambda i: lead_names[i])

    plt.figure(figsize=(14,4))
    plt.plot(wave[lead_idx][:2000])
    plt.title(f"Lead {lead_names[lead_idx]} — First 2000 points")
    plt.grid(True)

    st.pyplot(plt)

    # Download CSV
    df = pd.DataFrame(wave.T, columns=lead_names)
    csv_path = "predicted_ecg.csv"
    df.to_csv(csv_path, index=False)

    st.download_button(
        "Download Digital ECG (CSV)",
        data=open(csv_path, "rb").read(),
        file_name="predicted_ecg.csv",
        mime="text/csv"
    )
