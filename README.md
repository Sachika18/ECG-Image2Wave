# ECG Digitizer

Paper ECG → Digital waveforms (12-lead) using deep learning

ECG Digitizer converts photographed or scanned 12-panel ECG images into
digital 12-lead ECG waveforms (CSV). The model uses a ResNet18 encoder to
extract features from each panel and a 1D UNet-style decoder to reconstruct
high-resolution waveforms.

Features
- Upload 12 ECG panel PNG images via Streamlit UI
- Preprocessing (resize, normalize) and model inference
- Produces 12 waveforms (10,250 timepoints per lead)
- View plots in-browser and download CSV of digitized ECG

Requirements
- Python 3.8+
- See `requirements.txt` for exact packages (PyTorch, torchvision, streamlit, numpy, pandas, pillow, matplotlib)

Quick start (Windows CMD)

1. Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```cmd
pip install -r requirements.txt
```

3. Run the Streamlit app:

```cmd
python -m streamlit run predict.py
```

Notes
- Place `resnet_unet_best.pth` in the project root (this repo already contains it).
- The app expects exactly 12 PNG panel images for prediction.

Model & architecture
- Encoder: ResNet18 per-panel feature extractor
- Fusion: Mean pooling across 12 panels + small FC fusion network
- Decoder: 1D UNet-style ConvTranspose decoder producing (12, 10250) outputs


