# ECG Digitizer

Paper ECG → Digital waveforms (12-lead) using deep learning

Short description

ECG Digitizer converts photographed or scanned 12-panel ECG images into
digital 12-lead ECG waveforms (CSV). The model uses a ResNet18 encoder to
extract features from each panel and a 1D UNet-style decoder to reconstruct
high-resolution waveforms.

Suggested repository description (one line):
Convert paper ECG panels to digital 12-lead waveforms using ResNet+UNet.

Project name suggestions (pick one):
- ECG Digitizer (recommended)
- Paper2ECG
- CardioScan
- ECG-Image2Wave
- PaperECG-Digitizer

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

Usage tips for GitHub description
- Use the suggested one-line description above as the repository "description".
- Pin a short README excerpt as the repo README summary (the first paragraph).

License
- Add a license file (e.g., MIT) if you plan to open-source the code.

Contact
- Add your name/email or a link to your GitHub profile here.

---

If you want, I can also:
- Add a `LICENSE` (MIT) and `.gitignore` tailored to Python projects
- Commit these files and provide the exact `git` commands for Windows
- Create a short GitHub Actions workflow to run a basic lint/test on push
