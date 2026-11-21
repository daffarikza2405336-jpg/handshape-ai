# HandShape AI

A gesture-recognition app built with Streamlit, OpenCV, and ONNX hand-detection
models.

## Run locally

Install the dependencies and start Streamlit from the project directory:

```bash
python3 -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Use **Kamera** to capture a hand
gesture or **Unggah gambar** to select a JPEG/PNG image. Images are processed
for recognition and are not saved. The Streamlit version analyzes one image at
a time rather than continuously streaming video.

The original desktop app can still be started with:

```bash
python3 main.py
```

## Publish on Streamlit Community Cloud

1. Push this repository, including `assets/models/`, to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and select **Create app**.
3. Choose this repository, the `main` branch, and `streamlit_app.py` as the
   app file, then deploy.
4. Share the HTTPS URL Streamlit provides. Camera access requires HTTPS and
   permission from the browser.

The app uses the free Community Cloud tier. Apps may sleep after inactivity and
take a short time to wake when opened again.
