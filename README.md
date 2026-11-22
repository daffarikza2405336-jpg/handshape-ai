# HandShape AI

A gesture-recognition web app built with OpenCV and ONNX hand-detection models.

## Run locally

Install the web app dependencies and start the local server from the project
directory:

```bash
python3 -m pip install -r requirements-web.txt
python3 web_app.py
```

Open `http://localhost:8000` in your browser and allow camera access. The app
processes the camera locally. Press `Ctrl+C` in the terminal to stop the server.

The original desktop app can still be started with:

```bash
python3 main.py
```
