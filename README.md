# HandShape AI Web App

## Run locally

From the project directory, install the web runtime and start the server:

```bash
python3 -m pip install -r requirements-web.txt
python3 web_app.py
```

Open [http://localhost:8000](http://localhost:8000). Camera access is available on
`localhost` and on HTTPS sites. Video frames are sent to this app for recognition
and are not stored.

## Publish on Render

1. Push this project, including `assets/models/`, `web/`, `web_app.py`,
   `web_server.py`, `requirements-web.txt`, and `render.yaml`, to a GitHub
   repository. Confirm that the ONNX model files are included.
2. In Render, choose **New → Blueprint**, connect the repository, and apply the
   `render.yaml` service configuration.
3. Wait for the build and deploy to finish, then open the HTTPS URL Render
   provides.
4. Allow camera access in the browser when prompted.

The browser keeps the camera feed on the device and sends individual frames to
the recognition server over the site's connection. The server processes those
frames in memory and does not save them. Do not expose the app over plain HTTP
on a public network; browser camera access requires HTTPS.
