const video = document.querySelector("#camera-video");
const overlay = document.querySelector("#landmark-canvas");
const cameraStage = document.querySelector("#camera-stage");
const placeholder = document.querySelector("#camera-placeholder");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const captureButton = document.querySelector("#capture-frame");
const statusLabel = document.querySelector("#camera-status");
const message = document.querySelector("#camera-message");
const sessionId = globalThis.crypto?.randomUUID?.().replaceAll("-", "") ??
  `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
const frameCanvas = document.createElement("canvas");
const frameContext = frameCanvas.getContext("2d", { willReadFrequently: false });
const overlayContext = overlay.getContext("2d");
const shapeCanvas = document.querySelector("#shape-canvas");
const shapeContext = shapeCanvas.getContext("2d");
const connectionPairs = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

let cameraStream = null;
let frameTimer = null;
let frameNumber = 0;
let isProcessing = false;
let lastResult = null;

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function clearVisualizations() {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  shapeContext.clearRect(0, 0, shapeCanvas.width, shapeCanvas.height);
}

function drawLandmarks(hands, result) {
  const width = overlay.width;
  const height = overlay.height;
  overlayContext.clearRect(0, 0, width, height);
  for (const hand of hands) {
    const points = hand.landmarks.map(([x, y]) => [x * width, y * height]);
    overlayContext.lineWidth = Math.max(2, width / 360);
    overlayContext.strokeStyle = "rgba(185, 230, 197, .86)";
    overlayContext.fillStyle = "#e7f6eb";
    overlayContext.shadowColor = "rgba(185, 230, 197, .55)";
    overlayContext.shadowBlur = 7;
    for (const [start, end] of connectionPairs) {
      overlayContext.beginPath();
      overlayContext.moveTo(...points[start]);
      overlayContext.lineTo(...points[end]);
      overlayContext.stroke();
    }
    for (const [x, y] of points) {
      overlayContext.beginPath();
      overlayContext.arc(x, y, Math.max(2, width / 190), 0, Math.PI * 2);
      overlayContext.fill();
    }
  }
  overlayContext.shadowBlur = 0;
  drawShape(result);
}

function drawShape(result) {
  const width = shapeCanvas.width;
  const height = shapeCanvas.height;
  shapeContext.clearRect(0, 0, width, height);
  shapeContext.strokeStyle = "rgba(185, 230, 197, .14)";
  shapeContext.lineWidth = 1;
  for (let x = 25; x < width; x += 25) {
    shapeContext.beginPath();
    shapeContext.moveTo(x, 0);
    shapeContext.lineTo(x, height);
    shapeContext.stroke();
  }
  for (let y = 20; y < height; y += 20) {
    shapeContext.beginPath();
    shapeContext.moveTo(0, y);
    shapeContext.lineTo(width, y);
    shapeContext.stroke();
  }
  if (!result) {
    shapeContext.fillStyle = "#777f82";
    shapeContext.font = "11px 'DM Mono', monospace";
    shapeContext.textAlign = "center";
    shapeContext.fillText("MENUNGGU GESTUR", width / 2, height / 2);
    return;
  }

  shapeContext.strokeStyle = "#b9e6c5";
  shapeContext.fillStyle = "rgba(185, 230, 197, .07)";
  shapeContext.lineWidth = 3;
  shapeContext.lineJoin = "round";
  const points = result.points;
  if (result.label === "CIRCLE") {
    shapeContext.beginPath();
    shapeContext.arc(width / 2, height / 2, Math.min(width, height) * .28, 0, Math.PI * 2);
    shapeContext.fill();
    shapeContext.stroke();
  } else if (result.label === "V SHAPE") {
    shapeContext.beginPath();
    shapeContext.moveTo(width * .29, height * .32);
    shapeContext.lineTo(width * .5, height * .73);
    shapeContext.lineTo(width * .71, height * .32);
    shapeContext.stroke();
  } else if (points?.length >= 3) {
    const xs = points.map(([x]) => x);
    const ys = points.map(([, y]) => y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const scale = Math.min(width * .66 / Math.max(maxX - minX, .03), height * .62 / Math.max(maxY - minY, .03));
    const mapped = points.map(([x, y]) => [
      width / 2 + (x - (minX + maxX) / 2) * scale,
      height / 2 + (y - (minY + maxY) / 2) * scale,
    ]);
    shapeContext.beginPath();
    shapeContext.moveTo(...mapped[0]);
    for (const point of mapped.slice(1)) shapeContext.lineTo(...point);
    shapeContext.closePath();
    shapeContext.fill();
    shapeContext.stroke();
    for (const [x, y] of mapped) {
      shapeContext.beginPath();
      shapeContext.arc(x, y, 4, 0, Math.PI * 2);
      shapeContext.fillStyle = "#e9f8ed";
      shapeContext.fill();
    }
  }
}

function updateResults(data) {
  document.querySelector("#hands-count").textContent = String(data.hand_count).padStart(2, "0");
  document.querySelector("#processing-speed").textContent = `${data.processing_ms} ms`;
  document.querySelector("#detection-count").textContent = String(data.total_detections).padStart(2, "0");
  document.querySelector("#landmark-count").textContent = String(data.hand_count * 21).padStart(2, "0");
  document.querySelector("#frame-counter").textContent = `FRAME ${String(frameNumber).padStart(4, "0")}`;
  const result = data.result;
  document.querySelector("#gesture-result").textContent = result?.label ?? "—";
  document.querySelector("#shape-result").textContent = result ? result.shape_name.toUpperCase() : (data.hand_count ? "GESTUR BELUM DIKENALI" : "TIDAK ADA TANGAN");
  drawLandmarks(data.hands, result);
  lastResult = result;
}

async function processFrame() {
  if (!cameraStream || isProcessing || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  isProcessing = true;
  frameNumber += 1;
  frameCanvas.width = 640;
  frameCanvas.height = 480;
  frameContext.save();
  frameContext.translate(frameCanvas.width, 0);
  frameContext.scale(-1, 1);
  frameContext.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  frameContext.restore();

  try {
    const blob = await new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", .72));
    if (!blob) throw new Error("Frame kamera tidak dapat disiapkan.");
    const response = await fetch("/api/recognize", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Session-ID": sessionId,
      },
      body: blob,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Pengenalan gestur gagal.");
    updateResults(data);
    setMessage("");
  } catch (error) {
    if (cameraStream) setMessage(error.message || "Koneksi ke server pengenalan terputus.", true);
  } finally {
    isProcessing = false;
  }
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setMessage("Akses kamera memerlukan HTTPS atau localhost. Buka situs melalui koneksi HTTPS.", true);
    return;
  }
  startButton.disabled = true;
  setMessage("Meminta izin kamera…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    cameraStream = stream;
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    cameraStage.classList.add("camera-active");
    placeholder.hidden = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    captureButton.disabled = false;
    statusLabel.textContent = "LIVE";
    document.querySelector("#analysis-pulse").classList.add("active");
    setMessage("Kamera aktif. Tunjukkan gestur di dalam frame.");
    clearVisualizations();
    frameTimer = window.setInterval(processFrame, 180);
  } catch (error) {
    startButton.disabled = false;
    const explanation = error.name === "NotAllowedError"
      ? "Izin kamera ditolak. Izinkan akses kamera di pengaturan browser."
      : error.name === "NotFoundError"
        ? "Kamera tidak ditemukan pada perangkat ini."
        : error.message || "Kamera tidak dapat diaktifkan.";
    setMessage(explanation, true);
  }
}

function stopCamera() {
  if (frameTimer !== null) window.clearInterval(frameTimer);
  frameTimer = null;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.srcObject = null;
  cameraStage.classList.remove("camera-active");
  placeholder.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  captureButton.disabled = true;
  statusLabel.textContent = "STANDBY";
  document.querySelector("#analysis-pulse").classList.remove("active");
  document.querySelector("#gesture-result").textContent = "—";
  document.querySelector("#shape-result").textContent = "MENUNGGU INPUT";
  document.querySelector("#hands-count").textContent = "00";
  document.querySelector("#processing-speed").textContent = "— ms";
  document.querySelector("#landmark-count").textContent = "00";
  lastResult = null;
  isProcessing = false;
  clearVisualizations();
  setMessage("Kamera dihentikan.");
}

function saveFrame() {
  if (!cameraStream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  const capture = document.createElement("canvas");
  capture.width = video.videoWidth;
  capture.height = video.videoHeight;
  const context = capture.getContext("2d");
  context.translate(capture.width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, capture.width, capture.height);
  capture.toBlob((blob) => {
    if (!blob) {
      setMessage("Frame tidak dapat disimpan.", true);
      return;
    }
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `handshape-${new Date().toISOString().replaceAll(":", "-")}.jpg`;
    link.click();
    URL.revokeObjectURL(link.href);
    setMessage(lastResult ? `Frame tersimpan · ${lastResult.label}` : "Frame tersimpan.");
  }, "image/jpeg", .92);
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
captureButton.addEventListener("click", saveFrame);
window.addEventListener("pagehide", stopCamera);

for (const card of document.querySelectorAll("[data-gesture]")) {
  card.addEventListener("click", () => {
    const requestedGesture = card.dataset.gesture;
    document.querySelector("#shape-result").textContent = `TARGET: ${requestedGesture}`;
    if (!cameraStream) {
      setMessage(`Pilih "Mulai kamera" untuk mencoba gestur ${requestedGesture}.`);
    }
  });
}

fetch("/api/health")
  .then((response) => {
    if (!response.ok) throw new Error("Server pengenalan tidak tersedia.");
    return response.json();
  })
  .then((health) => {
    if (!health.model_loaded) throw new Error("Model pengenalan belum siap.");
  })
  .catch((error) => setMessage(error.message, true));

drawShape(null);
