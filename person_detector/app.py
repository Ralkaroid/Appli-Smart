import time
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# ---------- Configuration ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "detect.tflite"

THRESHOLD = 0.50
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
NUM_THREADS = 2

# Limit model inference to reduce CPU load.
TARGET_INFERENCE_FPS = 4.0
JPEG_QUALITY = 55
WEB_PORT = 5000

app = Flask(__name__)

lock = threading.Lock()
frame_ready = threading.Condition(lock)

latest_jpeg = None
frame_id = 0

status = {
    "person_detected": False,
    "confidence": 0.0,
    "inference_ms": 0.0,
    "fps": 0.0,
    "error": "Starting detector..."
}

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Raspberry Pi Person Detection</title>
<style>
body {
    margin: 0; padding: 18px; background: #101827;
    color: #f3f4f6; font-family: Arial, sans-serif;
}
main { max-width: 850px; margin: auto; }
h1 { margin-bottom: 6px; }
.muted { color: #aab5c5; }
#status {
    padding: 15px; margin: 16px 0; border-radius: 10px;
    font-size: 21px; font-weight: bold;
}
.detected { background: #14532d; color: #bbf7d0; }
.absent { background: #7f1d1d; color: #fecaca; }
.waiting { background: #374151; color: white; }
img {
    width: 100%; height: auto; display: block;
    background: #000; border-radius: 10px;
}
.stats {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 10px; margin-top: 14px;
}
.card { background: #1f2937; padding: 14px; border-radius: 9px; }
.label { color: #9ca3af; font-size: 13px; }
.value { font-size: 20px; font-weight: bold; margin-top: 7px; }
@media(max-width: 500px) {
    .stats { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<main>
    <h1>Person Detection</h1>
    <p class="muted">Raspberry Pi Zero 2 W | MobileNet SSD</p>

    <div id="status" class="waiting">Starting camera...</div>
    <img src="/video_feed" alt="Live detection stream">

    <div class="stats">
        <div class="card">
            <div class="label">Confidence</div>
            <div id="confidence" class="value">--</div>
        </div>
        <div class="card">
            <div class="label">Inference latency</div>
            <div id="latency" class="value">--</div>
        </div>
        <div class="card">
            <div class="label">Inference FPS</div>
            <div id="fps" class="value">--</div>
        </div>
    </div>
</main>

<script>
async function refreshStatus() {
    const box = document.getElementById('status');

    try {
        const response = await fetch('/status', {cache: 'no-store'});
        const d = await response.json();

        if (d.error) {
            box.className = 'waiting';
            box.textContent = d.error;
        } else if (d.person_detected) {
            box.className = 'detected';
                        box.textContent = 'PERSON DETECTED';
        } else {
            box.className = 'absent';
            box.textContent = 'NO PERSON DETECTED';
        }

        document.getElementById('confidence').textContent =
            d.person_detected ? (d.confidence * 100).toFixed(1) + '%' : '--';

        document.getElementById('latency').textContent =
            d.inference_ms.toFixed(0) + ' ms';

        document.getElementById('fps').textContent =
            d.fps.toFixed(1);

    } catch (e) {
        box.className = 'waiting';
        box.textContent = 'Waiting for server...';
    }
}

setInterval(refreshStatus, 1500);
refreshStatus();
</script>
</body>
</html>
"""


def detector_worker():
    global latest_jpeg, frame_id

    camera = None

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        print("Loading model:", MODEL_PATH)

        interpreter = Interpreter(
            model_path=str(MODEL_PATH),
            num_threads=NUM_THREADS
        )
        interpreter.allocate_tensors()

        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()

        shape = input_info["shape"]
        model_h, model_w = int(shape[1]), int(shape[2])

        if input_info["dtype"] != np.uint8:
            raise RuntimeError("Expected a quantized uint8 model.")

        if len(output_info) < 3:
            raise RuntimeError("Unexpected model output format.")

        print("Model input:", shape)
        print("Initializing camera...")

        camera = Picamera2()
        camera.configure(
            camera.create_video_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": "RGB888"
                },
                buffer_count=2
            )
        )
        camera.start()
        time.sleep(1)

        print("Camera ready. Detection is running.")

        period = 1.0 / TARGET_INFERENCE_FPS
        stats_start = time.monotonic()
        inference_count = 0
        measured_fps = 0.0

        while True:
            loop_start = time.monotonic()

            # Capture and preprocess
            frame = camera.capture_array()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (model_w, model_h))
            tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

            # Run inference
            interpreter.set_tensor(input_info["index"], tensor)

            t0 = time.monotonic()
            interpreter.invoke()
            inference_ms = (time.monotonic() - t0) * 1000.0

            outputs = [
                interpreter.get_tensor(item["index"])
                for item in output_info
            ]

            boxes = np.asarray(outputs[0]).reshape(-1, 4)
            classes = np.asarray(outputs[1]).reshape(-1)
            scores = np.asarray(outputs[2]).reshape(-1)

            person_scores = []
            for box, class_id, score in zip(boxes, classes, scores):
                score = float(score)

                if int(class_id) != 0 or score < THRESHOLD:
                    continue

                person_scores.append(score)

                y1, x1, y2, x2 = box
                h, w = frame.shape[:2]

                x1 = max(0, min(w - 1, int(x1 * w)))
                x2 = max(0, min(w - 1, int(x2 * w)))
                y1 = max(0, min(h - 1, int(y1 * h)))
                y2 = max(0, min(h - 1, int(y2 * h)))

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 220, 80), 2
                )
                cv2.putText(
                    frame, f"Person {score:.2f}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 80), 1
                )

            detected = bool(person_scores)
            confidence = max(person_scores, default=0.0)

            # Calculate actual inference-loop rate
            inference_count += 1
            elapsed = time.monotonic() - stats_start

            if elapsed >= 5.0:
                measured_fps = inference_count / elapsed
                inference_count = 0
                stats_start = time.monotonic()

            label = (
                f"PERSON DETECTED {confidence:.2f}"
                if detected else "NO PERSON DETECTED"
            ) 
            color = (0, 160, 0) if detected else (0, 0, 190)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), color, -1)
            cv2.putText(
                frame, label, (5, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1
            )

            # Encode once; clients share the same JPEG frame
            ok, encoded = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            if ok:
                with frame_ready:
                    latest_jpeg = encoded.tobytes()
                    frame_id += 1

                    status.update({
                        "person_detected": detected,
                        "confidence": confidence,
                        "inference_ms": inference_ms,
                        "fps": measured_fps,
                        "error": ""
                    })

                    frame_ready.notify_all()

            # Cap inference rate to avoid unnecessary CPU use
            remaining = period - (time.monotonic() - loop_start)

            if remaining > 0:
                time.sleep(remaining)

    except Exception as exc:
        print("Detector error:", repr(exc))
        with lock:
            status["error"] = f"Detector error: {exc}"

    finally:
        if camera is not None:
            camera.stop()
            camera.close()

        print("Camera stopped.")


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/status")
def get_status():
    with lock:
        return jsonify(status.copy())

@app.route("/video_feed")
def video_feed():

    def generate():
        last_frame_id = -1

        while True:
            with frame_ready:
                frame_ready.wait_for(
                    lambda: frame_id != last_frame_id
                    or bool(status["error"]),
                    timeout=5.0
                )

                current_id = frame_id
                jpeg = latest_jpeg
                error = status["error"]

            if error and jpeg is None:
                break

            if jpeg is None or current_id == last_frame_id:
                continue

            last_frame_id = current_id

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpeg + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    threading.Thread(
        target=detector_worker,
        daemon=True,
        name="person-detector"
    ).start()

    print("Dashboard: http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=WEB_PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )


