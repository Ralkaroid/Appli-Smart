
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from tflite_runtime.interpreter import Interpreter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "detect.tflite"

THRESHOLD = 0.50

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

NUM_THREADS = 2
TARGET_INFERENCE_FPS = 4.0
JPEG_QUALITY = 55

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

# BCM GPIO17 = physical pin 11
PERSON_PIN = 17

# Switch output LOW only after this many seconds
# without detecting a person.
NO_PERSON_DELAY = 5.0


# ============================================================
# FLASK AND GPIO
# ============================================================

app = Flask(__name__)

person_output = DigitalOutputDevice(
    PERSON_PIN,
    active_high=True,
    initial_value=False
)

lock = threading.Lock()
frame_ready = threading.Condition(lock)

latest_jpeg = None
frame_id = 0

status = {
    "person_detected": False,
    "confidence": 0.0,
    "inference_ms": 0.0,
    "fps": 0.0,
    "gpio17": "LOW",
    "error": ""
}

# Shared event lets Flask stop the detector during shutdown.
stop_event = threading.Event()


# ============================================================
# WEB DASHBOARD
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Raspberry Pi Person Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 20px auto;
            padding: 0 14px;
            background: #f4f6f8;
            color: #202124;
        }
        h1 { font-size: 25px; }
        .card {
            background: white;
            padding: 16px;
            margin-bottom: 14px;
            border-radius: 10px;
            box-shadow: 0 1px 5px #0001;
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            background: #222;
        }
        .value { font-size: 20px; font-weight: bold; }
        .present { color: #16803c; }
        .absent { color: #b42318; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        }
        .label { color: #666; font-size: 14px; margin-bottom: 7px; }
        .note { color: #555; font-size: 14px; }
    </style>
</head>
<body>
    <h1>Person Detection and GPIO Control</h1>

    <div class="card">
        <img src="/video_feed" alt="Live camera stream">
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Person status</div>
            <div id="person" class="value">Starting...</div>
        </div>

        <div class="card">
            <div class="label">GPIO17 output</div>
            <div id="gpio" class="value">LOW</div>
        </div>

        <div class="card">
            <div class="label">Confidence</div>
            <div id="confidence" class="value">--</div>
        </div>

        <div class="card">
            <div class="label">Inference time</div>
            <div id="latency" class="value">--</div>
        </div>

        <div class="card">
            <div class="label">Detection FPS</div>
            <div id="fps" class="value">--</div>
        </div>
    </div>

    <div class="card">
        <div class="label">System status</div>
        <div id="error">Waiting for camera and model...</div>
        <p class="note">
            GPIO17 is HIGH when a person is detected. It returns LOW
            after the configured absence delay.
        </p>
    </div>

<script>
async function updateStatus() {
    try {
        const response = await fetch("/status", {cache: "no-store"});
        const data = await response.json();

        const person = document.getElementById("person");
        person.textContent = data.person_detected
            ? "PERSON PRESENT"
            : "NO PERSON";
        person.className = "value " +
            (data.person_detected ? "present" : "absent");

        const gpio = document.getElementById("gpio");
        gpio.textContent = data.gpio17;
        gpio.className = "value " +
            (data.gpio17 === "HIGH" ? "present" : "absent");

        document.getElementById("confidence").textContent =
            (data.confidence * 100).toFixed(1) + "%";

        document.getElementById("latency").textContent =
            data.inference_ms.toFixed(0) + " ms";

        document.getElementById("fps").textContent =
            data.fps.toFixed(1);

        document.getElementById("error").textContent =
            data.error || "Camera and detector running";
    } catch (error) {
        document.getElementById("error").textContent =
            "Cannot reach the detector. Check the Pi and server.";
    }
}

updateStatus();
setInterval(updateStatus, 1500);
</script>
</body>
</html>
"""


# ============================================================
# PERSON DETECTION AND CAMERA WORKER
# ============================================================

def detector_worker():
    global latest_jpeg, frame_id

    camera = None
    last_person_seen = None
    previous_frame_time = time.monotonic()
    measured_fps = 0.0

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        print("Loading TensorFlow Lite model...", flush=True)

        interpreter = Interpreter(
            model_path=str(MODEL_PATH),
            num_threads=NUM_THREADS
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index = input_details[0]["index"]
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]

        model_height = int(input_shape[1])
        model_width = int(input_shape[2])

        print(
            f"Model input: {input_shape}, "
            f"dtype: {input_dtype}",
            flush=True
        )

        print("Starting IMX219 camera...", flush=True)

        camera = Picamera2()

        camera_config = camera.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "RGB888"
            },
            buffer_count=2
        )

        camera.configure(camera_config)
        camera.start()
        time.sleep(1.0)

        print("Camera started.", flush=True)

        while not stop_event.is_set():
            loop_start = time.monotonic()

            # Capture camera image.
            frame = camera.capture_array()

            # Picamera2 RGB888 frames on this setup are BGR-ordered
            # in memory, so convert to RGB before model inference.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resized = cv2.resize(
                rgb_frame,
                (model_width, model_height)
            )

            model_input = np.expand_dims(resized, axis=0)

            # Handle either uint8 or float model inputs.
            if input_dtype == np.float32:
                model_input = model_input.astype(np.float32)
                scale, zero_point = input_details[0]["quantization"]
                if scale:
                    model_input = (
                        model_input.astype(np.float32) / 255.0
                    )
            else:
                model_input = model_input.astype(input_dtype)

            # Run inference.
            inference_start = time.monotonic()

            interpreter.set_tensor(input_index, model_input)
            interpreter.invoke()

            inference_ms = (
                time.monotonic() - inference_start
            ) * 1000.0

            outputs = [
                interpreter.get_tensor(item["index"])
                for item in output_details
            ]

            # Standard SSD output order:
            # boxes, classes, scores, number of detections.
            if len(outputs) < 3:
                raise RuntimeError(
                    "Unexpected model output format."
                )

            boxes = np.squeeze(outputs[0])
            classes = np.squeeze(outputs[1])
            scores = np.squeeze(outputs[2])

            boxes = np.atleast_2d(boxes)
            classes = np.atleast_1d(classes)
            scores = np.atleast_1d(scores)

            person_detected = False
            best_confidence = 0.0
            best_box = None

            for box, class_id, score in zip(boxes, classes, scores):
                confidence = float(score)

                # This model uses person class ID 0.
                if int(class_id) == 0 and confidence >= THRESHOLD:
                    person_detected = True

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_box = box

            # ------------------------------------------------
            # GPIO CONTROL
            # ------------------------------------------------

            now = time.monotonic()

            if person_detected:
                last_person_seen = now
                person_output.on()
            elif (
                last_person_seen is None
                or now - last_person_seen >= NO_PERSON_DELAY
            ):
                person_output.off()

            gpio_state = "HIGH" if person_output.value else "LOW"

            # ------------------------------------------------
            # DRAW DETECTION
            # ------------------------------------------------

            display = frame.copy()

            if person_detected and best_box is not None:
                ymin, xmin, ymax, xmax = best_box

                x1 = max(0, int(xmin * CAMERA_WIDTH))
                y1 = max(0, int(ymin * CAMERA_HEIGHT))
                x2 = min(CAMERA_WIDTH - 1, int(xmax * CAMERA_WIDTH))
                y2 = min(CAMERA_HEIGHT - 1, int(ymax * CAMERA_HEIGHT))

                cv2.rectangle(
                    display, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    display,
                    f"Person: {best_confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            label = (
                "PERSON PRESENT"
                if person_detected
                else "NO PERSON"
            )

            label_color = (
                (0, 255, 0)
                if person_detected
                else (0, 0, 255)
            )

            cv2.putText(
                display,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                label_color,
                2
            )

            cv2.putText(
                display,
                f"GPIO17: {gpio_state}",
                (8, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

            # Calculate approximate detector FPS.
            current_time = time.monotonic()
            elapsed = current_time - previous_frame_time
            previous_frame_time = current_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                measured_fps = (
                    instant_fps if measured_fps == 0
                    else 0.8 * measured_fps + 0.2 * instant_fps
                )

            # Encode JPEG for browser streaming.
            ok, encoded = cv2.imencode(
                ".jpg",
                display,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if ok:
                with frame_ready:
                    latest_jpeg = encoded.tobytes()
                    frame_id += 1

                    status.update({
                        "person_detected": person_detected,
                        "confidence": best_confidence,
                        "inference_ms": inference_ms,
                        "fps": measured_fps,
                        "gpio17": gpio_state,
                        "error": ""
                    })

                    frame_ready.notify_all()

            # Limit processing rate to reduce CPU load.
            elapsed_loop = time.monotonic() - loop_start
            remaining = (1.0 / TARGET_INFERENCE_FPS) - elapsed_loop

            if remaining > 0:
                stop_event.wait(remaining)

    except Exception as exc:
        print(f"Detector error: {exc}", flush=True)

        with frame_ready:
            status["error"] = str(exc)
            frame_ready.notify_all()

    finally:
        # Fail safe: set GPIO LOW if the camera/model stops.
        person_output.off()

        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

            try:
                camera.close()
            except Exception:
                pass

        print("Detector stopped; GPIO17 set LOW.", flush=True)


# ============================================================
# FLASK ROUTES
# ============================================================

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
        last_sent_frame = -1

        while not stop_event.is_set():
            with frame_ready:
                frame_ready.wait_for(
                    lambda: (
                        stop_event.is_set()
                        or (
                            latest_jpeg is not None
                            and frame_id != last_sent_frame
                        )
                    ),
                    timeout=2.0
                )

                if stop_event.is_set():
                    break

                if latest_jpeg is None or frame_id == last_sent_frame:
                    continue

                jpeg = latest_jpeg
                last_sent_frame = frame_id

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    worker = threading.Thread(
        target=detector_worker,
        daemon=True
    )
    worker.start()

    print(
        f"Dashboard: http://<PI-IP>:{WEB_PORT}",
        flush=True
    )
    print(
        f"GPIO17: physical pin 11; absence delay: "
        f"{NO_PERSON_DELAY}s",
        flush=True
    )

    try:
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()

        with frame_ready:
            frame_ready.notify_all()

        worker.join(timeout=5)
        person_output.off()
        person_output.close()
