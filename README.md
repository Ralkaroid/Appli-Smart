# Appli-Smart: Person Detection and GPIO-Based Occupancy Control Using Raspberry Pi Zero 2 W

A real-time edge AI project that detects people using an IMX219 camera and a pretrained TensorFlow Lite MobileNet SSD model on the Raspberry Pi Zero 2 W. The system automatically controls a GPIO output based on person detection and provides a live web dashboard for monitoring.

## 1. Project Overview

This project implements a camera-based person detection and occupancy control system using a Raspberry Pi Zero 2 W.

The IMX219 camera captures video frames, a quantized MobileNet SSD model detects people, and the detection result determines the state of GPIO17. A Flask web application provides live video streaming and displays the detection status, confidence score, inference time, FPS, and GPIO state.

The system is designed for lightweight edge AI applications that combine computer vision, web technologies, and hardware control.

### Features

- Real-time person detection using TensorFlow Lite.
- Pretrained MobileNet SSD v1 model trained on the COCO dataset.
- Raspberry Pi Zero 2 W and IMX219 camera integration.
- Live annotated video stream through a web dashboard.
- Automatic GPIO17 control based on person presence.
- Configurable confidence threshold and absence delay.
- Displays confidence, inference latency, and detection FPS.
- Lightweight processing settings for resource-constrained hardware.
- Local network access through a browser.

## 2. Hardware Requirements

| Component | Purpose |
|---|---|
| Raspberry Pi Zero 2 W | Runs the model and web server |
| IMX219 camera module | Captures video |
| microSD card | Stores the OS, application, and model |
| Stable power supply | Powers the Raspberry Pi and camera |
| Relay module or transistor driver (optional) | Interfaces GPIO with an external load |
| External load (optional) | Light, indicator, or other controlled device |

## 3. Software Requirements

- Raspberry Pi OS
- Python 3.11
- Picamera2
- OpenCV
- NumPy
- TensorFlow Lite Runtime
- Flask
- gpiozero

## 4. System Architecture

```text
             IMX219 Camera
                   |
                   v
          Raspberry Pi Zero 2 W
                   |
                   v
            Capture Camera Frame
                   |
                   v
          Image Preprocessing
          Resize to 300 x 300
                   |
                   v
       TensorFlow Lite Inference
           MobileNet SSD v1
                   |
                   v
          Person Detection Result
                   |
           +-------+-------+
           |               |
           v               v
    Person Detected    No Detection
           |               |
           v               v
       GPIO17 HIGH      Absence Timer
                           |
                           v
                    Delay Expires
                           |
                           v
                       GPIO17 LOW

     Annotated Frames and Status
                   |
                   v
              Flask Server
                   |
                   v
           Web Browser Dashboard
```

## 5. How the System Works

### Step 1: Camera Capture

The IMX219 camera captures frames using Picamera2. The configured resolution is 320 × 240 pixels to reduce processing overhead.

### Step 2: Image Preprocessing

Each captured frame is converted to the appropriate color ordering and resized to 300 × 300 pixels, matching the model's expected input dimensions.

### Step 3: Person Detection

The pretrained MobileNet SSD model processes the image and returns bounding boxes, class IDs, and confidence scores.

The application treats a detection as a person when the class ID is `0` and the confidence score is at least `0.50`.

### Step 4: GPIO Control

The application controls BCM GPIO17 based on the detection result.

| Condition | GPIO17 State | Voltage |
|---|---|---|
| Person detected | HIGH | Approximately 3.3 V |
| No person detected for 3 seconds | LOW | Approximately 0 V |

The absence delay helps prevent unnecessary switching when the model briefly misses a person.

### Step 5: Web Dashboard

The Flask dashboard displays:

- Live annotated camera feed.
- Person presence status.
- GPIO17 output state.
- Detection confidence.
- Inference time in milliseconds.
- Approximate detection FPS.
- Application error status.

## 6. Project Directory Structure

```text
person_detector/
├── app.py
├── requirements.txt
├── test_model.py
├── README.md
├── models/
│   └── detect.tflite
├── .gitignore
└── .venv/                 # Local virtual environment; do not commit
```

## 7. Installation and Setup

The following instructions assume that Raspberry Pi OS is already installed and the IMX219 camera is connected.

### Step 1: Update the Raspberry Pi

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install System Dependencies

Install Python, camera support, image-processing libraries, and GPIO control:

```bash
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-gpiozero \
    unzip \
    wget
```

These packages provide camera access, image processing, and GPIO support through Raspberry Pi OS.

### Step 3: Create the Project Directory

```bash
mkdir -p ~/person_detector/models
cd ~/person_detector
```

### Step 4: Create a Python Virtual Environment

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

The `--system-site-packages` option allows the virtual environment to access system-installed libraries such as Picamera2, OpenCV, NumPy, and gpiozero.

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

## 8. Download the Pretrained MobileNet SSD Model

This project uses the quantized **MobileNet SSD v1 COCO model** distributed in TensorFlow Lite format.

The model is pretrained, so there is no need to train it from scratch.

### Step 1: Download the Model Archive

```bash
cd ~/person_detector/models

wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

### Step 2: Extract the Model

```bash
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

The archive contains the TensorFlow Lite model and associated label information.

Verify that the model exists:

```bash
ls -lh ~/person_detector/models/detect.tflite
```

The expected model path is:

```text
~/person_detector/models/detect.tflite
```

The application expects this filename. If the archive extracts the file to a different location, move it into the `models` directory and rename it to `detect.tflite`.

### Model Details

| Property | Value |
|---|---|
| Model | MobileNet SSD v1 |
| Dataset | COCO |
| Format | TensorFlow Lite |
| Quantization | Quantized |
| Input dimensions | 300 × 300 pixels |
| Input tensor shape | `[1, 300, 300, 3]` |
| Person class ID | `0` for the model used by this application |

**Important:** The model's license and redistribution terms should be checked before committing the model weights to a public repository. You can instead provide the download instructions and let users download the model themselves.

## 9. Install TensorFlow Lite Runtime and Python Libraries

TensorFlow Lite Runtime provides lightweight inference functionality without requiring the full TensorFlow framework.

### Step 1: Activate the Environment

```bash
cd ~/person_detector
source .venv/bin/activate
```

### Step 2: Install Python Dependencies

```bash
pip install flask tflite-runtime
```

Picamera2, OpenCV, NumPy, and gpiozero were installed using `apt` in the earlier steps.

The project uses the lightweight TensorFlow Lite Runtime package rather than the full TensorFlow package to reduce unnecessary installation overhead.

**Compatibility note:** TensorFlow Lite Runtime wheels depend on the Python version and CPU architecture. If pip reports that no compatible distribution is available, check the Python version and architecture before trying alternative installation methods. Avoid installing an incompatible wheel.

### Step 3: Create `requirements.txt`

From the project directory, create a dependency file:

```bash
nano requirements.txt
```

Add:

```text
Flask
tflite-runtime
```

Save the file.

These are the pip-installed dependencies. The remaining libraries are installed through Raspberry Pi OS packages.

## 10. Verify the Installation

### Test 1: Check Python Libraries

Run:

```bash
python -c "import tflite_runtime.interpreter as tflite; print('TensorFlow Lite Runtime OK')"
```

Then:

```bash
python -c "import cv2, numpy, flask, gpiozero; from picamera2 import Picamera2; print('All project libraries OK')"
```

Both commands should complete without import errors.

### Test 2: Verify the Model

Create a file named `test_model.py`:

```python
from pathlib import Path
from tflite_runtime.interpreter import Interpreter

model_path = Path("models/detect.tflite")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

interpreter = Interpreter(
    model_path=str(model_path),
    num_threads=2
)

interpreter.allocate_tensors()

print("Model loaded successfully")
print("Input details:", interpreter.get_input_details())
print("Output details:", interpreter.get_output_details())
```

Run the test from the project directory:

```bash
cd ~/person_detector
source .venv/bin/activate
python test_model.py
```

Confirm that the model loads successfully and reports an input shape of `[1, 300, 300, 3]`.

### Test 3: Verify the Camera

Close other applications that might be using the camera, then run:

```bash
rpicam-hello --timeout 5000
```

If the camera preview is unavailable over SSH, use the command to check camera startup and error messages. Stop the camera test before running `app.py`.

## 11. GPIO Connections

The application uses BCM GPIO17, which corresponds to physical pin 11.

| Raspberry Pi Connection | Function |
|---|---|
| GPIO17 / physical pin 11 | Digital output |
| GND / physical pin 6 | Ground reference for a compatible external interface |

### GPIO Output Logic

- Person detected: GPIO17 HIGH.
- No detection for the configured absence delay: GPIO17 LOW.
- Detector worker stops or encounters an error: the application attempts to set GPIO17 LOW.

**Electrical safety:** Raspberry Pi GPIO uses 3.3 V logic and cannot directly drive a relay coil or high-current load. Use a compatible relay module or transistor driver, with appropriate power and protection circuitry. Never apply 5 V to a GPIO input/output pin.

## 12. Configuration

The main settings are defined near the top of `app.py`.

| Parameter | Default | Purpose |
|---|---:|---|
| `THRESHOLD` | `0.50` | Minimum confidence for a person detection |
| `CAMERA_WIDTH` | `320` | Camera frame width |
| `CAMERA_HEIGHT` | `240` | Camera frame height |
| `NUM_THREADS` | `2` | TensorFlow Lite inference threads |
| `TARGET_INFERENCE_FPS` | `4.0` | Target inference rate |
| `JPEG_QUALITY` | `55` | JPEG streaming quality |
| `PERSON_PIN` | `17` | BCM GPIO output pin |
| `NO_PERSON_DELAY` | `3.0` seconds | Delay before switching GPIO LOW |
| `WEB_PORT` | `5000` | Flask web server port |

These settings can be adjusted according to detection accuracy, processing speed, and hardware requirements.

## 13. Run the Complete Application

Activate the virtual environment and start the application:

```bash
cd ~/person_detector
source .venv/bin/activate
python -u app.py
```

Find the Raspberry Pi's IP address:

```bash
hostname -I
```

Open the dashboard from a device on the same network:

```text
http://<RASPBERRY_PI_IP>:5000
```

Replace `<RASPBERRY_PI_IP>` with the current IP address of your Raspberry Pi.

## 14. Web Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Main web dashboard |
| `/video_feed` | Live MJPEG video stream |
| `/status` | JSON detection and GPIO status |

Example `/status` response:

```json
{
  "person_detected": true,
  "confidence": 0.63,
  "inference_ms": 182.0,
  "fps": 4.0,
  "gpio17": "HIGH",
  "error": ""
}
```

The values shown are illustrative. Actual confidence, inference time, and FPS depend on the hardware and scene.

## 15. Performance Considerations

The Raspberry Pi Zero 2 W has limited processing power and memory. The project uses a quantized model, a low camera resolution, a small number of inference threads, and a capped inference rate to reduce resource usage.

Performance can vary with:

- Lighting and camera placement.
- Confidence threshold.
- Camera resolution.
- CPU temperature and power stability.
- Model execution time.
- Network conditions during streaming.

The application detects people within the camera's field of view. It does not guarantee that a room is empty when no person is detected. Occlusion, poor lighting, and temporary detection failures can cause incorrect absence decisions.

## 16. Troubleshooting

### Camera Is Busy

Example error:

```text
RuntimeError: Failed to acquire camera: Device or resource busy
```

**Solution:** Stop other programs using the camera, including `rpicam-hello`, `rpicam-vid`, or another Python camera script. Run only one camera application at a time.

### Model File Not Found

Check the file:

```bash
ls -lh ~/person_detector/models/detect.tflite
```

Ensure the model path and filename match the application configuration.

### TensorFlow Lite Runtime Import Error

Run:

```bash
python --version
python -c "import platform; print(platform.machine())"
python -m pip show tflite-runtime
```

Verify that the virtual environment is active and that the installed package supports the current Python version and architecture.

### GPIO Busy or Permission Error

Check whether another program is using GPIO17. Stop any previous GPIO test before launching the application.

### Dashboard Cannot Be Opened

Check the current IP address:

```bash
hostname -I
```

Ensure the Flask server is running and that the client device can communicate with the Pi over port 5000.

### Unexpected Restarts

Inspect the power supply, USB cable, connectors, and camera ribbon cable. An unstable power connection can cause unexpected shutdowns or restarts.

### GPIO Switches OFF During Brief Detection Failures

Increase `NO_PERSON_DELAY` to require a longer period without detections before the output switches LOW.

## 17. Security Considerations

The Flask development server is intended for local testing and development.

- Do not expose port 5000 directly to the public internet.
- Use authentication and HTTPS if remote access is required.
- Keep the Raspberry Pi OS and application dependencies updated.
- Ensure that external loads have an appropriate safe default state.

## 18. Applications

- Automatic room lighting.
- Occupancy-triggered indicators.
- Smart room monitoring.
- Camera-based presence detection.
- Edge AI and IoT demonstrations.
- Low-cost computer vision automation.

## 19. Future Improvements

- Add a manual GPIO override to the dashboard.
- Add configurable absence delay and confidence threshold controls.
- Integrate a relay module for suitable external loads.
- Add event logging and occupancy history.
- Implement temporal filtering or object tracking.
- Add authentication and HTTPS for remote access.
- Evaluate more efficient detection models for edge devices.

## 20. GitHub Setup

Create a `.gitignore` file:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
```

Avoid committing your virtual environment or private credentials.

Before publishing the repository, decide whether to include the model file. If the model's redistribution terms do not permit redistribution, leave it out and retain the download instructions in this README.

## 21. License

Choose a project license before publishing, such as the MIT License. Check the model's own license separately; the project's license does not automatically change the model's terms.

## 22. Acknowledgements

- Raspberry Pi Foundation — Hardware and software ecosystem.
- TensorFlow Lite — Lightweight machine learning inference.
- Picamera2 — Raspberry Pi camera integration.
- OpenCV — Image processing and visualization.
- Flask — Web application and streaming.
- gpiozero — Raspberry Pi GPIO control.
