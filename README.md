
# Appli-Smart: Person Detection and GPIO-Based Occupancy Control Using Raspberry Pi Zero 2 W

A real-time computer vision project that detects people using an IMX219 camera and a TensorFlow Lite object detection model on the Raspberry Pi Zero 2 W. Based on the detection result, the system automatically controls a GPIO output and provides a live web dashboard for monitoring.

## Overview

This project uses a Raspberry Pi Zero 2 W to detect whether a person is present in a room. The camera captures frames, a quantized MobileNet SSD model detects people, and the detection result determines the state of GPIO17.

A Flask-based web dashboard displays the live camera feed, detection status, confidence score, inference time, FPS, and GPIO output state.

The system is designed for lightweight edge AI applications where computer vision and hardware control need to run on a resource-constrained device.

## Features

- Real-time person detection using TensorFlow Lite.
- Raspberry Pi Zero 2 W deployment.
- IMX219 camera integration using Picamera2.
- Live camera streaming through a web dashboard.
- Automatic GPIO output control based on person detection.
- Configurable confidence threshold.
- Configurable delay before switching the output OFF.
- Displays detection confidence, inference latency, and FPS.
- Lightweight processing settings suitable for low-power hardware.
- Fail-safe software behavior that sets GPIO17 LOW when the detector worker stops.

## Hardware Requirements

| Component | Purpose |
|---|---|
| Raspberry Pi Zero 2 W | Runs the detection model and web server |
| IMX219 camera module | Captures the room video |
| microSD card | Stores the operating system, model, and code |
| Stable power supply | Powers the Raspberry Pi and camera |
| Relay module or transistor driver (optional) | Interfaces GPIO with an external load |
| External load (optional) | Light, indicator, or other controlled device |

## Software Requirements

- Raspberry Pi OS
- Python 3.11
- Picamera2
- OpenCV
- NumPy
- TensorFlow Lite Runtime
- Flask
- gpiozero

## System Architecture

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
                   |
                   v
       TensorFlow Lite Inference
        (MobileNet SSD / COCO)
                   |
                   v
          Person Detection Result
                   |
          +--------+--------+
          |                 |
          v                 v
   Person Detected     No Detection
          |                 |
          v                 v
      GPIO17 HIGH       Start Absence
       (3.3V)              Timer
                            |
                            v
                    Absence Delay Expires
                            |
                            v
                       GPIO17 LOW
                         (0V)

Camera Frames and Detection Status
                   |
                   v
             Flask Web Server
                   |
                   v
           Web Browser Dashboard
```

## How It Works

### 1. Camera Capture

The IMX219 camera captures frames using Picamera2. The configured camera resolution is 320 × 240 pixels to keep processing requirements manageable.

### 2. Image Preprocessing

Each frame is converted to the required color ordering and resized to match the TensorFlow Lite model input dimensions of 300 × 300 pixels.

### 3. Person Detection

The quantized MobileNet SSD model processes the image and produces detection boxes, class IDs, and confidence scores. The application considers a detection valid when the class corresponds to a person and the confidence is at least the configured threshold.

### 4. GPIO Control

The application controls BCM GPIO17 according to the detection result:

- **Person detected:** GPIO17 becomes HIGH.
- **Person not detected:** A configurable absence timer starts.
- **No person detected for the configured delay:** GPIO17 becomes LOW.

The delay helps prevent unnecessary switching when the model briefly misses a person.

### 5. Web Dashboard

A Flask server provides a browser-based dashboard with:

- Live annotated camera stream.
- Person presence status.
- GPIO17 state.
- Detection confidence.
- Inference time in milliseconds.
- Approximate detection FPS.
- Error and system status messages.

The dashboard can be opened from another device on the same local network.

## Repository Structure

```text
person_detector/
├── app.py
├── models/
│   └── detect.tflite
├── README.md
└── .venv/                 # Local virtual environment; do not commit
```

The TensorFlow Lite model is stored in the `models/` directory. The virtual environment is created locally and should be excluded from version control.

## Installation and Setup

### Step 1: Update the Raspberry Pi

```bash
sudo apt update
sudo apt upgrade -y
```

Install the system packages:

```bash
sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-venv \
    python3-gpiozero
```

### Step 2: Create the Project Directory

```bash
mkdir -p ~/person_detector/models
cd ~/person_detector
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### Step 4: Install Python Dependencies

```bash
pip install flask tflite-runtime
```

The `--system-site-packages` option allows the virtual environment to access system-installed packages such as Picamera2, OpenCV, NumPy, and gpiozero.

### Step 5: Add the Model

Place the compatible quantized MobileNet SSD TensorFlow Lite model at:

```text
~/person_detector/models/detect.tflite
```

The model must be compatible with the code's expected input and output tensors. The implementation assumes a 300 × 300 image input and the standard SSD outputs for bounding boxes, classes, and scores.

### Step 6: Add the Application Code

Save the complete application code as:

```text
~/person_detector/app.py
```

### Step 7: Run the Application

```bash
cd ~/person_detector
source .venv/bin/activate
python -u app.py
```

When startup is successful, the terminal displays the dashboard address.

Open the dashboard from a device on the same network:

```text
http://<RASPBERRY_PI_IP>:5000
```

Replace `<RASPBERRY_PI_IP>` with the current IP address of the Raspberry Pi.

To find the IP address, run:

```bash
hostname -I
```

## GPIO Connections

The application uses BCM GPIO17, corresponding to physical pin 11 on the Raspberry Pi header.

| Raspberry Pi Pin | Connection |
|---|---|
| GPIO17 (BCM 17) | Digital output |
| Physical pin 11 | GPIO17 |
| Physical pin 6 | GND, if required by the external interface |

### Output Logic

| Condition | GPIO17 State | Voltage Level |
|---|---|---|
| Person detected | HIGH | Approximately 3.3 V |
| No person detected beyond the delay | LOW | Approximately 0 V |

**Hardware safety:** Raspberry Pi GPIO pins operate at 3.3 V logic and cannot directly drive a relay coil or a high-current load. Use a compatible relay module or transistor driver. Do not apply 5 V to a GPIO pin.

## Configuration

The main parameters are defined near the top of `app.py`.

| Parameter | Default | Description |
|---|---:|---|
| `THRESHOLD` | `0.50` | Minimum detection confidence |
| `CAMERA_WIDTH` | `320` | Camera frame width |
| `CAMERA_HEIGHT` | `240` | Camera frame height |
| `NUM_THREADS` | `2` | TensorFlow Lite inference threads |
| `TARGET_INFERENCE_FPS` | `4.0` | Target processing rate |
| `JPEG_QUALITY` | `55` | JPEG stream quality |
| `PERSON_PIN` | `17` | BCM GPIO output pin |
| `NO_PERSON_DELAY` | `3.0` seconds | Delay before output switches LOW |
| `WEB_PORT` | `5000` | Flask server port |

These are initial settings and can be adjusted according to detection accuracy, processing speed, and hardware requirements.

## Web Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Web dashboard |
| `/video_feed` | Live MJPEG video stream |
| `/status` | JSON status containing detection, confidence, latency, FPS, GPIO state, and errors |

Example status response:

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

The numbers above are illustrative; actual performance depends on the device, model, and scene.

## Performance Considerations

The Raspberry Pi Zero 2 W has limited memory and processing resources. The application uses a lightweight quantized model and reduced camera resolution to limit computational overhead.

Performance can be affected by:

- Lighting and camera placement.
- Model confidence threshold.
- Camera resolution.
- Number of inference threads.
- CPU temperature and power stability.
- Network quality during video streaming.

A brief missed detection can occur when a person is occluded or leaves the camera's field of view. The absence delay reduces output flickering but does not guarantee perfect room-occupancy detection.

## Troubleshooting

### 1. Camera Is Busy

Error example:

```text
RuntimeError: Failed to acquire camera: Device or resource busy
```

**Possible solution:**

Close other programs using the camera, such as `rpicam-hello`, `rpicam-vid`, or another Python camera script. Run only one camera application at a time.

### 2. Model File Not Found

Verify that the model exists:

```bash
ls -lh ~/person_detector/models/detect.tflite
```

Ensure the filename and path match the configuration in `app.py`.

### 3. GPIO Busy or Permission Error

Check whether another application is using GPIO17. Stop the previous GPIO test before launching the application.

### 4. Dashboard Cannot Be Opened

Check the Pi's IP address:

```bash
hostname -I
```

Check that the application is running and that port 5000 is reachable from the client device. Both devices should be on a network that permits communication between them.

### 5. Raspberry Pi Restarts Unexpectedly

Inspect the power supply, USB power cable, connectors, and camera ribbon cable. An unstable power connection can cause unexpected shutdowns or restarts.

### 6. GPIO Switches OFF During Brief Detection Failures

Increase `NO_PERSON_DELAY` to allow a longer interval without a detection before switching the output LOW.

## Limitations

- The system detects people only within the camera's effective field of view.
- Occlusion, poor lighting, and camera positioning can reduce accuracy.
- A missed detection lasting longer than the configured delay may cause a false absence result.
- The output control is based on visual detections, not guaranteed long-term occupancy tracking.
- If the Raspberry Pi loses power, software cannot guarantee the external load's safe state.
- The Flask development server is intended for local development and testing, not direct public internet deployment.

## Applications

- Automatic room lighting.
- Occupancy-triggered indicators.
- Smart room monitoring.
- Camera-based presence detection.
- Edge AI demonstrations.
- GPIO automation experiments.
- Low-cost computer vision and IoT prototypes.

## Future Improvements

- Add a manual override switch to the dashboard.
- Add configurable detection thresholds and absence delay controls.
- Integrate a relay module for lighting or other suitable loads.
- Add persistent occupancy state and event logging.
- Improve robustness using temporal filtering or tracking.
- Add authentication and HTTPS for remote access.
- Explore more efficient object detection models for edge devices.

## Contributing

Contributions, bug reports, and suggestions are welcome. For substantial changes, create a branch and submit a pull request with a description of the modification and testing performed.

## License

Choose a license before publishing the repository, such as the MIT License. Ensure that the model's license and redistribution terms are compatible with your project before including its weights in the repository.

## Acknowledgements

- Raspberry Pi Foundation — Raspberry Pi hardware and software ecosystem.
- TensorFlow Lite — Lightweight machine learning inference.
- Picamera2 — Raspberry Pi camera integration.
- OpenCV — Image processing and visualization.
- Flask — Web application and video streaming.
- gpiozero — Raspberry Pi GPIO control.
