from flask import Flask, Response, render_template, request, jsonify # type: ignore
import cv2
from ultralytics import YOLO
import torch
import os

app = Flask(__name__)

# YOLO and TTS setup
confidence_threshold = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

# Global variables for detection
is_detecting = False
cap = None  # Global variable to hold video capture object
detected_objects_global = set()  # Global set to store detected objects

def generate_frames():
    """Real-time frame processing with YOLO and object detection."""
    global last_speech_time, speech_interval, is_detecting, cap, detected_objects_global

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Start capturing from the webcam
        if not cap.isOpened():
            print("Error: Cannot open webcam.")
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = set()

        if is_detecting:
            # Process frame for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        label = model.names[class_id]

                        if confidence >= confidence_threshold:
                            detected_objects.add(label)

            # Update global detected objects
            detected_objects_global = detected_objects

            # Annotate frame
            frame = results[0].plot()

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cap = None  # Reset the webcam capture after the loop

@app.route('/')
def index():
    """Render the HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for real-time video feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Start or stop detection."""
    global is_detecting
    is_detecting = not is_detecting
    status = "started" if is_detecting else "stopped"
    return jsonify({"status": status})

@app.route('/get_detected_objects', methods=['GET'])
def get_detected_objects():
    """Fetch the currently detected objects."""
    return jsonify({"objects": list(detected_objects_global)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
