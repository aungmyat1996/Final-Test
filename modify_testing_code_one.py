import cv2
from ultralytics import YOLO
import time

# Initialize YOLOv8s model for car detection (YOLOv8s is a smaller, faster version)
model = YOLO('yolov8s.pt')  # Download the model weights from ultralytics

# Initialize video capture (use 0 for webcam or path to video file)
cap = cv2.VideoCapture('track_car.mp4')

# Initialize tracker (we'll use CSRT for more accurate tracking)
trackers = []
tracking = False

# Start time and frame count for FPS calculation
start_time = time.time()
frame_count = 0

def calculate_fps(start_time, frame_count):
    """ Calculate and update FPS. """
    current_time = time.time()
    fps = frame_count / (current_time - start_time) if current_time > start_time else 0
    return fps

def draw_hud(frame, fps):
    """ Draw HUD with FPS and other telemetry. """
    h, w = frame.shape[:2]
    text_props = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.5, "color": (0, 255, 0), "thickness": 1}
    
    # Telemetry info
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), **text_props)
    cv2.putText(frame, "Car Tracking Active", (20, 60), **text_props)
    # Add more telemetry data as needed
    cv2.putText(frame, "Speed: 45 km/h", (w - 150, 30), **text_props)
    cv2.putText(frame, "Altitude: 200 m", (w - 150, 60), **text_props)
    cv2.putText(frame, "Battery: 80%", (w - 150, 90), **text_props)

def process_yolo_detections(frame, results):
    """ Detect cars using YOLOv8 and initialize trackers for each detected car. """
    global trackers, tracking
    trackers = []  # Reset the list of trackers

    try:
        detections = results[0].boxes  # Adjust this based on the structure of results
    except AttributeError:
        print("Error: Unable to access 'boxes'. Printing results for debugging:")
        print(results)
        return

    for box in detections:
        cls = int(box.cls[0].item())  # Class label
        if model.names[cls] == 'car':  # Check if the detected object is a car
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Extract bounding box coordinates

            # Initialize a CSRT tracker for each detected car
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            trackers.append(tracker)
    
    tracking = len(trackers) > 0

def update_trackers(frame):
    """ Update all trackers and draw bounding boxes for tracked cars. """
    global trackers
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    global tracking
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if not tracking:
            # Perform YOLOv8 detection when not tracking
            results = model(frame)
            process_yolo_detections(frame, results)

        # Update all trackers
        if tracking:
            update_trackers(frame)

        # Calculate FPS
        fps = calculate_fps(start_time, frame_count)

        # Draw the HUD (telemetry, FPS, etc.)
        draw_hud(frame, fps)

        # Display the frame with tracking and HUD
        cv2.imshow('YOLOv8 Car Tracking with HUD UI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
