import cv2
from ultralytics import YOLO
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

# Initialize YOLOv8 model for person/car detection
model = YOLO('yolov8s.pt')

# Connect to the drone (UDP connection)
CONNECTION_STRING = 'udp:192.168.1.26:14550'
vehicle = connect(CONNECTION_STRING, wait_ready=True)

# Global variable to store target confirmation status
target_confirmed = False

# Function to arm the drone and take off
def arm_and_takeoff(target_altitude=30):
    print("Arming motors and taking off to search altitude...")
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialize...")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print(f"Taking off to {target_altitude} meters!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        print(f"Altitude: {vehicle.location.global_relative_frame.alt:.2f} meters")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            print(f"Reached target altitude: {target_altitude} meters")
            break
        time.sleep(1)

    print("Starting auto-search for targets...")

# Perform target detection using YOLOv8
def detect_target(frame):
    results = model(frame)
    detections = results[0].boxes

    # Process detections and return the first detected object
    for box in detections:
        cls = int(box.cls[0].item())
        if model.names[cls] in ['car', 'person']:  # Detect cars or people
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Bounding box coordinates
            return (x1, y1, x2, y2)  # Return the bounding box of the detected target
    return None  # No target detected

# Function to move the drone based on detected target's position
def move_drone_based_on_position(obj_x, obj_y, center_x, center_y):
    x_offset = obj_x - center_x
    y_offset = obj_y - center_y
    x_threshold = 20
    y_threshold = 20

    # Reduced Yaw Control (Channel 4)
    if abs(x_offset) > x_threshold:
        if x_offset < 0:  # Object is left of center, turn left (yaw)
            vehicle.channels.overrides[4] = 1500 - min(50 + abs(x_offset) // 10, 100)  # Reduced yaw adjustment
            print(f"Turning left (yaw), offset: {x_offset}")
        else:  # Object is right of center, turn right (yaw)
            vehicle.channels.overrides[4] = 1500 + min(50 + abs(x_offset) // 10, 100)  # Reduced yaw adjustment
            print(f"Turning right (yaw), offset: {x_offset}")
    else:
        vehicle.channels.overrides[4] = 1500  # Keep yaw steady
        print("Yaw steady")

    # Pitch Control (Channel 2) - Forward/backward
    if abs(y_offset) > y_threshold:
        if y_offset < 0:  # Object is above center, move forward
            vehicle.channels.overrides[2] = 1500 - min(50 + abs(y_offset) // 10, 100)
            print(f"Moving forward (pitch), offset: {y_offset}")
        else:  # Object is below center, move backward
            vehicle.channels.overrides[2] = 1500 + min(50 + abs(y_offset) // 10, 100)
            print(f"Moving backward (pitch), offset: {y_offset}")
    else:
        vehicle.channels.overrides[2] = 1500  # Keep pitch steady
        print("Pitch steady")

# Function to handle confirmation of the target from GCS
def confirm_target():
    global target_confirmed
    target_confirmed = True
    print("Target confirmed. Starting tracking.")

# Function to track the target after confirmation
def track_target(frame):
    global target_confirmed

    # Wait for GCS confirmation to start tracking
    if not target_confirmed:
        print("Waiting for target confirmation from GCS...")
        return

    # Detect target
    target_box = detect_target(frame)

    if target_box is not None:
        x1, y1, x2, y2 = target_box
        obj_x = (x1 + x2) // 2  # Center of the detected object (car/person)
        obj_y = (y1 + y2) // 2

        # Move the drone based on the target's position in the frame
        move_drone_based_on_position(obj_x, obj_y, frame.shape[1] // 2, frame.shape[0] // 2)
    else:
        print("No target detected. Holding position.")
        vehicle.channels.overrides = {'2': 1500, '3': 1500, '4': 1500}  # Steady position

# Main function for GCS-based target confirmation and tracking
def main():
    # Takeoff to a height of 30 meters for auto-search
    arm_and_takeoff(30)

    # Initialize video capture for target detection
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform target detection and wait for confirmation from GCS
        track_target(frame)

        # Display the frame (for testing and visualization)
        cv2.imshow('Target Tracking with GCS Confirmation', frame)

        # Simulate user confirming the target via GCS
        if cv2.waitKey(1) & 0xFF == ord('c'):  # Press 'c' to confirm target in GCS
            confirm_target()

        # Exit on key press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After tracking is done, return to launch
    vehicle.mode = VehicleMode("RTL")  # Return to Launch mode

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Closing vehicle connection")
        vehicle.close()
